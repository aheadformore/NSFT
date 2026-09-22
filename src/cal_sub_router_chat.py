"""MoE Sub-expert Routing Dual Analyzer.

This module provides a dual-perspective analysis tool for Mixture-of-Experts
(MoE) models.  Beyond the router-logits-based expert usage statistics
provided by ``cal_router_chat.py``, this tool registers forward hooks on
every expert's ``gate_proj`` to monitor the internal SiLU-activated feature
distribution and verifies consistency between the two statistical
approaches.

Supported model architectures (accessed via
``model.model.layers[layer_idx].mlp.experts``):

    - OLMoE

Usage::

    python src/cal_sub_router_chat.py --model_path <path> \\
        --input_file <path> --save_path <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze expert activation in MoE model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the MoE model",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input jsonl file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum length of input sequence",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Number of top experts to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--test_number",
        type=int,
        default=0,
        help="Number of samples to process (0 for all)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="The directory to save the results",
    )
    return parser.parse_args()


class MoEDualAnalyzer:
    """Dual-perspective analyzer for MoE expert routing.

    Combines router-logits-based expert usage statistics with forward-hook
    monitoring of each expert's ``gate_proj`` projection to compute SiLU
    activated feature distributions and verify consistency between the two
    statistical approaches.

    Attributes:
        tokenizer: The model tokenizer.
        model: The pretrained MoE model.
        top_k: Number of top experts selected per token.
        num_layers: Number of hidden layers in the model.
        num_experts: Number of experts per layer.
        hidden_dim: Intermediate size of each expert's MLP.
        total_tokens: Cumulative token count processed.
        total_samples: Cumulative sample count processed.
        router_counts: Expert selection counts from router logits, shape
            ``(num_layers, num_experts)``.
        hook_counts: Expert invocation counts from forward hooks, shape
            ``(num_layers, num_experts)``.
        gate_accumulated_dist: Accumulated L1-normalized SiLU feature
            distribution per expert, shape
            ``(num_layers, num_experts, hidden_dim)``.
    """

    def __init__(
        self, model_path: str, top_k: int = 2, device: str = "cuda"
    ) -> None:
        """Initialize the analyzer by loading the model and registering hooks.

        Args:
            model_path: Path to the pretrained MoE model.
            top_k: Number of top experts to select per token.
            device: Device to load the model on (e.g. ``"cuda"`` or
                ``"cpu"``).
        """
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        self.top_k = top_k
        self.num_layers = self.model.config.num_hidden_layers
        self.num_experts = self.model.config.num_experts
        self.hidden_dim = self.model.config.intermediate_size

        self.total_tokens = 0
        self.total_samples = 0

        # --- Statistical accumulators ---
        # 1. Router-logits-based counts (N_router).
        self.router_counts = torch.zeros(
            (self.num_layers, self.num_experts), device="cpu",
            dtype=torch.long,
        )

        # 2. Gate-output (hook) based counts (N_hook).
        self.hook_counts = torch.zeros(
            (self.num_layers, self.num_experts), device="cpu",
            dtype=torch.long,
        )

        # 3. Accumulated intra-expert feature distribution (M).
        self.gate_accumulated_dist = torch.zeros(
            (self.num_layers, self.num_experts, self.hidden_dim),
            device="cpu",
            dtype=torch.float32,
        )

        self.hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on every expert's ``gate_proj``.

        Each hook captures the ``gate_proj`` output, applies SiLU activation,
        L1-normalizes along the feature dimension, and accumulates the
        distribution into ``gate_accumulated_dist``.

        Supported model architectures:

            - Mixtral
            - Qwen2-MoE

        These architectures expose experts via
        ``model.model.layers[layer_idx].mlp.experts``.
        """
        for layer_idx in range(self.num_layers):
            # Adapt to Mixtral / Qwen2-MoE structure.
            experts = self.model.model.layers[layer_idx].mlp.experts
            for expert_idx in range(self.num_experts):
                target_layer = experts[expert_idx].gate_proj

                def get_hook(l_idx: int, e_idx: int) -> Callable[..., None]:
                    def hook_fn(
                        module: torch.nn.Module,
                        input: Any,
                        output: torch.Tensor,
                    ) -> None:
                        # SiLU activation.
                        activated = F.silu(output)
                        n_tokens = activated.shape[0]
                        if n_tokens == 0:
                            return

                        # L1 normalization over the feature dimension (M).
                        normed = F.normalize(
                            activated.float().abs(), p=1, dim=-1
                        )

                        # Update hook statistics.
                        self.hook_counts[l_idx, e_idx] += n_tokens
                        self.gate_accumulated_dist[l_idx, e_idx] += (
                            normed.sum(dim=0).cpu()
                        )

                    return hook_fn

                self.hooks.append(
                    target_layer.register_forward_hook(
                        get_hook(layer_idx, expert_idx)
                    )
                )

    def process_data(
        self, jsonl_path: str, max_length: int = 8192, test_number: int = 0
    ) -> None:
        """Process chat data and accumulate expert routing statistics.

        Reads a JSONL file where each line is a JSON object containing a
        ``messages`` field.  Messages are formatted with the chat template,
        tokenized with truncation to ``max_length``, and fed through the
        model to collect router logits.

        Args:
            jsonl_path: Path to the input JSONL file.
            max_length: Maximum sequence length for tokenization truncation.
            test_number: Number of samples to process (0 for all).
        """
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Inference"):
                msg = json.loads(line)["messages"]

                # Apply chat template to text, then tokenize with
                # truncation to prevent OOM on long sequences.
                text = self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(
                        **inputs, output_router_logits=True
                    )

                # --- Router-logits-based statistics ---
                # router_logits is a tuple; each layer has shape
                # [batch, seq_len, num_experts].
                for l_idx, logits in enumerate(outputs.router_logits):
                    # Flatten batch and seq dims:
                    # [total_tokens, num_experts].
                    flat_logits = logits.view(-1, self.num_experts)

                    # Apply top-k selection logic.
                    routing_weights = F.softmax(
                        flat_logits, dim=-1, dtype=torch.float
                    )
                    _, selected_experts = torch.topk(
                        routing_weights, self.top_k, dim=-1
                    )

                    # Count occurrences of each expert.
                    # selected_experts shape: [total_tokens, top_k].
                    expert_indices, counts = torch.unique(
                        selected_experts, return_counts=True
                    )
                    for e_idx, count in zip(expert_indices, counts):
                        self.router_counts[l_idx, e_idx] += count.item()

                self.total_samples += 1
                self.total_tokens += inputs.input_ids.shape[-1]
                if test_number > 0 and self.total_samples >= test_number:
                    break

    def report(self) -> None:
        """Print a detailed consistency report for every layer-expert pair.

        Displays the router-logits count (N_Router), the hook count (N_Hook),
        the summed feature distribution (Sum(M)), and whether the two
        consistency checks pass.
        """
        print("\n" + "=" * 80)
        print(
            f"{'Layer-Exp':<12} | {'N_Router':<10} | {'N_Hook':<10} "
            f"| {'Sum(M)':<10} | {'Match?'}"
        )
        print("-" * 80)

        for l in range(self.num_layers):
            for e in range(self.num_experts):
                n_router = self.router_counts[l, e].item()
                n_hook = self.hook_counts[l, e].item()
                m_sum = self.gate_accumulated_dist[l, e].sum().item()

                # Check 1: router count equals hook count.
                is_match_n = n_router == n_hook
                # Check 2: Sum(M) equals N (normalization consistency).
                is_match_m = (
                    abs(m_sum - n_hook) < 1e-2 if n_hook > 0 else True
                )

                status = "✅" if (is_match_n and is_match_m) else "❌"

                if n_router > 0:  # Only print selected experts.
                    print(
                        f"L{l:02d}-E{e:02d}      | {int(n_router):<10} | "
                        f"{int(n_hook):<10} | {m_sum:<10.2f} | {status}"
                    )

    def check(self) -> None:
        """Assert consistency between router and hook statistics.

        Raises:
            AssertionError: If router counts do not match hook counts, or
                if the summed feature distribution deviates from the hook
                count beyond a tolerance of 0.5.
        """
        for l in range(self.num_layers):
            for e in range(self.num_experts):
                n_router = self.router_counts[l, e].item()
                n_hook = self.hook_counts[l, e].item()
                m_sum = self.gate_accumulated_dist[l, e].sum().item()

                # Check 1: router count equals hook count.
                is_match_n = n_router == n_hook
                # Check 2: Sum(M) equals N (normalization consistency).
                is_match_m = (
                    abs(m_sum - n_hook) < 0.5 if n_hook > 0 else True
                )
                assert is_match_n and is_match_m, (
                    f"Layer {l}, Expert {e}: N_Router={n_router}, "
                    f"N_Hook={n_hook}, Sum(M)={m_sum}"
                )

        print("All checks passed!!")

    def cleanup(self) -> None:
        """Remove all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __del__(self) -> None:
        """Safety net to remove hooks if cleanup was not called explicitly."""
        try:
            self.cleanup()
        except Exception:
            pass

    def save_results(self, save_path: str) -> None:
        """Save analysis results to the specified directory.

        Outputs:

            - ``experts-usage.png``: Expert activation ratio heatmap.
            - ``experts-usage.json``: Numeric activation ratios and totals.
            - ``gate_dist.pt``: Accumulated gate feature distribution tensor.
            - ``gate_heatmaps/layer_*.png``: Per-layer feature heatmaps.

        Args:
            save_path: Directory to save the results.
        """
        # Create output directory if needed.
        os.makedirs(save_path, exist_ok=True)

        # Calculate expert activation ratios.
        expert_activation_ratios = (
            self.router_counts.float() / self.total_tokens / self.top_k
        )

        # 1. Save expert activation ratio heatmap.
        expert_ratios_np = expert_activation_ratios.numpy()
        plt.figure(figsize=(12, 8))
        heatmap = plt.imshow(
            expert_ratios_np, cmap="viridis", aspect="auto", vmin=0, vmax=1
        )
        cbar = plt.colorbar(heatmap, label="Activation Ratio")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0.0", "1.0"])
        plt.xlabel("Expert Index")
        plt.ylabel("Layer Index")
        plt.title("Expert Activation Ratio per Layer")
        plt.tight_layout()
        heatmap_file = os.path.join(save_path, "experts-usage.png")
        plt.savefig(heatmap_file, dpi=300)
        plt.close()

        # 2. Save expert activation ratio numeric results.
        expert_ratios_file = os.path.join(save_path, "experts-usage.json")
        results_data = {
            "total_samples": self.total_samples,
            "total_tokens": self.total_tokens,
            "expert_usage_ratios": expert_activation_ratios.tolist(),
        }

        with open(expert_ratios_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Json results saved to {expert_ratios_file}")

        # 3. Save gate_accumulated_dist tensor.
        torch.save(
            self.gate_accumulated_dist,
            os.path.join(save_path, "gate_dist.pt"),
        )

        # 4. Plot per-layer gate feature heatmaps.
        # The feature dimension M may be large (e.g. 4096); heatmaps are
        # downsampled for performance.
        os.makedirs(
            os.path.join(save_path, "gate_heatmaps"), exist_ok=True
        )
        for l_idx in range(self.num_layers):
            layer_dist = self.gate_accumulated_dist[l_idx].numpy()
            if np.sum(layer_dist) == 0:
                continue

            plt.figure(figsize=(15, 5))
            # Downsample feature dimension to keep image manageable.
            step = max(1, self.hidden_dim // 1024)
            plt.imshow(
                layer_dist[:, ::step], cmap="viridis", aspect="auto"
            )
            plt.title(
                f"Layer {l_idx} Expert Feature Distribution (Summed M)"
            )
            plt.xlabel(f"Feature Dimension (Downsampled x{step})")
            plt.ylabel("Expert ID")
            plt.savefig(
                os.path.join(
                    save_path,
                    "gate_heatmaps",
                    f"layer_{l_idx}_feature_dist.png",
                )
            )
            plt.close()

        print(f"All results saved to: {save_path}")


def plot_expert_usage_ratios_and_gate_heatmaps(
    results_data: dict, gate_dist: torch.Tensor, save_path: str
) -> None:
    """Re-plot expert usage ratios and gate heatmaps from saved results.

    This standalone utility regenerates visualizations from previously
    saved ``experts-usage.json`` and ``gate_dist.pt`` files.

    Args:
        results_data: Dictionary containing ``expert_usage_ratios``.
        gate_dist: Accumulated gate distribution tensor of shape
            ``(num_layers, num_experts, hidden_dim)``.
        save_path: Directory where plots will be written (must contain or
            create a ``gate_heatmaps`` subfolder).
    """
    # 1. Plot expert activation ratio heatmap.
    expert_ratios_np = np.array(results_data["expert_usage_ratios"])
    plt.figure(figsize=(12, 8))
    heatmap = plt.imshow(
        expert_ratios_np, cmap="viridis", aspect="auto", vmin=0, vmax=1
    )
    cbar = plt.colorbar(heatmap, label="Activation Ratio")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0.0", "1.0"])
    plt.xlabel("Expert Index")
    plt.ylabel("Layer Index")
    plt.title("Expert Activation Ratio per Layer")
    plt.tight_layout()
    heatmap_file = os.path.join(save_path, "experts-usage.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()

    # The feature dimension M may be large (e.g. 4096); heatmaps are
    # downsampled for performance.
    os.makedirs(os.path.join(save_path, "gate_heatmaps"), exist_ok=True)
    for l_idx in range(gate_dist.shape[0]):
        layer_dist = gate_dist[l_idx].numpy()
        if np.sum(layer_dist) == 0:
            continue

        plt.figure(figsize=(15, 5))
        # Downsample feature dimension to keep image manageable.
        step = max(1, gate_dist.shape[-1] // 1024)
        heatmap = plt.imshow(
            layer_dist[:, ::step], cmap="viridis", aspect="auto"
        )
        cbar = plt.colorbar(heatmap, label="Activation Ratio")
        plt.title(
            f"Layer {l_idx} Expert Feature Distribution (Summed M)"
        )
        plt.xlabel(f"Feature Dimension (Downsampled x{step})")
        plt.ylabel("Expert ID")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                save_path,
                "gate_heatmaps",
                f"layer_{l_idx}_feature_dist.png",
            )
        )
        plt.close()


def main(args: argparse.Namespace) -> None:
    """Run the sub-expert routing dual analysis pipeline.

    If results already exist in ``args.save_path``, regenerates the plots
    from saved data.  Otherwise, loads the model, processes the data,
    validates consistency, and saves the results.

    Args:
        args: Parsed command-line arguments.
    """
    # If results already exist, just re-plot from saved data.
    if os.path.exists(args.save_path):
        print(
            f"Results already exist in {args.save_path}, "
            "loading and plotting..."
        )
        expert_ratios_file = os.path.join(
            args.save_path, "experts-usage.json"
        )
        gate_dist_file = os.path.join(args.save_path, "gate_dist.pt")
        if os.path.exists(expert_ratios_file) and os.path.exists(
            gate_dist_file
        ):
            with open(expert_ratios_file, "r") as f:
                results_data = json.load(f)

            gate_dist = torch.load(gate_dist_file)

            # Re-plot expert activation ratio heatmap and gate heatmaps.
            plot_expert_usage_ratios_and_gate_heatmaps(
                results_data, gate_dist, args.save_path
            )
        else:
            print("Required result files not found, skipping re-plotting.")
        return

    # Run the full analysis pipeline.
    analyzer = MoEDualAnalyzer(
        args.model_path, top_k=args.top_k, device=args.device
    )
    try:
        analyzer.process_data(
            args.input_file,
            max_length=args.max_length,
            test_number=args.test_number,
        )

        # Print detailed statistics before running assertions.
        analyzer.report()

        # Validate consistency between router and hook statistics.
        analyzer.check()

        # Print total token summary.
        print(
            f"Total tokens: {analyzer.total_tokens}, "
            f"calculated by hook counts: "
            f"{analyzer.hook_counts.sum().float() / args.top_k / analyzer.num_layers}"
        )

        # Save the results.
        analyzer.save_results(args.save_path)
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
