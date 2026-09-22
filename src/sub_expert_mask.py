"""Model-agnostic gradient mask register for selective sub-expert training.

This module provides :class:`SubExpertMaskRegister`, which freezes all model
parameters except selected sub-expert channels and registers gradient hooks
that apply a mask to gradients.  When ``gate_scores`` are provided, the mask
is an entropy-based adaptive gradient scale; otherwise it is a binary mask.

The register supports any MoE architecture (OLMoE, Mixtral, Qwen2-MoE, etc.)
by configuring the expert parameter pattern.

Compatible with DeepSpeed ZeRO-3.

Example usage (binary mask)::

    from sub_expert_mask import SubExpertMaskRegister

    register = SubExpertMaskRegister(
        model=model,
        sub_experts_config=selected_sub_experts,  # from select_top_sub_experts.py
        num_layers=model.config.num_hidden_layers,
        num_experts=model.config.num_experts,
        intermediate_size=model.config.intermediate_size,
    )

    # Apply: freeze params, unfreeze selected, register mask hooks
    num_trainable = register.apply()

    # Get trainable params for optimizer
    trainable_params = register.get_trainable_params()

    # ... training loop ...

    # Cleanup after training
    register.cleanup()

Example with gradient scaling::

    from sub_expert_mask import SubExpertMaskRegister

    # Load gate scores from cal_sub_router_chat.py output
    import torch
    gate_dist = torch.load("output/gate_dist.pt")

    register = SubExpertMaskRegister(
        model=model,
        sub_experts_config=selected_sub_experts,
        num_layers=model.config.num_hidden_layers,
        num_experts=model.config.num_experts,
        intermediate_size=model.config.intermediate_size,
        gate_scores=gate_dist,       # Enable entropy-based gradient scaling
        scale_grad_max=5.0,          # Maximum scale factor
        group_size=1,                # Per-channel scaling
    )

    num_trainable = register.apply()
    trainable_params = register.get_trainable_params()
    # ... training loop ...
    register.cleanup()
"""

from __future__ import annotations

import re
from typing import Any, Callable

import torch
import torch.nn as nn


class SubExpertMaskRegister:
    """Model-agnostic gradient mask register for selective sub-expert training.

    Freezes all model parameters except selected sub-expert channels, and
    registers gradient hooks that apply a (optionally entropy-scaled) mask to
    gradients.

    Supports any MoE architecture by configuring the expert parameter pattern.
    Compatible with DeepSpeed ZeRO-3.

    Attributes:
        model: The MoE model to apply masks to.
        sub_experts_config: Normalized config mapping layer_id (str) to
            ``{expert_id (str): [channel_ids]}``.
        num_layers: Number of layers in the model.
        num_experts: Number of experts per layer.
        intermediate_size: Intermediate dimension of expert MLPs.
        expert_re: Compiled regex to extract layer_id and expert_id from
            parameter names.
        proj_suffixes: Tuple of parameter name suffixes to match.
        device: Device for the mask buffer.
        gate_scores: Optional activation energy tensor of shape
            ``[num_layers, num_experts, intermediate_size]`` for entropy-based
            gradient scaling.
        scale_grad_max: Maximum gradient scale factor.
        group_size: Number of channels per scaling group.
        scale_buffer: Reference to ``model.moe_scale_buffer``, a gradient
            scale/mask tensor of shape
            ``[num_layers, num_experts, intermediate_size]``.  When
            ``gate_scores`` is provided this is an entropy-based adaptive
            scale; otherwise a binary mask.  Set by :meth:`apply` and
            registered on the model so that ``DynamicScaleCallback`` can
            update it in-place during training.
        hooks: List of registered gradient hook handles.
    """

    def __init__(
        self,
        model: nn.Module,
        sub_experts_config: dict[str, dict[str, list[int]]],
        num_layers: int,
        num_experts: int,
        intermediate_size: int,
        *,
        expert_pattern: str = r"layers\.(\d+)\.mlp\.experts\.(\d+)",
        proj_suffixes: tuple[str, ...] = (
            ".gate_proj.weight",
            ".up_proj.weight",
            ".down_proj.weight",
        ),
        device: str = "cuda",
        gate_scores: torch.Tensor | None = None,
        scale_grad_max: float = 5.0,
        group_size: int = 1,
    ) -> None:
        """Initialize the mask register.

        Args:
            model: The MoE model (e.g., AutoModelForCausalLM).
            sub_experts_config: Dictionary mapping layer_id (str) to
                {expert_id (str): [channel_ids]} selected for training.
            num_layers: Number of layers in the model.
            num_experts: Number of experts per layer.
            intermediate_size: Intermediate dimension of expert MLPs.
            expert_pattern: Regex pattern to extract layer_id and expert_id
                from parameter names. Default matches OLMoE/Mixtral/Qwen2-MoE.
            proj_suffixes: Tuple of parameter name suffixes to match for
                projection weights.
            device: Device for the scale/mask buffer.
            gate_scores: Optional tensor of shape
                ``[num_layers, num_experts, intermediate_size]`` loaded from
                ``cal_sub_router_chat.py``'s ``gate_dist.pt`` (i.e., the
                accumulated L1-normalized SiLU gate feature distribution).
                When provided, enables entropy-based adaptive gradient
                scaling instead of a binary mask.
            scale_grad_max: Maximum gradient scale factor. Default 5.0.
            group_size: Number of channels per group for scaling. Default 1
                (per-channel scaling).
        """
        self.model = model
        # Normalize config keys to strings for consistent lookup.
        self.sub_experts_config: dict[str, dict[str, list[int]]] = {
            str(k): {str(e): list(ch) for e, ch in v.items()}
            for k, v in sub_experts_config.items()
        }
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.expert_re = re.compile(expert_pattern)
        self.proj_suffixes = proj_suffixes
        self.device = device
        self.gate_scores = gate_scores
        self.scale_grad_max = scale_grad_max
        self.group_size = group_size
        self.scale_buffer: torch.Tensor | None = None
        self.hooks: list[Any] = []

    # ------------------------------------------------------------------
    # Scale buffer computation
    # ------------------------------------------------------------------

    def _compute_scale_buffer(self) -> torch.Tensor:
        """Compute gradient scale buffer with entropy-based per-channel scaling.

        When gate_scores is provided, each selected channel receives an
        adaptive gradient scale based on its relative activation energy.
        The scale is modulated by the entropy of the activation distribution:
        uniform distributions yield gamma ≈ 1.0 (equal scaling), while
        polarized distributions yield gamma ≈ 0.5 (emphasizing high-energy
        channels).

        Returns:
            Scale buffer tensor of shape [num_layers, num_experts, intermediate_size].
        """
        scale_buffer = torch.zeros(
            (self.num_layers, self.num_experts, self.intermediate_size),
            device=self.device,
            dtype=torch.float32,
        )

        # Normalize gate_scores over the last dimension
        gate_scores = self.gate_scores.to(self.device)
        gate_scores = gate_scores / (gate_scores.sum(dim=2, keepdim=True) + 1e-8)

        for layer_str, expert_dict in self.sub_experts_config.items():
            layer_idx = int(layer_str)
            for expert_str, channels in expert_dict.items():
                expert_idx = int(expert_str)
                if not channels:
                    continue

                # Get this expert's normalized gate scores
                expert_scores = gate_scores[layer_idx, expert_idx]  # [intermediate_size]

                # With group_size=1, each channel is its own group
                # Identify active groups (channels that are selected)
                num_groups = self.intermediate_size // self.group_size

                active_groups = []  # list of (group_start_index, group_energy)
                for g_idx in range(0, num_groups * self.group_size, self.group_size):
                    # Check if this group's starting channel is in the selected set
                    if g_idx in channels:
                        group_energy = expert_scores[
                            g_idx : g_idx + self.group_size
                        ].sum().item()
                        active_groups.append((g_idx, group_energy))

                num_active = len(active_groups)
                if num_active == 0:
                    continue

                # Total captured energy
                group_energies = [ge for _, ge in active_groups]
                total_rho = max(sum(group_energies), 1e-6)
                avg_energy = total_rho / num_active

                # global_base_scale = 1.0
                global_base_scale = 1.0

                # Compute entropy-based gamma
                group_energies_tensor = torch.tensor(
                    group_energies, device=self.device
                )
                normalized_energies = group_energies_tensor / (
                    group_energies_tensor.sum() + 1e-8
                )
                entropy = -torch.sum(
                    normalized_energies * torch.log(normalized_energies + 1e-8)
                )

                if num_active > 1:
                    max_entropy = torch.log(
                        torch.tensor(float(num_active), device=self.device)
                    )
                    gamma = 0.5 + 0.5 * entropy / max_entropy
                else:
                    gamma = 0.5

                # Compute per-group scale
                for g_start, g_energy in active_groups:
                    rel_importance = g_energy / max(avg_energy, 1e-8) ** gamma
                    g_scale = global_base_scale * rel_importance
                    g_scale = max(1.0, min(self.scale_grad_max, g_scale))
                    scale_buffer[
                        layer_idx,
                        expert_idx,
                        g_start : g_start + self.group_size,
                    ] = g_scale

                # Print statistics for this expert
                scales = [
                    max(
                        1.0,
                        min(
                            self.scale_grad_max,
                            ge / max(avg_energy, 1e-8) ** gamma,
                        ),
                    )
                    for _, ge in active_groups
                ]
                print(
                    f"Layer {layer_idx} Expert {expert_idx}: "
                    f"num_active={num_active}, total_rho={total_rho:.4f}, "
                    f"gamma={gamma:.4f}, "
                    f"scale range=[{min(scales):.4f}, {max(scales):.4f}]"
                )

        return scale_buffer

    # ------------------------------------------------------------------
    # Apply masks to model
    # ------------------------------------------------------------------

    def apply(self) -> int:
        """Apply gradient mask: freeze all, unfreeze selected, register hooks.

        When ``gate_scores`` is provided, builds an entropy-based adaptive
        gradient scale buffer where high-energy channels receive larger
        gradient steps.  Otherwise, builds a binary mask (1.0 for selected,
        0.0 for unselected).

        1. Freezes all model parameters.
        2. Unfreezes selected sub-expert projection weights.
        3. Builds the scale/mask buffer and registers it on the model as
           ``moe_scale_buffer`` (a persistent buffer) so that
           ``DynamicScaleCallback`` can update it in-place during training.
        4. Registers gradient hooks that apply the scale/mask.

        Returns:
            Number of trainable parameter tensors.
        """
        # 1. Build scale buffer.
        if self.gate_scores is not None:
            print("Computing entropy-based gradient scale buffer...")
            self.scale_buffer = self._compute_scale_buffer()
        else:
            # Binary mask: 1.0 for selected, 0.0 for unselected
            self.scale_buffer = torch.zeros(
                (self.num_layers, self.num_experts, self.intermediate_size),
                device=self.device,
                dtype=torch.float32,
            )
            for layer_str, expert_dict in self.sub_experts_config.items():
                layer_idx = int(layer_str)
                for expert_str, channels in expert_dict.items():
                    expert_idx = int(expert_str)
                    if channels:
                        self.scale_buffer[layer_idx, expert_idx, channels] = 1.0

        # 2. Register scale buffer on the model for dynamic scale updates.
        # DynamicScaleCallback can update this buffer in-place during training.
        if hasattr(self.model, "moe_scale_buffer"):
            self.model.moe_scale_buffer.copy_(self.scale_buffer)
        else:
            self.model.register_buffer(
                "moe_scale_buffer",
                self.scale_buffer.clone(),
                persistent=True,
            )
        # Keep a reference for convenience.
        self.scale_buffer = self.model.moe_scale_buffer

        # 3. DeepSpeed ZeRO-3 compatibility: mark buffer as unpartitioned.
        if hasattr(self.model.moe_scale_buffer, "ds_status"):
            self.model.moe_scale_buffer.ds_status = "unpartitioned"  # type: ignore[attr-defined]

        # 4. Freeze all parameters.
        self.model.requires_grad_(False)

        # 5. Define gradient hook factory (reads model.moe_scale_buffer so
        #    that DynamicScaleCallback updates are visible to the hook).
        def make_hook(
            layer_id: int, expert_id: int, is_down: bool
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            def hook(grad: torch.Tensor) -> torch.Tensor:
                mask_vec = self.model.moe_scale_buffer[layer_id, expert_id]
                if is_down:
                    # down_proj: [Hidden, Inter], mask on dim 1
                    mask_view = mask_vec.view(1, -1)
                else:
                    # gate_proj/up_proj: [Inter, Hidden], mask on dim 0
                    mask_view = mask_vec.view(-1, 1)
                return grad * mask_view.to(
                    dtype=grad.dtype, device=grad.device
                )
            return hook

        # 6. Iterate parameters, unfreeze selected, register hooks.
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            match = self.expert_re.search(name)
            if match is None:
                continue

            layer_str = match.group(1)
            expert_str = match.group(2)

            # Check if this (layer, expert) pair is selected.
            layer_config = self.sub_experts_config.get(layer_str)
            if layer_config is None or expert_str not in layer_config:
                continue

            # Check if the parameter name ends with a projection suffix.
            if not any(name.endswith(suffix) for suffix in self.proj_suffixes):
                continue

            # Auto-detect mask dimension based on parameter shape.
            if param.shape[0] == self.intermediate_size:
                is_down = False  # gate_proj/up_proj
            elif param.dim() >= 2 and param.shape[1] == self.intermediate_size:
                is_down = True  # down_proj
            else:
                # Shape does not match expected projection weight pattern.
                continue

            # Unfreeze the parameter.
            param.requires_grad = True

            # DeepSpeed ZeRO-3 compatibility.
            if hasattr(param, "ds_status"):
                param.ds_status = "unpartitioned"  # type: ignore[attr-defined]

            # Register gradient hook.
            l_idx = int(layer_str)
            e_idx = int(expert_str)
            handle = param.register_hook(make_hook(l_idx, e_idx, is_down))
            self.hooks.append(handle)

            # Mark as custom-masked for downstream tooling.
            param._is_custom_masked = True  # type: ignore[attr-defined]

            unfrozen_count += 1

        # 7. Print statistics.
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        percentage = (
            (trainable_params / total_params * 100) if total_params > 0 else 0.0
        )
        print(
            f"SubExpertMaskRegister: Unfroze {unfrozen_count} parameter "
            f"tensors."
        )
        print(
            f"SubExpertMaskRegister: Trainable parameters: "
            f"{trainable_params:,} ({percentage:.4f}% of {total_params:,} total)."
        )

        return unfrozen_count

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_lr_scales(self, lr_scale_max: float = 5.0) -> dict[str, float]:
        """Compute per-parameter learning rate scale factors.

        For each selected expert's projection weight, the LR scale is
        ``min(intermediate_size / num_selected_channels, lr_scale_max)``.
        This compensates for the reduced parameter count: when only a fraction
        of channels are trained, the learning rate is scaled up proportionally
        so that the effective gradient step size is comparable to full training.

        Args:
            lr_scale_max: Maximum LR scale factor to prevent explosion.
                Default 5.0.

        Returns:
            Dictionary mapping parameter names to LR scale factors.
            Parameters not in the dictionary use the default learning rate.
        """
        lr_scales: dict[str, float] = {}

        for name, param in self.model.named_parameters():
            match = self.expert_re.search(name)
            if match is None:
                continue

            l_str = match.group(1)
            e_str = match.group(2)

            layer_config = self.sub_experts_config.get(l_str)
            if layer_config is None or e_str not in layer_config:
                continue

            if not any(name.endswith(suffix) for suffix in self.proj_suffixes):
                continue

            channels = layer_config[e_str]
            num_active = len(channels)

            if 0 < num_active < self.intermediate_size:
                scale = min(self.intermediate_size / num_active, lr_scale_max)
                lr_scales[name] = scale

        return lr_scales

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return list of parameters with requires_grad=True.

        Returns:
            List of trainable parameters.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def cleanup(self) -> None:
        """Remove all registered gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self) -> None:
        """Safety net to remove hooks if cleanup was not called explicitly."""
        try:
            self.cleanup()
        except Exception:
            pass
