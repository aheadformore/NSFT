"""Select top sub-experts for MoE models using the group_continuous algorithm.

This script loads expert router ratios and gate distribution data, computes
combined gate scores, and selects sub-expert groups whose cumulative score
reaches a specified threshold.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace containing input_path, router_threshold,
        and group_size.
    """
    parser = argparse.ArgumentParser(
        description="Select top sub-experts for MoE models using group_continuous mode"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the directory containing experts-usage.json and gate_dist.pt",
    )
    parser.add_argument(
        "--router_threshold",
        type=float,
        default=0.2,
        help="Threshold for cumulative sum selection. [Default: 0.2]",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=16,
        help="Number of sub-experts per group. [Default: 16]",
    )
    return parser.parse_args()


def load_data(input_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load expert router ratios and gate distribution from disk.

    Args:
        input_path: Directory containing ``experts-usage.json`` and
            ``gate_dist.pt``.

    Returns:
        A tuple of (expert_router_ratios, gate_dist) where
        expert_router_ratios has shape [layers, experts] and gate_dist
        has shape [layers, experts, sub_experts].

    Raises:
        ValueError: If either required file does not exist.
    """
    expert_ratios_file = os.path.join(input_path, "experts-usage.json")
    gate_dist_file = os.path.join(input_path, "gate_dist.pt")

    if not (os.path.exists(expert_ratios_file) and os.path.exists(gate_dist_file)):
        raise ValueError(
            "Expert ratios file or gate distribution file does not exist"
        )

    with open(expert_ratios_file, "r") as f:
        data = json.load(f)

    expert_router_ratios = torch.Tensor(np.array(data["expert_usage_ratios"]))
    gate_dist = torch.load(gate_dist_file)

    return expert_router_ratios, gate_dist


def combined_gate_scores(
    expert_router_ratios: torch.Tensor,
    gate_dist: torch.Tensor,
) -> torch.Tensor:
    """Compute combined gate scores from router ratios and gate distribution.

    The expert router ratios (shape [layers, experts]) are expanded to
    [layers, experts, 1] and multiplied with the normalised gate
    distribution (shape [layers, experts, sub_experts]).

    Args:
        expert_router_ratios: Tensor of shape [layers, experts].
        gate_dist: Tensor of shape [layers, experts, sub_experts].

    Returns:
        Combined gate scores tensor of shape [layers, experts, sub_experts].
    """
    # Expand router ratios to [layers, experts, 1] for broadcasting
    expert_router_ratios_expanded = expert_router_ratios.unsqueeze(2)
    # Normalise the last dimension to avoid division by zero
    gate_dist = gate_dist / (gate_dist.sum(dim=2, keepdim=True) + 1e-6)
    # Compute combined gate scores
    combined = expert_router_ratios_expanded * gate_dist
    return combined


def select_sub_experts(
    combined_gate_scores: torch.Tensor,
    router_threshold: float,
    group_size: int = 16,
) -> dict[int, dict[int, list[int]]]:
    """Select sub-experts using the group_continuous algorithm.

    For each layer, sub-experts are grouped by ``group_size``. Group sums are
    sorted in descending order and accumulated until the cumulative sum
    exceeds ``router_threshold``. Selected groups are then converted back
    to ``{expert_id: [sub_expert_ids]}`` format.

    Args:
        combined_gate_scores: Tensor of shape [layers, experts, sub_experts].
        router_threshold: Cumulative score threshold for selection.
        group_size: Number of sub-experts per group.

    Returns:
        Dictionary mapping layer index to a dictionary of
        ``{expert_id: [sub_expert_ids]}``.
    """
    layers, experts, sub_experts = combined_gate_scores.shape
    selected_experts: dict[int, dict[int, list[int]]] = {}
    total_cum_sum = 0.0

    for layer_idx in range(layers):
        layer_scores_raw = combined_gate_scores[layer_idx]

        # Skip layers with all-zero gate scores
        if layer_scores_raw.sum() == 0:
            print(f"Layer {layer_idx} has zero gate scores, skipping.")
            continue

        # Group sub-experts and compute per-group sums
        num_groups = sub_experts // group_size
        reshaped = layer_scores_raw[:, : num_groups * group_size].view(
            experts, num_groups, group_size
        )
        g_sums = reshaped.sum(dim=-1).view(-1)

        # Sort groups by score in descending order
        s_g_sums, s_g_idxs = torch.sort(g_sums, descending=True)

        # Cumulate until the threshold is exceeded
        c_g_sum = torch.cumsum(s_g_sums, dim=0)
        exceed_mask = c_g_sum > router_threshold
        if exceed_mask.any():
            count = torch.where(exceed_mask)[0][0].item() + 1
        else:
            count = len(s_g_sums)

        # Collect flat indices of selected groups
        final_flat_indices: list[int] = []
        for g_idx in s_g_idxs[:count]:
            e_id, g_in_e = g_idx // num_groups, g_idx % num_groups
            base = (e_id * sub_experts) + (g_in_e * group_size)
            final_flat_indices.extend(
                [base.item() + o for o in range(group_size)]
            )

        total_cum_sum += s_g_sums[:count].sum().item()

        # Convert flat indices to {expert_id: [sub_expert_ids]} format
        final_flat_indices_tensor = torch.tensor(
            final_flat_indices, device=combined_gate_scores.device
        )
        expert_ids = (final_flat_indices_tensor // sub_experts).tolist()
        sub_expert_ids = (final_flat_indices_tensor % sub_experts).tolist()

        layer_selection: dict[int, list[int]] = {}
        for eid, sid in zip(expert_ids, sub_expert_ids):
            eid = int(eid)
            if eid not in layer_selection:
                layer_selection[eid] = []
            layer_selection[eid].append(int(sid))

        selected_experts[layer_idx] = {
            eid: sorted(layer_selection[eid])
            for eid in sorted(layer_selection.keys())
        }

    # Print the average actual threshold across layers
    if total_cum_sum > 0:
        print(f"Average actual threshold per layer: {total_cum_sum / layers}")

    return selected_experts


def print_selection_stats(
    selected_experts: dict[int, dict[int, list[int]]],
) -> None:
    """Print statistics about selected experts per layer.

    Args:
        selected_experts: Dictionary mapping layer index to a dictionary of
            ``{expert_id: [sub_expert_ids]}``.
    """
    print("\nSelection Statistics:")
    print("=" * 50)

    total_count = 0

    for layer_id, experts_dict in selected_experts.items():
        total_sub_experts = sum(
            len(sub_experts) for sub_experts in experts_dict.values()
        )

        list_sub_experts_str = ""
        for key, value in experts_dict.items():
            list_sub_experts_str += f"[{key}: {len(value)}]"

        print(
            f"Layer {layer_id}:  Total Sub-Experts: {total_sub_experts}, "
            f"{list_sub_experts_str}"
        )
        total_count += total_sub_experts

    print(f"Total Selected Experts: {total_count}")


def save_data(
    selected_experts: dict[int, dict[int, list[int]]],
    save_path: str,
    save_name: str = "selected_sub_experts.json",
) -> None:
    """Save selected experts to a JSON file.

    Args:
        selected_experts: Dictionary mapping layer index to a dictionary of
            ``{expert_id: [sub_expert_ids]}``.
        save_path: Directory where the output file will be written.
        save_name: Name of the output JSON file.
    """
    output_path = os.path.join(save_path, save_name)
    results = {"selected_sub_experts": selected_experts}
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Data saved to {output_path}")


def main() -> None:
    """Main entry point: load data, select sub-experts, and save results."""
    args = parse_args()
    expert_router_ratios, gate_dist = load_data(args.input_path)
    gate_scores = combined_gate_scores(expert_router_ratios, gate_dist)
    selected_experts = select_sub_experts(
        gate_scores, args.router_threshold, group_size=args.group_size
    )
    print_selection_stats(selected_experts)
    save_name = (
        f"selected_sub_experts_group_{args.group_size}"
        f"_threshold_{args.router_threshold}.json"
    )
    save_data(selected_experts, args.input_path, save_name)


if __name__ == "__main__":
    main()
