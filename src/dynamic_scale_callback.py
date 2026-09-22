"""Dynamic gradient scale callback for HuggingFace Trainer.

This module provides :class:`DynamicScaleCallback`, a
:class:`~transformers.TrainerCallback` that dynamically updates
``model.moe_scale_buffer`` during training.

It works in tandem with :class:`~sub_expert_mask.SubExpertMaskRegister`:

1. ``SubExpertMaskRegister`` sets up the initial ``moe_scale_buffer`` and
   registers gradient hooks that read from it.  The hooks multiply incoming
   gradients by the buffer values, so any in-place update to the buffer
   immediately affects gradient scaling.

2. ``DynamicScaleCallback`` registers forward hooks on selected experts'
   ``gate_proj`` layers to collect activation statistics.  At configurable
   intervals, it EMA-smooths the statistics and recomputes
   ``moe_scale_buffer`` using an entropy-based adaptive formula -- the same
   formula used by ``SubExpertMaskRegister._compute_scale_buffer``.

This separation allows the static initial scales (computed from calibration
data) to be progressively refined by live training activations.

Compatible with DeepSpeed ZeRO-3 and distributed training (all_reduce +
broadcast).

Example usage with HuggingFace Trainer::

    from sub_expert_mask import SubExpertMaskRegister
    from dynamic_scale_callback import DynamicScaleCallback

    # Step 1: Set up gradient mask (static initial scales)
    gate_dist = torch.load("output/gate_dist.pt")
    register = SubExpertMaskRegister(
        model=model,
        sub_experts_config=selected_sub_experts,
        num_layers=model.config.num_hidden_layers,
        num_experts=model.config.num_experts,
        intermediate_size=model.config.intermediate_size,
        gate_scores=gate_dist,
    )
    register.apply()

    # Step 2: Set up dynamic scale callback (updates scales during training)
    dynamic_callback = DynamicScaleCallback(
        model=model,
        sub_expert_map=selected_sub_experts,
        initial_gate_dist=gate_dist,
        group_size=1,
        ema_beta=0.99,
        scale_grad_max=5.0,
        scale_update_interval=10,
    )

    # Step 3: Add callback to trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[dynamic_callback],
    )
    trainer.train()
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback


class DynamicScaleCallback(TrainerCallback):
    """Dynamically update gradient scale buffer during training.

    Registers forward hooks on selected experts' gate_proj to collect
    activation statistics. Periodically updates model.moe_scale_buffer
    using EMA-smoothed activations and entropy-based adaptive scaling.

    Works with SubExpertMaskRegister: the register sets up gradient hooks
    that read from model.moe_scale_buffer, and this callback updates that
    buffer in-place during training.

    Attributes:
        model: The MoE model with a ``moe_scale_buffer`` attribute.
        sub_expert_map: Dictionary mapping layer_id (str) to
            ``{expert_id (str): [group_start_indices]}``.
        group_size: Number of channels per scaling group.
        ema_beta: EMA decay factor for smoothing activation statistics.
        global_base_scale: Base scale multiplier.
        scale_grad_max: Maximum gradient scale factor.
        scale_update_interval: Number of optimizer steps between updates.
        num_layers: Number of layers in the model.
        num_experts: Number of experts per layer.
        hidden_dim: Intermediate dimension of expert MLPs.
        step_gate_accumulator: Per-step activation energy accumulator.
        ema_gate_energy: EMA-smoothed normalized activation distribution.
        step_counter: Counter for update interval.
        mon_l_str: Monitoring layer id (string) for logging.
        mon_e_str: Monitoring expert id (string) for logging.
    """

    def __init__(
        self,
        model: nn.Module,
        sub_expert_map: dict[str, dict[str, list[int]]],
        initial_gate_dist: torch.Tensor | None = None,
        group_size: int = 1,
        ema_beta: float = 0.99,
        global_base_scale: float = 1.0,
        scale_grad_max: float = 5.0,
        scale_update_interval: int = 10,
    ) -> None:
        """Initialize the dynamic scale callback.

        Args:
            model: The MoE model. Must expose ``model.config`` with
                ``num_hidden_layers``, ``num_experts``, and
                ``intermediate_size`` (or ``moe_intermediate_size``).
                A ``moe_scale_buffer`` will be created if it does not
                already exist.
            sub_expert_map: Dictionary mapping layer_id (str) to
                ``{expert_id (str): [group_start_indices]}`` for selected
                sub-experts.  Forward hooks are registered on these experts'
                ``gate_proj`` layers.
            initial_gate_dist: Optional tensor of shape
                ``[num_layers, num_experts, intermediate_size]`` loaded
                from ``cal_sub_router_chat.py``'s ``gate_dist.pt``.  When
                provided, the EMA is initialized from this distribution and
                an initial scale is computed immediately in
                :meth:`on_train_begin`.
            group_size: Number of channels per scaling group. Default 1
                (per-channel scaling).
            ema_beta: EMA decay factor. Higher values produce smoother
                updates. Default 0.99.
            global_base_scale: Base scale multiplier applied to all groups.
                Default 1.0.
            scale_grad_max: Maximum gradient scale factor (upper clamp).
                Default 5.0.
            scale_update_interval: Number of optimizer steps between scale
                updates. Default 10.
        """
        self.model = model
        self.sub_expert_map = sub_expert_map
        self.group_size = group_size
        self.ema_beta = ema_beta
        self.global_base_scale = global_base_scale
        self.scale_grad_max = scale_grad_max
        self.scale_update_interval = scale_update_interval
        self.step_counter = 0
        self.initial_gate_dist = initial_gate_dist

        # Get model dimensions (compatible with moe_intermediate_size).
        self.num_layers = model.config.num_hidden_layers
        self.num_experts = model.config.num_experts
        self.hidden_dim = (
            model.config.moe_intermediate_size
            if hasattr(model.config, "moe_intermediate_size")
            else model.config.intermediate_size
        )

        # Initialize accumulators as None; aligned to device in on_train_begin.
        self.step_gate_accumulator: torch.Tensor | None = None
        self.ema_gate_energy: torch.Tensor | None = None

        # Ensure moe_scale_buffer exists on the model.
        if not hasattr(model, "moe_scale_buffer"):
            model.register_buffer(
                "moe_scale_buffer",
                torch.ones(
                    (self.num_layers, self.num_experts, self.hidden_dim)
                ),
                persistent=True,
                requires_grad=False,
            )
        # DeepSpeed ZeRO-3: mark buffer as unpartitioned so it is not
        # sharded across processes.
        try:
            model.moe_scale_buffer.ds_status = "unpartitioned"  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            pass

        # Select a monitoring expert for logging.
        try:
            self.mon_l_str: str | None = next(iter(self.sub_expert_map))
            self.mon_e_str: str | None = next(
                iter(self.sub_expert_map[self.mon_l_str])
            )
            print(
                f"[Init] DynamicScaleCallback: Monitoring Layer "
                f"{self.mon_l_str}, Expert {self.mon_e_str}"
            )
        except StopIteration:
            self.mon_l_str = None
            self.mon_e_str = None

        self._register_forward_hooks()

    # ------------------------------------------------------------------
    # TrainerCallback lifecycle
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Align all tensors to the correct GPU device at training start.

        If ``initial_gate_dist`` was provided, the EMA energy distribution
        is initialized from it (normalized to a probability distribution)
        and an initial scale is computed immediately.  Otherwise, the EMA
        is initialized to zeros and scales remain at 1.0 until the first
        update interval.

        Args:
            args: ``TrainingArguments`` from the HuggingFace Trainer.
            state: ``TrainerState`` from the HuggingFace Trainer.
            control: ``TrainerControl`` from the HuggingFace Trainer.
            **kwargs: Additional keyword arguments from the Trainer.
        """
        # Determine target device (current GPU under distributed training).
        if torch.cuda.is_available():
            target_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
            )
        else:
            target_device = next(self.model.parameters()).device

        # Move moe_scale_buffer to target device (ZeRO-3 may leave it on
        # meta/cpu after model init).
        self.model.moe_scale_buffer = self.model.moe_scale_buffer.to(
            target_device
        )

        # Initialize or move step accumulator.
        if self.step_gate_accumulator is None:
            self.step_gate_accumulator = torch.zeros(
                (self.num_layers, self.num_experts, self.hidden_dim),
                device=target_device,
            )
        else:
            self.step_gate_accumulator = self.step_gate_accumulator.to(
                target_device
            )

        # Initialize EMA energy distribution.
        if self.initial_gate_dist is not None:
            raw_dist = self.initial_gate_dist.to(target_device).float()
            # Normalize each expert's channel energy to sum to 1.
            norm_factor = raw_dist.sum(dim=-1, keepdim=True) + 1e-10
            self.ema_gate_energy = raw_dist / norm_factor
            # Immediately compute initial scales from the gate distribution.
            print(
                "[Callback] Pre-calculating initial scales from "
                "gate distribution..."
            )
            self._update_scales(self.ema_gate_energy, is_initial=True)
        elif self.ema_gate_energy is None:
            self.ema_gate_energy = torch.zeros_like(
                self.step_gate_accumulator
            )

        # Ensure moe_scale_buffer is not on meta or cpu.
        if (
            self.model.moe_scale_buffer.is_meta
            or self.model.moe_scale_buffer.device.type == "cpu"
        ):
            real_buffer = torch.ones_like(
                self.model.moe_scale_buffer, device=target_device
            )
            self.model.moe_scale_buffer = real_buffer

        print(f"[Callback] Buffers initialized on device: {target_device}")

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Periodically update scales from accumulated activation statistics.

        Every ``scale_update_interval`` optimizer steps:
        1. All-reduce the accumulator across processes.
        2. Normalize and EMA-smooth the per-expert activation energy.
        3. Recompute scales via :meth:`_update_scales`.
        4. Log monitoring statistics and reset the accumulator.

        Args:
            args: ``TrainingArguments`` from the HuggingFace Trainer.
            state: ``TrainerState`` from the HuggingFace Trainer.
            control: ``TrainerControl`` from the HuggingFace Trainer.
            **kwargs: Additional keyword arguments from the Trainer.
        """
        if self.step_gate_accumulator is None:
            return

        self.step_counter += 1
        if self.step_counter < self.scale_update_interval:
            return

        self.step_counter = 0

        # All-reduce accumulator across processes for global statistics.
        if dist.is_initialized():
            dist.all_reduce(
                self.step_gate_accumulator, op=dist.ReduceOp.SUM
            )

        rank = dist.get_rank() if dist.is_initialized() else 0

        with torch.no_grad():
            device = self.step_gate_accumulator.device

            for l_str, experts in self.sub_expert_map.items():
                l_idx = int(l_str)
                for e_str, active_groups in experts.items():
                    e_idx = int(e_str)

                    # Use float64 for distributed numerical consistency.
                    raw_current = (
                        self.step_gate_accumulator[l_idx, e_idx]
                        .detach()
                        .clone()
                        .double()
                    )
                    current_sum = raw_current.sum()

                    # Skip if no activation was captured this interval.
                    if current_sum < 1e-12:
                        continue

                    norm_current = raw_current / (current_sum + 1e-14)

                    # EMA update in float64 shadow space.
                    old_ema = (
                        self.ema_gate_energy[l_idx, e_idx]
                        .detach()
                        .clone()
                        .double()
                    )
                    if (old_ema == 0).all():
                        new_ema = norm_current
                    else:
                        new_ema = (
                            old_ema * self.ema_beta
                            + norm_current * (1 - self.ema_beta)
                        )
                    new_ema = new_ema / (new_ema.sum() + 1e-14)

                    # NaN/Inf safety check.
                    if torch.isnan(new_ema).any() or torch.isinf(
                        new_ema
                    ).any():
                        print(
                            f"[Warning] NaN in EMA at L{l_idx}E{e_idx}, "
                            f"skipping update."
                        )
                        continue

                    # Commit EMA update (back to float32).
                    self.ema_gate_energy[l_idx, e_idx].copy_(
                        new_ema.float()
                    )

            # Recompute scales from updated EMA.
            self._update_scales(self.ema_gate_energy)

            # Print monitoring info.
            if rank == 0 and self.mon_l_str is not None:
                mon_scale = self.model.moe_scale_buffer[
                    int(self.mon_l_str), int(self.mon_e_str)
                ]
                print(
                    f"| Step {state.global_step} | "
                    f"L{self.mon_l_str}E{self.mon_e_str} | "
                    f"Scale Mean: {mon_scale.mean().item():.4f} |"
                )

            # Clear accumulator for next interval.
            self.step_gate_accumulator.zero_()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _register_forward_hooks(self) -> None:
        """Register forward hooks on selected experts' gate_proj layers.

        Each hook computes ``F.silu(output).abs()`` (the activation energy
        of the gate projection), sums over the token dimension, and
        accumulates the result into
        ``self.step_gate_accumulator[l_idx, e_idx]``.

        The expert module is accessed via
        ``self.model.model.layers[l_idx].mlp.experts[e_idx]``.
        """

        def get_hook(l_idx: int, e_idx: int):
            def hook_fn(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
                with torch.no_grad():
                    # Activation energy: |SiLU(gate_output)|
                    activated = F.silu(output).detach().abs().float()
                    # Sum over token dimensions (batch, seq) or (seq,).
                    if activated.ndim == 3:
                        summed = activated.sum(dim=(0, 1))
                    else:
                        summed = activated.sum(dim=0)

                    # Guard: accumulator may be None before on_train_begin.
                    if self.step_gate_accumulator is None:
                        return

                    # Clone to ensure independent memory before add_.
                    clean_activation = summed.clone()
                    self.step_gate_accumulator[l_idx, e_idx].add_(
                        clean_activation
                    )

            return hook_fn

        for l_str, experts in self.sub_expert_map.items():
            l_idx = int(l_str)
            for e_str in experts.keys():
                e_idx = int(e_str)
                target_expert = (
                    self.model.model.layers[l_idx].mlp.experts[e_idx]
                )
                target_expert.gate_proj.register_forward_hook(
                    get_hook(l_idx, e_idx)
                )

    def _update_scales(
        self,
        ema_energy: torch.Tensor,
        is_initial: bool = False,
    ) -> None:
        """Update ``model.moe_scale_buffer`` from EMA energy distribution.

        Uses entropy-based adaptive gamma:
        - gamma = 0.5 + 0.5 * entropy / max_entropy
        - scale = clamp(global_base_scale * (g_energy / avg_energy)^gamma,
                        1.0, scale_grad_max)

        Computation is performed on rank 0 only, then the result is
        broadcast to all processes to ensure consistency.

        Args:
            ema_energy: EMA-smoothed activation energy tensor of shape
                ``[num_layers, num_experts, hidden_dim]``.  Each expert's
                vector should be normalized to sum to 1.
            is_initial: If True, indicates this is the initial scale
                computation (for logging purposes).
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        device = ema_energy.device

        # Temporary buffer for computed scales (float32 for precision).
        temp_scales = torch.ones_like(
            self.model.moe_scale_buffer,
            dtype=torch.float32,
            device=device,
        )

        if rank == 0:
            if is_initial:
                print(
                    "[DynamicScaleCallback] Computing initial scales "
                    "from gate distribution..."
                )
            with torch.no_grad():
                for l_str, experts in self.sub_expert_map.items():
                    l_idx = int(l_str)
                    for e_str, active_groups in experts.items():
                        e_idx = int(e_str)
                        ema_val = ema_energy[l_idx, e_idx].float()
                        if (ema_val == 0).all():
                            continue

                        # Compute group energies.
                        g_energies = torch.stack(
                            [
                                ema_val[g : g + self.group_size].sum()
                                for g in active_groups
                            ]
                        )
                        total_energy = g_energies.sum() + 1e-10
                        num_active = len(g_energies)
                        avg_energy = total_energy / num_active

                        # Entropy-based gamma.
                        if num_active > 1:
                            probs = torch.clamp(
                                g_energies / total_energy, min=1e-10
                            )
                            entropy = -torch.sum(
                                probs * torch.log(probs)
                            )
                            max_ent = torch.log(
                                torch.tensor(
                                    float(num_active), device=device
                                )
                            )
                            gamma = torch.clamp(
                                0.5
                                + 0.5
                                * (entropy / (max_ent + 1e-8)),
                                0.5,
                                1.0,
                            )
                        else:
                            gamma = 0.5

                        # Per-group scale.
                        for i, g_start in enumerate(active_groups):
                            ratio = torch.clamp(
                                g_energies[i] / (avg_energy + 1e-10),
                                min=0.1,
                                max=10.0,
                            )
                            scale = torch.clamp(
                                self.global_base_scale * ratio.pow(gamma),
                                1.0,
                                self.scale_grad_max,
                            )
                            temp_scales[
                                l_idx,
                                e_idx,
                                g_start : g_start + self.group_size,
                            ] = scale

        # Broadcast from rank 0 to all processes.
        if dist.is_initialized():
            dist.broadcast(temp_scales, src=0)

        # Write to model buffer in-place.
        self.model.moe_scale_buffer.copy_(
            temp_scales.to(self.model.moe_scale_buffer.dtype)
        )
        # Ensure computation is complete before proceeding.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
