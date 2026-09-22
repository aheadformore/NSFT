"""HuggingFace Trainer with per-parameter learning rate scaling.

This module provides :class:`SubExpertTrainer`, which extends
``transformers.Trainer`` to apply different learning rates to sub-expert
parameters based on their channel selection ratio.

Parameters with fewer active channels receive a proportionally higher
learning rate (M/k compensation, clamped to ``lr_scale_max``).  The scale
factors are provided by :class:`SubExpertMaskRegister.get_lr_scales`.

Example usage::

    from sub_expert_mask import SubExpertMaskRegister
    from sub_expert_trainer import SubExpertTrainer

    # Set up mask register
    register = SubExpertMaskRegister(
        model=model,
        sub_experts_config=selected_sub_experts,
        num_layers=model.config.num_hidden_layers,
        num_experts=model.config.num_experts,
        intermediate_size=model.config.intermediate_size,
    )
    register.apply()

    # Create trainer with LR scaling
    trainer = SubExpertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        mask_register=register,
        lr_scale_max=5.0,
    )
    trainer.train()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled

if TYPE_CHECKING:
    from sub_expert_mask import SubExpertMaskRegister


class SubExpertTrainer(Trainer):
    """HuggingFace Trainer with per-parameter learning rate scaling.

    Overrides ``create_optimizer`` to apply different learning rates to
    sub-expert parameters based on their channel selection ratio.
    Parameters with fewer active channels receive a proportionally
    higher learning rate (M/k compensation, clamped to lr_scale_max).

    Works with :class:`SubExpertMaskRegister`: the register provides
    ``get_lr_scales()`` which returns per-parameter LR scale factors.

    Attributes:
        mask_register: The SubExpertMaskRegister instance, or None for
            standard Trainer behavior.
        lr_scale_max: Maximum LR scale factor.
    """

    def __init__(
        self,
        *args: Any,
        mask_register: SubExpertMaskRegister | None = None,
        lr_scale_max: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer.

        Args:
            *args: Positional arguments passed to ``Trainer.__init__``.
            mask_register: The SubExpertMaskRegister instance. If None,
                behaves identically to the standard Trainer.
            lr_scale_max: Maximum LR scale factor. Default 5.0.
            **kwargs: Keyword arguments passed to ``Trainer.__init__``.
        """
        self.mask_register = mask_register
        self.lr_scale_max = lr_scale_max
        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with optional per-parameter LR scaling.

        When ``mask_register`` is set and provides non-empty LR scales,
        parameters are split into sub-groups: parameters listed in the
        scale dict are assigned ``lr = base_lr * scale``, while all other
        trainable parameters keep the base learning rate.

        Parameter names are matched exactly (not by substring) against the
        keys returned by ``get_lr_scales``.

        Returns:
            The configured optimizer instance.
        """
        # If no mask register, use parent's create_optimizer.
        if self.mask_register is None:
            return super().create_optimizer()

        # Get LR scales from the register.
        lr_scales = self.mask_register.get_lr_scales(self.lr_scale_max)
        if not lr_scales:
            print("No LR scales computed, using default optimizer.")
            return super().create_optimizer()

        print(f"Applying LR scaling to {len(lr_scales)} parameter tensors.")

        opt_model = (
            self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        )

        if self.optimizer is None:
            # 1. Get decay parameter names.
            decay_parameters = self.get_decay_parameter_names(opt_model)

            # 2. Initial grouping by weight decay.
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # 3. Get optimizer class and kwargs.
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = (
                    self.optimizer_cls_and_kwargs
                )
            else:
                optimizer_cls, optimizer_kwargs = (
                    self.get_optimizer_cls_and_kwargs(self.args, opt_model)
                )

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            elif "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            elif "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop(
                    "optimizer_dict"
                )

            # 4. Apply LR scaling: split parameters by scale factor.
            #    Build a name->param lookup for exact matching.
            param_name_map: dict[int, str] = {}
            for n, p in opt_model.named_parameters():
                param_name_map[id(p)] = n

            final_grouped_parameters: list[dict[str, Any]] = []
            selected_count = 0

            for group in optimizer_grouped_parameters:
                base_lr = group.get("lr", self.args.learning_rate)

                # expert_subgroups: {scale_factor: [params]}
                expert_subgroups: dict[float, list[nn.Parameter]] = {}
                remaining_params: list[nn.Parameter] = []

                for p in group["params"]:
                    param_name = param_name_map.get(id(p))

                    matched_scale = 1.0
                    if param_name and param_name in lr_scales:
                        matched_scale = lr_scales[param_name]

                    if matched_scale != 1.0:
                        if matched_scale not in expert_subgroups:
                            expert_subgroups[matched_scale] = []
                        expert_subgroups[matched_scale].append(p)
                    else:
                        remaining_params.append(p)

                # Common parameters (no scaling).
                if remaining_params:
                    new_group = group.copy()
                    new_group["params"] = remaining_params
                    final_grouped_parameters.append(new_group)

                # Expert parameters (with scaling).
                for scale, params in expert_subgroups.items():
                    new_expert_group = group.copy()
                    new_expert_group["params"] = params
                    new_expert_group["lr"] = base_lr * scale
                    final_grouped_parameters.append(new_expert_group)
                    print(
                        f"  LR group: {len(params)} params, "
                        f"scale={scale:.2f}x, lr={base_lr * scale:.2e}"
                    )
                    selected_count += len(params)

            optimizer_grouped_parameters = final_grouped_parameters
            print(
                f"LR scaling complete: {selected_count} parameter tensors "
                f"assigned to scaled groups."
            )

            # 5. Create optimizer.
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            # 6. SageMaker compatibility.
            if is_sagemaker_mp_enabled():
                import smdistributed.modelparallel.torch as smp

                self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
