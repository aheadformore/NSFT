"""MoE sub-expert training script with DeepSpeed and OOM protection.

This script orchestrates selective sub-expert training for MoE models
(OLMoE, Mixtral, Qwen2-MoE, etc.) by combining three components:

1. :class:`~sub_expert_mask.SubExpertMaskRegister` -- freezes all model
   parameters except selected sub-expert channels and registers gradient
   hooks that apply a mask (binary or entropy-scaled) to gradients.

2. :class:`~sub_expert_trainer.SubExpertTrainer` -- extends HuggingFace
   ``Trainer`` with per-parameter learning rate scaling (M/k compensation).

3. :class:`~dynamic_scale_callback.DynamicScaleCallback` -- (optional)
   dynamically updates the gradient scale buffer during training using
   EMA-smoothed activation statistics.

Usage examples
---------------

Basic training (binary mask)::

    python src/train.py \
        --model_path /path/to/model \
        --train_file /path/to/tokenized_data \
        --sub_experts_file outputs/selected_sub_experts.json \
        --output_dir ./output

Entropy-based gradient scaling with gate distribution::

    python src/train.py \
        --model_path /path/to/model \
        --train_file /path/to/tokenized_data \
        --sub_experts_file outputs/selected_sub_experts.json \
        --gate_dist_file outputs/gate_dist.pt \
        --output_dir ./output

Dynamic gradient scaling during training::

    python src/train.py \
        --model_path /path/to/model \
        --train_file /path/to/tokenized_data \
        --sub_experts_file outputs/selected_sub_experts.json \
        --gate_dist_file outputs/gate_dist.pt \
        --dynamic_scale \
        --output_dir ./output

With packing (enabled by default, requires Flash Attention 2 + binpacking library)::

    python src/train.py \
        --model_path /path/to/model \
        --train_file /path/to/tokenized_data \
        --sub_experts_file outputs/selected_sub_experts.json \
        --flash_attention_2 \
        --output_dir ./output

    # Install binpacking: pip install binpacking
    # Packing is enabled by default; it is automatically disabled if
    # Flash Attention 2 is not enabled.

With DeepSpeed ZeRO-3 (single node, 4 GPUs)::

    torchrun --nproc_per_node=4 src/train.py \
        --model_path /path/to/moe_model \
        --train_file /path/to/tokenized_data \
        --sub_experts_file output/selected_sub_experts.json \
        --deepspeed ds_config.json

Or using the deepspeed launcher::

    deepspeed src/train.py \
        --deepspeed ds_config.json \
        --model_path /path/to/moe_model \
        --train_file /path/to/tokenized_data \
        --sub_experts_file output/selected_sub_experts.json

Example DeepSpeed ZeRO-3 config (``ds_config.json``)::

    {
        "bf16": {"enabled": true},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": true,
            "contiguous_gradients": true
        },
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto"
    }
"""

from __future__ import annotations

import argparse
import json
import os
import types

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments

from sub_expert_mask import SubExpertMaskRegister
from sub_expert_trainer import SubExpertTrainer

# Delayed import: DynamicScaleCallback depends on transformers.TrainerCallback
# which is always available, but we keep this pattern for consistency and
# to allow the script to run without the callback module if unused.
try:
    from dynamic_scale_callback import DynamicScaleCallback
except ImportError:
    DynamicScaleCallback = None  # type: ignore[assignment,misc]


class PackedDataset(Dataset):
    """Dataset with optional binpacking for pre-tokenized Arrow data.

    Loads a pre-tokenized dataset from disk via
    ``datasets.load_from_disk()``.  Each sample must contain ``input_ids``
    and ``labels`` fields.

    When ``packing=True``, multiple short sequences are combined to fill
    ``max_length`` using the ``binpacking`` library.  Position IDs reset
    to 0 for each sub-sequence.  Requires Flash Attention 2.

    Attributes:
        dataset: Underlying Arrow dataset.
        max_length: Maximum packed sequence length.
        packing: Whether binpacking is enabled.
    """

    def __init__(
        self,
        data_path: str,
        max_length: int = 8192,
        packing: bool = False,
        pad_token_id: int = 0,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_path: Path to pre-tokenized Arrow dataset directory.
            max_length: Maximum packed sequence length.
            packing: If True, enable binpacking.
            pad_token_id: Token ID for padding to max_length.
        """
        self.max_length = max_length
        self.packing = packing
        self.pad_token_id = pad_token_id

        from datasets import load_from_disk

        self.dataset = load_from_disk(data_path)
        print(f"Loaded pre-tokenized dataset: {len(self.dataset)} samples")

        if packing:
            self._compute_packing_indices()

    def _compute_packing_indices(self) -> None:
        """Use binpacking library to compute packing plan.

        Groups short sequences together to fill ``max_length`` bins using
        first-fit-decreasing algorithm via
        ``binpacking.to_constant_volume()``.
        """
        import binpacking

        lengths: list[tuple[int, int]] = []
        for i in range(len(self.dataset)):
            length = len(self.dataset[i]["input_ids"])
            lengths.append((i, length))

        valid = [
            (idx, length) for idx, length in lengths
            if length <= self.max_length
        ]
        skipped = len(lengths) - len(valid)
        if skipped > 0:
            print(f"Warning: {skipped} sequences exceed max_length, skipped")

        bins = binpacking.to_constant_volume(
            valid, self.max_length, weight_pos=1
        )

        self.packed_indices: list[list[int]] = [
            [item[0] for item in bin_items] for bin_items in bins
        ]

        total_samples = len(valid)
        total_packed = len(self.packed_indices)
        avg_pack = (
            total_samples / total_packed if total_packed > 0 else 0
        )
        print(
            f"Packing: {total_samples} sequences -> {total_packed} packed bins "
            f"(avg {avg_pack:.1f} sequences/bin)"
        )

    def __len__(self) -> int:
        """Return the number of (packed) samples."""
        if self.packing:
            return len(self.packed_indices)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a (packed or single) sequence."""
        if self.packing:
            return self._get_packed_item(idx)
        return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> dict[str, list]:
        """Return a single (unpacked) sequence.

        Args:
            idx: Sample index.

        Returns:
            Dict with ``input_ids`` and ``labels`` as Python lists.
        """
        data = self.dataset[idx]
        return {
            "input_ids": list(data["input_ids"]),
            "labels": list(data["labels"]),
        }

    def _get_packed_item(self, idx: int) -> dict[str, list]:
        """Return a packed sequence combining multiple sub-sequences.

        Position IDs reset to 0 for each sub-sequence. This triggers
        FA2's block-diagonal attention via `_is_packed_sequence` detection
        in standard transformers (requires batch_size=1 and
        attention_mask=None at the attention layer).

        Args:
            idx: Packed bin index.

        Returns:
            Dict with ``input_ids``, ``labels``, and ``position_ids``
            as Python lists (no ``attention_mask`` — let
            ``DataCollatorForSeq2Seq`` generate all-1s mask so that
            ``_update_causal_mask`` returns None for FA2).
        """
        packed_ids: list = []
        packed_labels: list = []
        packed_position_ids: list[int] = []

        for sample_idx in self.packed_indices[idx]:
            data = self.dataset[sample_idx]
            input_ids = data["input_ids"]
            labels = data["labels"]

            packed_ids.extend(input_ids)
            packed_labels.extend(labels)
            packed_position_ids.extend(range(len(input_ids)))

        # Truncate if exceeds max_length.
        if len(packed_ids) > self.max_length:
            packed_ids = packed_ids[: self.max_length]
            packed_labels = packed_labels[: self.max_length]
            packed_position_ids = packed_position_ids[: self.max_length]

        # Pad to max_length at dataset level (matches original code).
        # position_ids padding uses range(diff) like the original
        # SFTPackingDataset — padding position_id=0 marks a new
        # sub-sequence boundary, which FA2 handles correctly.
        if len(packed_ids) < self.max_length:
            diff = self.max_length - len(packed_ids)
            packed_ids.extend([self.pad_token_id] * diff)
            packed_labels.extend([-100] * diff)
            packed_position_ids.extend(range(diff))

        return {
            "input_ids": packed_ids,
            "labels": packed_labels,
            "position_ids": packed_position_ids,
        }


class PackedCollator:
    """Collator for packed sequences.

    Pads ``input_ids``, ``labels``, and ``position_ids`` to the longest
    sequence in each batch.

    When packing is enabled, ``attention_mask`` is NOT generated — Flash
    Attention 2 uses ``position_ids`` for block-diagonal attention.
    When packing is disabled, ``attention_mask`` is generated for padding.

    Attributes:
        pad_token_id: Token ID used for padding ``input_ids``.
        packing: Whether packed sequences are used.
    """

    def __init__(self, pad_token_id: int = 0, packing: bool = False) -> None:
        """Initialize the collator.

        Args:
            pad_token_id: Token ID for padding ``input_ids``.
            packing: If True, do not generate attention_mask (FA2 uses
                position_ids for block-diagonal attention).
        """
        self.pad_token_id = pad_token_id
        self.packing = packing

    def __call__(
        self, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collate a list of feature dicts into a padded batch.

        Args:
            features: List of dicts with keys ``input_ids``, ``labels``,
                and ``position_ids``, each a 1-D tensor.

        Returns:
            Dictionary of stacked batched tensors. When packing is
            enabled, only ``input_ids``, ``labels``, ``position_ids``
            are returned (no ``attention_mask`` — FA2 uses position_ids
            for block-diagonal attention). When packing is disabled,
            ``attention_mask`` is also included.
        """
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        position_ids_list: list[torch.Tensor] = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids_list.append(torch.cat([
                f["input_ids"],
                torch.full(
                    (pad_len,), self.pad_token_id, dtype=torch.long
                ),
            ]))
            labels_list.append(torch.cat([
                f["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ]))
            position_ids_list.append(torch.cat([
                f["position_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))

        # Always generate attention_mask. For packed sequences,
        # all tokens are max_length (padded at dataset level), so
        # attention_mask is all 1s — matching DataCollatorForSeq2Seq
        # behavior in the original code.
        attention_mask_list: list[torch.Tensor] = []
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            attention_mask_list.append(torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]))

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "position_ids": torch.stack(position_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="MoE sub-expert training script with DeepSpeed support."
    )

    # Paths
    parser.add_argument(
        "--model_path",
        required=True,
        help="MoE model path (HuggingFace format).",
    )
    parser.add_argument(
        "--train_file",
        required=True,
        help="Path to pre-tokenized dataset directory (Arrow format).",
    )
    parser.add_argument(
        "--sub_experts_file",
        required=True,
        help="Path to selected_sub_experts.json.",
    )
    parser.add_argument(
        "--gate_dist_file",
        default="",
        help="Path to gate_dist.pt (for gradient scaling, empty=binary mask).",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        help="Output directory for checkpoints and final model.",
    )
    # local_rank is passed by deepspeed/torchrun launchers.
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by launcher).",
    )

    # Sequence length
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length for tokenization.",
    )

    # Packing & attention
    parser.add_argument(
        "--packing",
        type=bool,
        default=True,
        help="Enable binpacking to pack multiple sequences into max_length.",
    )
    parser.add_argument(
        "--no_packing",
        action="store_true",
        default=False,
        help="Disable packing (use individual sequences with padding).",
    )
    parser.add_argument(
        "--flash_attention_2",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 (required for packing).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.02,
        help="Warmup ratio of total steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Checkpoint save interval in steps.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="polynomial",
        help="LR scheduler type (cosine_with_min_lr requires transformers >= 4.44).",
    )
    parser.add_argument(
        "--min_lr_rate",
        type=float,
        default=3e-6,
        help="Minimum LR absolute value (for cosine_with_min_lr, converted to fraction of base LR).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )

    # Sub-expert training options
    parser.add_argument(
        "--group_size",
        type=int,
        default=1,
        help="Number of channels per scaling group.",
    )
    parser.add_argument(
        "--scale_grad_max",
        type=float,
        default=5.0,
        help="Maximum gradient scale factor.",
    )
    parser.add_argument(
        "--lr_scale_max",
        type=float,
        default=5.0,
        help="Maximum learning rate scale factor (M/k compensation).",
    )
    parser.add_argument(
        "--no_gradient_scaling",
        action="store_true",
        help="Use binary mask instead of entropy-based scaling.",
    )
    parser.add_argument(
        "--dynamic_scale",
        action="store_true",
        help="Enable dynamic gradient scaling during training.",
    )
    parser.add_argument(
        "--scale_update_interval",
        type=int,
        default=5,
        help="Optimizer steps between dynamic scale updates.",
    )
    parser.add_argument(
        "--ema_beta",
        type=float,
        default=0.9,
        help="EMA decay factor for dynamic scale smoothing.",
    )

    # Memory & distributed
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="",
        help="DeepSpeed config JSON file path.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision.",
    )

    # Liger Kernel & loss normalization
    parser.add_argument(
        "--use_liger_kernel",
        action="store_true",
        default=True,
        help="Enable Liger Kernel for memory-efficient attention.",
    )
    parser.add_argument(
        "--average_tokens_across_devices",
        action="store_true",
        default=True,
        help="Average token count across devices for loss normalization.",
    )
    parser.add_argument(
        "--use_reference_modeling",
        action="store_true",
        default=False,
        help="Load and register reference/modeling_olmoe.py (with FA2 "
             "position_ids fix for packing). Required for block-diagonal "
             "attention with packed sequences.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the sub-expert training pipeline."""
    args = parse_args()

    # Check packing + flash attention compatibility.
    if args.no_packing:
        args.packing = False
    if args.packing and not args.flash_attention_2:
        print("WARNING: Packing requires Flash Attention 2. Disabling packing.")
        args.packing = False

    # ---------------------------------------------------------------
    # 1. Load tokenizer and model.
    # ---------------------------------------------------------------
    # Enable Liger Kernel for memory-efficient attention.
    if args.use_liger_kernel:
        os.environ["USE_LIGER_KERNEL"] = "True"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # Use reference modeling file if requested (FA2 position_ids fix).
    # Directly use OlmoeForCausalLM instead of AutoModelForCausalLM to
    # avoid re-registration conflicts with the existing "olmoe" type.
    if args.use_reference_modeling:
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reference_dir = os.path.join(project_root, "reference")
        if reference_dir not in sys.path:
            sys.path.insert(0, reference_dir)
        from modeling_olmoe import OlmoeForCausalLM
        print(f"Loaded reference modeling_olmoe.py from {reference_dir}")
        print("FA2 block-diagonal attention for packing enabled")
        model = OlmoeForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=(
                "flash_attention_2" if args.flash_attention_2 else "eager"
            ),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=(
                "flash_attention_2" if args.flash_attention_2 else "eager"
            ),
        )

    # Enable gradient checkpointing for memory savings.
    # enable_input_require_grads() is required because most parameters
    # are frozen — without it, checkpointing finds no inputs requiring
    # gradients and backward fails.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    # ---------------------------------------------------------------
    # 2. Load sub-experts config.
    # ---------------------------------------------------------------
    with open(args.sub_experts_file, "r", encoding="utf-8") as f:
        sub_experts_config = json.load(f)["selected_sub_experts"]

    # ---------------------------------------------------------------
    # 3. Load gate distribution (optional).
    # ---------------------------------------------------------------
    gate_dist: torch.Tensor | None = None
    if args.gate_dist_file and os.path.exists(args.gate_dist_file):
        gate_dist = torch.load(args.gate_dist_file)
        print(f"Loaded gate_dist: shape={gate_dist.shape}")
    elif not args.no_gradient_scaling:
        print(
            "Warning: gate_dist not provided, falling back to binary mask."
        )

    if args.no_gradient_scaling:
        gate_dist = None  # Force binary mask mode.

    # ---------------------------------------------------------------
    # 4. Set up mask register.
    # ---------------------------------------------------------------
    register = SubExpertMaskRegister(
        model=model,
        sub_experts_config=sub_experts_config,
        num_layers=model.config.num_hidden_layers,
        num_experts=model.config.num_experts,
        intermediate_size=model.config.intermediate_size,
        gate_scores=gate_dist,
        scale_grad_max=args.scale_grad_max,
        group_size=args.group_size,
    )
    register.apply()

    # ---------------------------------------------------------------
    # 5. Create dataset and collator.
    # ---------------------------------------------------------------
    train_dataset = PackedDataset(
        data_path=args.train_file,
        max_length=args.max_length,
        packing=args.packing,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    # Use DataCollatorForSeq2Seq to exactly match the original code's
    # collator behavior (auto-generates attention_mask).
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # ---------------------------------------------------------------
    # 6. Set up dynamic scale callback (optional).
    # ---------------------------------------------------------------
    callbacks: list = []
    if args.dynamic_scale and gate_dist is not None:
        if DynamicScaleCallback is None:
            print(
                "Warning: DynamicScaleCallback import failed. "
                "Dynamic scaling will be disabled."
            )
        else:
            dynamic_callback = DynamicScaleCallback(
                model=model,
                sub_expert_map=sub_experts_config,
                initial_gate_dist=gate_dist,
                group_size=args.group_size,
                ema_beta=args.ema_beta,
                scale_grad_max=args.scale_grad_max,
                scale_update_interval=args.scale_update_interval,
            )
            callbacks.append(dynamic_callback)
            print("Dynamic gradient scaling enabled.")

    # ---------------------------------------------------------------
    # 7. Training arguments.
    # ---------------------------------------------------------------
    # Set min LR kwargs based on scheduler type.
    # polynomial uses lr_end (absolute value),
    # cosine_with_min_lr uses min_lr_rate (fraction of base LR).
    lr_scheduler_type = args.lr_scheduler_type
    if lr_scheduler_type == "polynomial":
        lr_scheduler_kwargs = {"lr_end": args.min_lr_rate}
    elif lr_scheduler_type == "cosine_with_min_lr":
        min_lr_fraction = (
            args.min_lr_rate / args.learning_rate
            if args.learning_rate > 0
            else 0.1
        )
        lr_scheduler_kwargs = {"min_lr_rate": min_lr_fraction}
    else:
        lr_scheduler_kwargs = {}

    training_kwargs: dict = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        report_to="tensorboard",
        save_total_limit=3,
        save_safetensors=True,
        average_tokens_across_devices=args.average_tokens_across_devices,
        remove_unused_columns=False,
    )
    if args.deepspeed:
        training_kwargs["deepspeed"] = args.deepspeed

    training_args = TrainingArguments(**training_kwargs)

    # ---------------------------------------------------------------
    # 8. Create trainer.
    # ---------------------------------------------------------------
    trainer_kwargs: dict = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        mask_register=register,
        lr_scale_max=args.lr_scale_max,
        callbacks=callbacks,
    )

    # Handle transformers version compatibility for tokenizer argument.
    major_version = int(transformers.__version__.split(".")[0])
    if major_version >= 5:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SubExpertTrainer(**trainer_kwargs)

    # ---------------------------------------------------------------
    # 9. Print configuration summary.
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model:                  {args.model_path}")
    print(f"Train data:             {args.train_file}")
    print(f"Sub-experts:            {args.sub_experts_file}")
    print(
        f"Gate dist:              "
        f"{args.gate_dist_file or 'None (binary mask)'}"
    )
    print(
        f"Gradient scaling:       "
        f"{'entropy-based' if gate_dist is not None else 'binary mask'}"
    )
    print(
        f"Dynamic scaling:        "
        f"{'enabled' if args.dynamic_scale else 'disabled'}"
    )
    print(f"LR scaling:             enabled (max={args.lr_scale_max})")
    print(f"DeepSpeed:              {args.deepspeed or 'disabled'}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"Packing:                {args.packing}")
    print(f"Flash Attention 2:      {args.flash_attention_2}")
    print(f"LR scheduler:           {lr_scheduler_type}")
    print(f"Weight decay:           {args.weight_decay}")
    print(f"Max grad norm:          {args.max_grad_norm}")
    print(
        f"Liger kernel:           "
        f"{'enabled' if args.use_liger_kernel else 'disabled'}"
    )
    print(
        f"Avg tokens across devs: "
        f"{'enabled' if args.average_tokens_across_devices else 'disabled'}"
    )
    print(
        f"Batch size:             "
        f"{args.batch_size} x {args.gradient_accumulation_steps} accumulation"
    )
    print(f"Learning rate:          {args.learning_rate}")
    print(f"Epochs:                 {args.num_train_epochs}")
    print("=" * 60 + "\n")

    # ---------------------------------------------------------------
    # 10. Train.
    # ---------------------------------------------------------------
    trainer.train()

    # ---------------------------------------------------------------
    # 11. Save and cleanup.
    # ---------------------------------------------------------------
    # Remove training-only buffers before saving (not part of original model).
    if hasattr(model, "moe_scale_buffer"):
        del model._buffers["moe_scale_buffer"]
        print("Removed moe_scale_buffer before saving (not needed for inference)")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    register.cleanup()

    print(f"\nTraining complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
