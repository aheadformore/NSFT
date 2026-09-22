# MoE Sub-Expert Training

A training toolkit for selective sub-expert fine-tuning of Mixture-of-Experts (MoE) models, supporting channel-level gradient masking and learning rate scaling.

The pipeline consists of three steps:

- **Step 1**: Compute expert routing scores and gate feature distributions using `cal_sub_router_chat.py`
- **Step 2**: Select sub-experts based on the computed scores using `select_top_sub_experts.py`
- **Step 3**: Train with selective freezing using `train.py`

## Project Structure

```
NSFT/
├── configs/
│   └── deepspeed_config_stage1_no_optim.json  # DeepSpeed config
├── data/
│   └── SciRiFF_train_8192_samples_1k.jsonl   # Example data
├── src/
│   ├── cal_sub_router_chat.py       # Step 1: Compute routing scores
│   ├── select_top_sub_experts.py    # Step 2: Select sub-experts
│   ├── train.py                     # Step 3: Training entry point
│   ├── sub_expert_mask.py           # Freeze + gradient mask + LR scaling
│   ├── sub_expert_trainer.py        # Trainer with LR scaling
│   └── dynamic_scale_callback.py    # Dynamic gradient scaling (optional)
├── reference/
│   └── modeling_olmoe.py            # FA2 position_ids fix for OLMoE
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

### Step 1: Compute Routing Scores

Use `cal_sub_router_chat.py` to compute expert routing scores and gate feature distributions.

**Features:**
- Router logits based expert usage statistics (N_router)
- Forward hooks on `gate_proj` to collect SiLU-activated L1-normalized feature distributions (M)
- Consistency verification: N_router == N_hook, Sum(M) ≈ N_hook
- Supports OLMoE / Mixtral / Qwen2-MoE architectures

```bash
python src/cal_sub_router_chat.py \
    --model_path /path/to/moe/model \
    --input_file data/your_data.jsonl \
    --save_path outputs/
```

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `--model_path` | str | Yes | — | Path to the MoE model |
| `--input_file` | str | Yes | — | Input JSONL data file |
| `--max_length` | int | No | 8192 | Maximum input sequence length |
| `--top_k` | int | No | 8 | Number of top-k experts |
| `--device` | str | No | cuda | Device |
| `--test_number` | int | No | 0 | Number of samples (0 = all) |
| `--save_path` | str | Yes | — | Output directory |

**Outputs:**
- `experts-usage.json`: Expert activation ratios
- `gate_dist.pt`: Gate feature distribution tensor, shape `(num_layers, num_experts, intermediate_size)`
- `experts-usage.png`: Expert activation heatmap
- `gate_heatmaps/`: Per-layer gate feature heatmaps

### Step 2: Select Sub-Experts

Use `select_top_sub_experts.py` to select sub-experts based on Step 1 scores.

**Features:**
- Group channels into contiguous groups of `group_size`
- Sort groups by score sum (descending), accumulate until `router_threshold`
- Output selected channel indices per layer per expert

```bash
python src/select_top_sub_experts.py \
    --input_path outputs/ \
    --group_size 16 \
    --router_threshold 0.2
```

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `--input_path` | str | Yes | — | Step 1 output directory (containing `experts-usage.json` and `gate_dist.pt`) |
| `--group_size` | int | No | 16 | Group size for contiguous grouping |
| `--router_threshold` | float | No | 0.2 | Cumulative score threshold |

**Output:**
- `selected_sub_experts_group_16_threshold_0.2.json`: Selected sub-expert configuration

### Step 3: Selective Freezing Training

See the recommended configuration and training components below.

## Recommended Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| GPUs | 8 | 8-GPU distributed training |
| `per_device_batch_size` | 1 | Batch size per device |
| `gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| Global batch size | 8 | 8 GPUs × 1 × 1 |
| `learning_rate` | 3e-5 | Base learning rate |
| `lr_scheduler_type` | polynomial | Linear decay scheduler |
| `min_lr_rate` | 3e-6 | Minimum learning rate (absolute) |
| `warmup_ratio` | 0.02 | Warmup ratio |
| `weight_decay` | 0 | Weight decay |
| `num_train_epochs` | 2 | Number of epochs |
| `max_length` | 8192 | Maximum sequence length |
| `packing` | enabled | Binpacking for sequence packing |
| `flash_attention_2` | enabled | Flash Attention 2 |
| `gradient_checkpointing` | enabled | Gradient checkpointing for memory |

### 8-GPU Training Command

```bash
deepspeed --num_gpus=8 src/train.py \
    --model_path /path/to/olmoe-model \
    --train_file /path/to/tokenized_dataset \
    --sub_experts_file outputs/selected_sub_experts_group_16_threshold_0.2.json \
    --gate_dist_file outputs/gate_dist.pt \
    --output_dir ./output \
    --deepspeed configs/deepspeed_config_stage1_no_optim.json \
    --use_reference_modeling \
    --learning_rate 3e-5 \
    --min_lr_rate 3e-6 \
    --lr_scheduler_type polynomial \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --warmup_ratio 0.02 \
    --weight_decay 0.0 \
    --max_length 8192 \
    --flash_attention_2 \
    --gradient_checkpointing
```

### Without Packing

If you choose not to use packing (`--no_packing`), the training configuration should be adjusted accordingly. Without sequence packing, each sample is processed individually with padding, resulting in lower GPU utilization. You may need to increase `batch_size` and adjust `learning_rate` to compensate for the reduced effective throughput.

## Training Components

The training stage uses three composable components for channel-level selective freezing:

| Component | Role | Required |
|-----------|------|:--------:|
| `SubExpertMaskRegister` | Freezes all parameters, unfreezes selected experts' projection weights, registers gradient mask hooks | Yes |
| `SubExpertTrainer` | Extends HuggingFace Trainer with per-parameter learning rate scaling | No |
| `DynamicScaleCallback` | Dynamically updates gradient scaling during training via EMA + forward hooks | No |

### Gradient Scaling

- **Binary mask** (`gate_scores=None`): selected channels = 1.0, unselected = 0.0
- **Entropy-adaptive scaling** (`gate_scores=gate_dist`): adaptively assigns gradient scale factors based on the entropy of channel activation energy distribution
  - `gamma = 0.5 + 0.5 * entropy / max_entropy` (uniform → 1.0, polarized → 0.5)
  - `scale = max(1.0, min(5.0, g_energy / avg_energy^gamma))`

### Learning Rate Scaling

- Formula: `lr_scale = min(M/k, lr_scale_max)`, where M = total channels, k = selected channels
- When training only 1/8 of channels, the learning rate is scaled up by 8×
- When `mask_register=None`, falls back to standard Trainer

### Dynamic Scaling

- Collects gate_proj activation statistics via forward hooks during training
- EMA smoothing followed by entropy-adaptive formula to update `model.moe_scale_buffer`
- Without `DynamicScaleCallback`, static scaling is used (initial values remain unchanged)

**Configuration summary:**

| Configuration | Behavior |
|---------------|---------|
| `gate_scores=None`, no callback, `mask_register=None` | Standard training (no scaling) |
| `gate_scores=None`, no callback, `mask_register=register` | Binary mask + LR scaling |
| `gate_scores=gate_dist`, no callback, `mask_register=register` | Entropy-adaptive gradient scaling + LR scaling |
| `gate_scores=gate_dist`, with callback, `mask_register=register` | Dynamic entropy-adaptive gradient scaling + LR scaling |

## Packing & Model Registration

Packing (binpacking) relies on Flash Attention 2's block-diagonal attention. The standard transformers OLMoE implementation does not pass `position_ids` to `_flash_attention_forward`, which prevents block-diagonal attention for packed sequences.

### Solution: Use `--use_reference_modeling`

```bash
python src/train.py --use_reference_modeling ...
```

This loads `reference/modeling_olmoe.py`, which is a copy of the standard OLMoE modeling file with one fix: `position_ids` is passed to `_flash_attention_forward` in the Flash Attention 2 forward method, enabling block-diagonal attention for packed sequences.

**Currently supported models:** OLMoE. For other MoE architectures (Mixtral, Qwen2-MoE, etc.), a similar one-line fix can be applied to the respective modeling file.

## Model Saving

After training, `train.py` automatically removes `moe_scale_buffer` (a training-only buffer not part of the original model architecture) before `trainer.save_model()`, ensuring the saved weights can be loaded directly by vLLM and other inference frameworks.

## Data Format

**Step 1** input: JSONL format (one JSON object per line) with `messages` field:

```json
{
  "dataset": "SciRIFF",
  "task_type": "entailment",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

**Step 3** input: Pre-tokenized HuggingFace Arrow dataset directory (generated by tokenization tools such as `tools/data_tokenize.py`).

## License

MIT
