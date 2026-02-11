# LAION400M Dataset Configs - Sampling Modes

This directory contains 4 different LAION400M configurations for different sampling strategies.

## Available Configs

### 1. `laion400m.yaml` (Default - "both" mode)
**Behavior**: Sample components AND optionally add relation (can mix in same caption)

**Example captions**:
- `"Ben Affleck and movie"` (components only)
- `"Ben Affleck and movie where Ben Affleck could be addition to cast"` (components + relation)
- `"The Great Gatsby cast"` (single component)

**Use case**: Default training with maximum diversity

**Key parameters**:
```yaml
sample_relation_or_components: "both"
relation_sample_prob: 0.8  # 80% chance to add relation when available
```

---

### 2. `laion400m_relation_only.yaml` ("relation_only" mode)
**Behavior**: ONLY sample relations, ignore components

**Example captions**:
- `"Ben Affleck could be addition to The Great Gatsby cast"`
- `"movie relates to drama"`
- `"actor part of cast"`

**Use case**: Focus purely on relational learning

**Key parameters**:
```yaml
sample_relation_or_components: "relation_only"
num_component_captions: 0  # Disabled
```

---

### 3. `laion400m_components_only.yaml` ("components_only" mode)
**Behavior**: ONLY sample components, never add relations

**Example captions**:
- `"Ben Affleck and movie"`
- `"The Great Gatsby cast"`
- `"drama and movie and actor"`

**Use case**: Focus purely on compositional learning

**Key parameters**:
```yaml
sample_relation_or_components: "components_only"
sample_relations: false  # Optional (ignored anyway)
```

---

### 4. `laion400m_either.yaml` ("either" mode) 🆕
**Behavior**: Randomly choose EITHER relation OR components per sample (never both in same caption)

**Example captions**:
- Sample 1: `"Ben Affleck could be addition to cast"` (relation only)
- Sample 2: `"Ben Affleck and movie"` (components only)
- Sample 3: `"movie relates to drama"` (relation only)
- Sample 4: `"drama"` (component only)

**Use case**: Balanced mix without overlap (clean separation)

**Key parameters**:
```yaml
sample_relation_or_components: "either"
relation_sample_prob: 0.5  # 50% relation, 50% components (per sample)
```

**Adjustable distribution**:
- `relation_sample_prob: 0.3` → 30% relation, 70% components
- `relation_sample_prob: 0.7` → 70% relation, 30% components

---

## Quick Comparison

| Config | Relations | Components | Can Mix | Best For |
|--------|-----------|------------|---------|----------|
| `laion400m.yaml` | ⚠️ Sometimes | ✅ Always | ✅ Yes | General training |
| `laion400m_relation_only.yaml` | ✅ Always | ❌ Never | ❌ N/A | Relation learning |
| `laion400m_components_only.yaml` | ❌ Never | ✅ Always | ❌ N/A | Compositional learning |
| `laion400m_either.yaml` 🆕 | ⚠️ Sometimes | ⚠️ Sometimes | ❌ No | Balanced mix without overlap |

---

## Usage

### Basic Usage

```bash
# Default (both mode)
python train.py --config configs/dataset/laion400m.yaml

# Relation only
python train.py --config configs/dataset/laion400m_relation_only.yaml

# Components only
python train.py --config configs/dataset/laion400m_components_only.yaml

# Either mode (new!)
python train.py --config configs/dataset/laion400m_either.yaml
```

### Override Parameters

You can override parameters on the command line:

```bash
# Use either mode with 70% relations
python train.py --config configs/dataset/laion400m_either.yaml \
    --dataset.dataset_kwargs.relation_sample_prob 0.7

# Use both mode with 90% relation probability
python train.py --config configs/dataset/laion400m.yaml \
    --dataset.dataset_kwargs.relation_sample_prob 0.9
```

---

## Ablation Study Template

Run experiments with all 4 modes:

```bash
# Experiment 1: Components only
python train.py --config configs/dataset/laion400m_components_only.yaml \
    --experiment_name "components_only" --output_dir runs/components_only

# Experiment 2: Relations only
python train.py --config configs/dataset/laion400m_relation_only.yaml \
    --experiment_name "relation_only" --output_dir runs/relation_only

# Experiment 3: Both (default)
python train.py --config configs/dataset/laion400m.yaml \
    --experiment_name "both" --output_dir runs/both

# Experiment 4: Either (new!)
python train.py --config configs/dataset/laion400m_either.yaml \
    --experiment_name "either" --output_dir runs/either

# Compare results
python compare_experiments.py runs/*/checkpoints/best.pt
```

---

## Expected Training Metrics

### `laion400m_relation_only.yaml`
```
num_valid_component_samples: 0  # All masked (no components)
num_masked_component_samples: 512  # All batch
component_loss: 0.0  # Always masked
```

### `laion400m_components_only.yaml` or `laion400m.yaml`
```
num_valid_component_samples: ~450/512  # ~88%
num_masked_component_samples: ~62/512  # ~12% single-component
component_loss: 1.234  # Computed
```

### `laion400m_either.yaml` (50% prob)
```
num_valid_component_samples: ~225/512  # ~44%
num_masked_component_samples: ~287/512  # ~56%
# (50% relations + ~6% single-component masked)
component_loss: 0.678  # Computed on valid samples
```

---

## Key Parameters Explained

### `sample_relation_or_components`
Controls the sampling strategy:
- `"both"`: Components + optional relation (default)
- `"relation_only"`: Only relations
- `"components_only"`: Only components
- `"either"`: Random choice per sample (never both)

### `relation_sample_prob`
- In `"both"` mode: Probability of adding relation to components
- In `"either"` mode: Probability of choosing relation vs components
- In `"relation_only"` or `"components_only"`: Ignored

### `negative_relation_sample_prob`
Probability of using negative relation vs negative component for negative sampling

### `inplace_replacement_prob`
Probability of replacing component in-place within original caption for negatives

### `num_component_captions`
Number of additional component-based captions to sample (0 = disabled)

---

## Research Questions

1. **Does "either" mode outperform "both" mode?**
   - Hypothesis: Clean separation might improve learning
   - Test: Compare `laion400m.yaml` vs `laion400m_either.yaml`

2. **What's the optimal relation_sample_prob for "either" mode?**
   - Test: Try 0.3, 0.5, 0.7, 0.9
   - Find: Which distribution performs best?

3. **Can pure relational learning match compositional learning?**
   - Compare: `laion400m_relation_only.yaml` vs `laion400m_components_only.yaml`

---

## See Also

- `RELATION_OR_COMPONENTS_SAMPLING.md` - Comprehensive guide
- `RELATION_OR_COMPONENTS_QUICKREF.md` - Quick reference
- `FOUR_MODE_SAMPLING_IMPLEMENTATION.md` - Implementation details
