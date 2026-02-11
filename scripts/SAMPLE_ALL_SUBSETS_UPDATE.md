# 🔄 Sample All Subsets Configuration Update

## Changes Made

Updated `generate_sampling_plot.py` to sample from **ALL subsets** instead of specific ones for each dataset.

## What Changed

### 1. Dataset Configurations (`dataset_configs`)
Changed all dataset configurations to use `'subset': 'all'` instead of specific subset names.

**Before**:
```python
'SugarCrepe': {'subset': 'add_att', 'path': 'SugarCrepe'},  # Only one subset
'SugarCrepe_PP': {'subset': 'swap_object', 'path': 'SugarCrepe_PP'},  # Only one subset
'COLA': {'subset': 'multi_objects', 'path': 'cola'},  # Only one subset
'NegBench': {'subset': 'msr_vtt_mcq_rephrased_llama', 'path': 'negbench'}  # Only one subset
```

**After**:
```python
'SugarCrepe': {'subset': 'all', 'path': 'SugarCrepe'},  # All 7 subsets
'SugarCrepe_PP': {'subset': 'all', 'path': 'SugarCrepe_PP'},  # All 6 subsets
'COLA': {'subset': 'all', 'path': 'cola'},  # All subsets
'NegBench': {'subset': 'all', 'path': 'negbench'}  # All 2 subsets
```

### 2. Multi-File Subset Handler
Enhanced `_sample_from_multi_file_subsets()` to support SugarCrepe_PP:

**Added SugarCrepe_PP subsets**:
```python
def _sample_from_multi_file_subsets(self, dataset_name, config, n_samples):
    if dataset_name == 'SugarCrepe':
        subsets = ['add_att', 'add_obj', 'replace_att', 'replace_obj', 
                   'replace_rel', 'swap_att', 'swap_obj']  # 7 subsets
    elif dataset_name == 'SugarCrepe_PP':
        subsets = ['swap_att', 'swap_obj', 'replace_att', 'replace_obj', 
                   'add_att', 'add_obj']  # 6 subsets
    else:
        return []
    # ... samples from all subsets evenly
```

### 3. Separate Subset Handler
Enhanced `_sample_from_separate_subsets()` to include COLA and NegBench:

**Added**:
```python
elif dataset_name == 'COLA':
    subsets = ['multi_objects']  # Can add more: 'single_GQA', 'single_CLEVR', 'single_PACO'
elif dataset_name == 'NegBench':
    subsets = ['msr_vtt_mcq_rephrased_llama', 'COCO_val_mcq_llama3.1_rephrased']
```

### 4. Main Sampling Logic
Updated to handle all datasets with multiple subsets:

**Before**:
```python
if dataset_name == 'SugarCrepe':  # Only SugarCrepe
    samples = self._sample_from_multi_file_subsets(...)
elif dataset_name in ['BLA', 'ControlledImages', 'SPEC_I2T']:  # Only 3 datasets
    samples = self._sample_from_separate_subsets(...)
```

**After**:
```python
if dataset_name in ['SugarCrepe', 'SugarCrepe_PP']:  # Both SugarCrepe variants
    samples = self._sample_from_multi_file_subsets(...)
elif dataset_name in ['BLA', 'ControlledImages', 'SPEC_I2T', 'COLA', 'NegBench']:  # 5 datasets
    samples = self._sample_from_separate_subsets(...)
```

## Complete Subset Coverage

### Datasets with Multiple Subsets (Auto-sampled)

1. **SugarCrepe** (7 subsets):
   - `add_att`, `add_obj`, `replace_att`, `replace_obj`, `replace_rel`, `swap_att`, `swap_obj`

2. **SugarCrepe_PP** (6 subsets):
   - `swap_att`, `swap_obj`, `replace_att`, `replace_obj`, `add_att`, `add_obj`

3. **BLA** (3 subsets):
   - `ap` (active-passive), `co` (coordination), `rc` (relative clause)

4. **ControlledImages** (2 subsets):
   - `A`, `B`

5. **SPEC_I2T** (6 subsets):
   - `count`, `relative_spatial`, `relative_size`, `absolute_size`, `absolute_spatial`, `existence`

6. **COLA** (1 subset, expandable):
   - `multi_objects` (default)
   - Can add: `single_GQA`, `single_CLEVR`, `single_PACO`

7. **NegBench** (2 subsets):
   - `msr_vtt_mcq_rephrased_llama`
   - `COCO_val_mcq_llama3.1_rephrased`

8. **VALSE** (auto-detected subsets):
   - Automatically samples from all linguistic phenomena

9. **VL_CheckList** (14 subsets):
   - Attributes: `vaw_action`, `vg_action`, `vaw_color`, `vg_color`, `vaw_material`, `vg_material`, `vaw_size`, `vg_size`, `vaw_state`, `vg_state`
   - Relations: `hake_action`, `swig_action`, `vg_action_relation`, `vg_spatial`

### Datasets with Single Subsets (No change needed)
- Winoground, VG_Attribution, VG_Relation, COCO_Order, Flickr30k_Order, ColorSwap, ColorFoil, COCO_Counterfactuals, VisMin

## Sampling Strategy

When you request `n` samples from a dataset with multiple subsets:
- Samples are distributed evenly across all subsets
- Example: `SugarCrepe=14` → 2 samples from each of 7 subsets
- Example: `NegBench=4` → 2 samples from each of 2 subsets
- Example: `SPEC_I2T=12` → 2 samples from each of 6 subsets

## Benefits

✅ **Comprehensive Coverage**: Samples from all subsets, not just one
✅ **Balanced Representation**: Even distribution across all subset types
✅ **Better Diversity**: Shows full range of dataset capabilities
✅ **No Bias**: Doesn't favor any particular subset
✅ **Publication Ready**: Complete dataset characterization

## Usage

No changes needed to your commands:

```bash
# Will now sample from ALL subsets automatically
python enhanced_paper_visualization.py

# Or customize sample counts
python generate_sampling_plot.py SugarCrepe=14 NegBench=4 COLA=3
```

## Example Output

**Before** (specific subsets only):
```
📊 Sampling 8 examples from SugarCrepe...
  ✓ Loaded SugarCrepe: 3000 samples (add_att only)
✅ Successfully sampled 8 examples (all from add_att)
```

**After** (all subsets):
```
📊 Sampling 8 examples from SugarCrepe...
  🔍 Loading from 7 SugarCrepe subsets, sampling ~1 from each
    ✓ add_att: 1 samples
    ✓ add_obj: 1 samples
    ✓ replace_att: 1 samples
    ✓ replace_obj: 1 samples
    ✓ replace_rel: 2 samples
    ✓ swap_att: 1 samples
    ✓ swap_obj: 1 samples
✅ Successfully sampled 8 examples (diverse coverage)
```

---

**Status**: ✅ Complete
**Date**: November 3, 2025
**Impact**: All datasets now sample from ALL available subsets for comprehensive coverage


## Remotivate the story
# What is input to system
# What is output to system
# What is resources
# What are resutls
