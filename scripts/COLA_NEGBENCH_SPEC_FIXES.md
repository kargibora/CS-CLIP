# 🔧 COLA, NegBench & SPEC Image Fixes

## Issues Resolved

### 1. ✅ Added COLA Dataset Support
**Problem**: 
```
📊 Sampling 3 examples from COLA...
❌ Unknown dataset: COLA
```

**Root Cause**: COLA was not in the `dataset_configs` dictionary in `generate_sampling_plot.py`

**Solution**: Added COLA configuration
```python
'COLA': {'subset': 'multi_objects', 'path': 'cola'},
```

**Details**:
- Uses `COLAMultiObjectDataset` by default
- Loads from `./datasets/cola`
- Subset: `multi_objects` (contrastive matching setting)

### 2. ✅ Added NegBench Dataset Support
**Problem**:
```
📊 Sampling 4 examples from NegBench...
❌ Unknown dataset: NegBench
```

**Root Cause**: NegBench was not in the `dataset_configs` dictionary

**Solution**: Added NegBench configuration
```python
'NegBench': {'subset': 'msr_vtt_mcq_rephrased_llama', 'path': 'negbench'}
```

**Details**:
- Uses `NegBenchDataset`
- Loads from `./datasets/negbench`
- Default subset: `msr_vtt_mcq_rephrased_llama`
- Other subset available: `COCO_val_mcq_llama3.1_rephrased`

### 3. ✅ Fixed SPEC Image Type Errors
**Problem**:
```
"Unsupported image type" for SPEC datasets
```

**Root Cause**: SPEC_I2T returns images in various formats that weren't handled:
- Could be lists containing images
- Could be dict-like objects with nested 'image' key
- Various tensor/array formats

**Solution**: Enhanced `_convert_image()` function with better handling:

**Before**:
```python
def _convert_image(self, image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        # ... conversion
    if isinstance(image, np.ndarray):
        # ... conversion
    raise ValueError(f"Unsupported image type: {type(image)}")
```

**After**:
```python
def _convert_image(self, image) -> Image.Image:
    # Handle list of images (take first one)
    if isinstance(image, list):
        if len(image) > 0:
            image = image[0]
        else:
            raise ValueError("Empty image list")
    
    if isinstance(image, Image.Image):
        return image
        
    if isinstance(image, torch.Tensor):
        # ... conversion
        
    if isinstance(image, np.ndarray):
        # ... conversion
    
    # Handle dict-like objects
    if hasattr(image, 'get') or isinstance(image, dict):
        if 'image' in image:
            return self._convert_image(image['image'])  # Recursive
        
    raise ValueError(f"Unsupported image type: {type(image)}")
```

**Key Improvements**:
1. **List handling**: Extracts first image from lists
2. **Dict handling**: Recursively extracts 'image' key from dict-like objects
3. **Better error messages**: More specific about what failed

## Updated Dataset Configuration

```python
# In scripts/generate_sampling_plot.py
self.dataset_configs = {
    # ... existing configs ...
    'SPEC_I2T': {'subset': 'all', 'path': 'SPEC'},
    'COLA': {'subset': 'multi_objects', 'path': 'cola'},           # ✅ NEW
    'NegBench': {'subset': 'msr_vtt_mcq_rephrased_llama', 'path': 'negbench'}  # ✅ NEW
}
```

## Testing

Run the visualization script:
```bash
cd scripts
python enhanced_paper_visualization.py
```

**Expected Results**:
✅ COLA samples loaded successfully
✅ NegBench samples loaded successfully  
✅ No "Unsupported image type" errors for SPEC_I2T
✅ All datasets visualize correctly

## Dataset Details

### COLA (Compose Objects Localized with Attributes)
- **Capability**: Relations (multi-object composition)
- **Subsets**: 
  - `multi_objects` (default) - Contrastive matching
  - `single_GQA`, `single_CLEVR`, `single_PACO` - Single object recognition
- **Format**: Image pairs with swapped attributes
- **Paper**: https://arxiv.org/abs/2305.03689

### NegBench (Negation Benchmark)
- **Capability**: Existence & Negation
- **Subsets**:
  - `msr_vtt_mcq_rephrased_llama` (default) - Video captions
  - `COCO_val_mcq_llama3.1_rephrased` - COCO captions
- **Format**: Multiple choice with negation understanding
- **Tests**: Models' ability to understand logical negation

### SPEC_I2T Image Formats
SPEC returns images in multiple formats:
- **Tensor format**: `torch.Tensor [C, H, W]` or `[1, C, H, W]`
- **Array format**: `numpy.ndarray`
- **List format**: `[PIL.Image]` or `[torch.Tensor]`
- **Dict format**: `{'image': <actual_image>}`

The enhanced converter now handles all these cases gracefully.

## Capability Mapping

Both datasets are properly mapped in `CAPABILITY_CATEGORIES`:

```python
'Relations': {
    'datasets': {
        # ...
        'COLA': ['multi_objects']  # ✅ COLA added
    }
},

'Existence & Negation': {
    'datasets': {
        # ...
        'NegBench': ['msr_vtt_mcq_rephrased_llama', 'COCO_val_mcq_llama3.1_rephrased']  # ✅ NegBench added
    }
}
```

---

**Status**: ✅ All issues resolved
**Date**: November 3, 2025
**Files Modified**: 
- `scripts/generate_sampling_plot.py` (added COLA and NegBench configs)
- `scripts/enhanced_paper_visualization.py` (enhanced image converter)
