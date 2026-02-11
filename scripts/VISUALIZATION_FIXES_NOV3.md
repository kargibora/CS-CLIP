# 🔧 Visualization Fixes - November 3, 2025

## Issues Resolved

### 1. ✅ Fixed Caption Overlap with Images Below
**Problem**: Captions from top row images were overlapping with images in the bottom row.

**Solution**:
- Increased vertical spacing (hspace) from `0.35` to `0.65`
- Adjusted caption starting position from `-0.02` to `-0.05` (further below image)
- Reduced gaps between label and caption text for tighter layout
- Adjusted grid top margin from `0.90` to `0.92`

**Code Changes**:
```python
# Before
gs = GridSpec(n_rows, n_cols, figure=fig,
             hspace=0.35, wspace=0.25,
             top=0.90, bottom=0.06, left=0.04, right=0.96)
caption_y = -0.02

# After
gs = GridSpec(n_rows, n_cols, figure=fig,
             hspace=0.65, wspace=0.25,  # More vertical space
             top=0.92, bottom=0.06, left=0.04, right=0.96)
caption_y = -0.05  # Further below image
```

### 2. ✅ Removed Category Descriptions
**Problem**: User indicated images should be self-explanatory without text descriptions.

**Solution**:
- Removed the subtitle with category description (`cap_info['description']`)
- Kept only the title with icon and category name
- Adjusted title position from `y=0.98` to `y=0.96`

**Code Changes**:
```python
# Before
fig.suptitle(f'{cap_icon} {capability}', 
            fontsize=22, fontweight='bold', 
            color=cap_color, y=0.98)
fig.text(0.5, 0.94, cap_info['description'],  # ❌ Removed
        ha='center', fontsize=14, style='italic', 
        color='#34495E', alpha=0.8)

# After
fig.suptitle(f'{cap_icon} {capability}', 
            fontsize=22, fontweight='bold', 
            color=cap_color, y=0.96)  # ✅ No description
```

### 3. ✅ Fixed VL_CheckList and VisMin Subset Matching
**Problem**: Warnings appeared for datasets returning `subset='all'` but capability categories only listed specific subsets.

```
⚠️  Warning: Dataset 'VL_CheckList' (subset: all) not found in any capability category - skipping
⚠️  Warning: Dataset 'VisMin' (subset: all) not found in any capability category - skipping
```

**Solution**:
- Updated `_get_capability_for_dataset()` to match when `subset == 'all'`
- When a dataset returns `subset='all'`, it means it aggregates results across all subsets
- This should match any dataset that has specific subsets listed in the category

**Code Changes**:
```python
# Before
if 'all' in subsets or subset is None or subset in subsets:
    return capability, info

# After
if 'all' in subsets or subset is None or subset in subsets or subset == 'all':
    return capability, info
```

**Explanation**: 
- `'all' in subsets` - Category allows all subsets (e.g., `{'ColorFoil': ['all']}`)
- `subset is None` - No subset specified
- `subset in subsets` - Specific subset matches (e.g., `'vg_color'` in `['vg_color', 'vaw_color']`)
- `subset == 'all'` - **NEW**: Dataset returns aggregated results over all subsets

### 4. ✅ Added Missing SPEC_I2T Subset
**Problem**: Warning for `absolute_spatial` subset not found.

```
⚠️  Warning: Dataset 'SPEC_I2T' (subset: absolute_spatial) not found in any capability category - skipping
```

**Solution**:
- Added `'absolute_spatial'` to SPEC_I2T subsets in Relations category

**Code Changes**:
```python
# Before
'SPEC_I2T': ['relative_spatial', 'absolute_position'],

# After
'SPEC_I2T': ['relative_spatial', 'absolute_position', 'absolute_spatial'],
```

### 5. ✅ Fixed BLA Dataset Caption Handling
**Problem**: BLA dataset showed "no correct caption" or "no foil caption" messages.

**Root Cause**: BLA uses different field names:
- BLA format: `caption` (correct) and `foil` (incorrect)
- Standard format: `positive_caption` and `negative_captions`

**Solution**:
- Added format detection to handle both structures
- Check for `positive_caption` first (standard format)
- Fall back to `caption`/`foil` format (BLA format)

**Code Changes**:
```python
# Before
pos_cap = sample['positive_caption']
neg_caps = sample.get('negative_captions', [])

# After
if 'positive_caption' in sample:
    pos_cap = sample['positive_caption']
    neg_caps = sample.get('negative_captions', [])
elif 'caption' in sample:
    # BLA format: caption (correct) and foil (incorrect)
    pos_cap = sample['caption']
    foil = sample.get('foil', '')
    neg_caps = [foil] if foil else []
else:
    pos_cap = "No correct caption found"
    neg_caps = []
```

## Testing

Run the script to verify all fixes:
```bash
cd scripts
python enhanced_paper_visualization.py
```

**Expected Results**:
✅ No overlap between captions and images below
✅ No category description subtitles (cleaner look)
✅ No warnings for VL_CheckList or VisMin with `subset='all'`
✅ No warnings for SPEC_I2T `absolute_spatial` subset
✅ BLA captions display correctly (no "no correct caption" messages)

## Dataset Format Reference

### Standard Format (Most Datasets)
```python
{
    'image': <PIL.Image or Tensor>,
    'positive_caption': 'The correct caption',
    'negative_captions': ['Foil 1', 'Foil 2', ...],
    'dataset_name': 'DatasetName',
    'subset': 'subset_name'
}
```

### BLA Format
```python
{
    'image': <PIL.Image or Tensor>,
    'caption': 'The correct caption',  # Note: 'caption' not 'positive_caption'
    'foil': 'The incorrect caption',   # Note: 'foil' not 'negative_captions'
    'phenomenon': 'ap',
    'image_file': '12345.jpg'
}
```

### Subset Handling
- **`subset='all'`**: Dataset aggregates results across all its subsets
- **`subset=None`**: No subset specified (matches any category listing)
- **`subset='specific_name'`**: Must match exactly in category's subset list
- **Category has `['all']`**: Matches any subset from that dataset

## Performance Impact

✅ **No performance degradation** - All changes are UI/layout adjustments
✅ **Better readability** - Increased spacing prevents visual clutter
✅ **More datasets included** - Fixed matching logic captures more samples
✅ **Cleaner presentation** - Removed unnecessary text descriptions

---

**Status**: ✅ All issues resolved
**Date**: November 3, 2025
**Files Modified**: `enhanced_paper_visualization.py`
