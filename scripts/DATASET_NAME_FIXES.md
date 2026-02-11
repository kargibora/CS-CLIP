# 🔧 Dataset Name Fixes Applied

## Problem Identified

The original script used incorrect dataset names:
- ❌ `'ARO'` - Doesn't exist as a dataset class
- ❌ `'SugarCrepe++'` - Wrong name format
- ❌ `'COCO-CF'` - Wrong name format
- ❌ `'SPEC'` - Should be `'SPEC_I2T'`

## Solution Applied

### 1. Fixed ARO References

**Before**:
```python
'Attribute Binding': {
    'datasets': {
        'ARO': ['VG_Attribution'],  # ❌ Wrong
    }
}
```

**After**:
```python
'Attribute Binding': {
    'datasets': {
        'VG_Attribution': ['all'],  # ✅ Actual dataset class name
    }
}
```

**All ARO Fixes**:
- `ARO` (VG_Attribution) → `VG_Attribution`
- `ARO` (VG_Relation) → `VG_Relation`
- `ARO` (COCO_Order) → `COCO_Order`
- `ARO` (Flickr30k_Order) → `Flickr30k_Order`

### 2. Fixed SugarCrepe++

**Before**: `'SugarCrepe++'`
**After**: `'SugarCrepe_PP'`

### 3. Fixed COCO-CF

**Before**: `'COCO-CF'`
**After**: `'COCO_Counterfactuals'`

### 4. Fixed SPEC

**Before**: `'SPEC'`
**After**: `'SPEC_I2T'`

## Updated CAPABILITY_CATEGORIES

```python
CAPABILITY_CATEGORIES = {
    'Attribute Recognition': {
        'datasets': {
            'ColorFoil': ['all'],
            'SugarCrepe': ['replace_att', 'add_att'],
            'SugarCrepe_PP': ['replace_att'],  # ✅ Fixed
            'VL_CheckList': ['vg_color', 'vaw_color'],
        }
    },
    'Attribute Binding': {
        'datasets': {
            'VG_Attribution': ['all'],  # ✅ Fixed (was 'ARO')
            'ColorSwap': ['all'],
            'SugarCrepe': ['swap_att'],
            'SugarCrepe_PP': ['swap_att'],  # ✅ Fixed
            'VisMin': ['attribute'],
            'Winoground': ['all']
        }
    },
    'Relations': {
        'datasets': {
            'VG_Relation': ['all'],  # ✅ Fixed (was 'ARO')
            'SPEC_I2T': ['relative_spatial', 'absolute_position'],  # ✅ Fixed
            'VisMin': ['relation'],
            'SugarCrepe': ['replace_rel', 'swap_rel'],
            'SugarCrepe_PP': ['replace_rel', 'swap_rel'],  # ✅ Fixed
            'VL_CheckList': ['vg_spatial'],
            'VALSE': ['relations'],
            'ControlledImages': ['A', 'B', 'VG-One', 'VG-Two', 'COCO-One', 'COCO-Two'],
            'COLA': ['multi_objects']
        }
    },
    'Quantitative': {
        'datasets': {
            'SPEC_I2T': ['count', 'absolute_size', 'relative_size'],  # ✅ Fixed
            'VisMin': ['counting'],
            'VALSE': ['counting', 'plurals'],
        }
    },
    'Existence & Negation': {
        'datasets': {
            'SPEC_I2T': ['existence'],  # ✅ Fixed
            'VALSE': ['existence'],
            'NegBench': ['msr_vtt_mcq_rephrased_llama', 'COCO_val_mcq_llama3.1_rephrased'],
        }
    },
    'Object & Role': {
        'datasets': {
            'VisMin': ['object'],
            'SugarCrepe': ['replace_obj', 'swap_obj', 'add_obj'],
            'SugarCrepe_PP': ['replace_obj', 'swap_obj', 'add_obj'],  # ✅ Fixed
            'VL_CheckList': ['hake_action', 'swig_action', 'vg_action', 'vaw_action'],
            'VALSE': ['actions', 'coreference', 'noun phrases'],
            'COCO_Counterfactuals': ['all']  # ✅ Fixed (was 'COCO-CF')
        }
    },
    'Linguistic': {
        'datasets': {
            'ColorSwap': ['all'],
            'BLA': ['ap', 'co', 'rc'],
            'COCO_Order': ['all'],  # ✅ Fixed (was 'ARO')
            'Flickr30k_Order': ['all'],  # ✅ Fixed (was 'ARO')
        }
    }
}
```

## Updated Sample Configuration

```python
sample_config = {
    # Attribute Recognition
    'ColorFoil': 3,
    'SugarCrepe': 8,
    'VL_CheckList': 6,
    
    # Attribute Binding
    'VG_Attribution': 4,      # ✅ Fixed (was 'ARO')
    'ColorSwap': 3,
    'VisMin': 6,
    'Winoground': 3,
    
    # Relations
    'VG_Relation': 4,         # ✅ Fixed (was 'ARO')
    'SPEC_I2T': 8,            # ✅ Fixed (was 'SPEC')
    'VALSE': 10,
    'ControlledImages': 6,
    'COLA': 3,
    
    # Existence & Negation
    'NegBench': 4,
    
    # Object & Role
    'COCO_Counterfactuals': 3,  # ✅ Fixed (was 'COCO-CF')
    
    # Linguistic
    'BLA': 4,
    'COCO_Order': 3,          # ✅ Fixed (was 'ARO')
    'Flickr30k_Order': 3,     # ✅ Fixed (was 'ARO')
    
    # Additional
    'CC3M': 2,
}
```

## Verification

All names now match the actual dataset classes defined in `__init__.py`:

```python
from .aro import VG_Attribution, VG_Relation, COCO_Order, Flickr30k_Order  # ✅
from .sugarcrepe_pp import SugarCrepePPDataset  # ✅
from .coco_counterfactuals import COCOCounterfactualsDataset  # ✅
from .spec import SPECImage2TextDataset  # ✅
```

And mapped correctly in `get_dataset_class()`:

```python
dataset_classes = {
    "VG_Attribution": VG_Attribution,  # ✅
    "VG_Relation": VG_Relation,  # ✅
    "COCO_Order": COCO_Order,  # ✅
    "Flickr30k_Order": Flickr30k_Order,  # ✅
    "SPEC_I2T": SPECImage2TextDataset,  # ✅
    "SugarCrepe_PP": SugarCrepePPDataset,  # ✅
    "COCO_Counterfactuals": COCOCounterfactualsDataset,  # ✅
}
```

## Testing

The script is now ready to run:

```bash
cd scripts
python enhanced_paper_visualization.py
```

Expected behavior:
✅ All dataset names will be recognized by BenchmarkSampler
✅ Samples will be correctly grouped by capability
✅ ARO datasets will appear under their proper categories
✅ No "dataset not found" errors

## Documentation Created

1. **`DATASET_NAME_MAPPING.md`** - Complete reference guide
   - Explains ARO → individual datasets mapping
   - Lists all correct dataset names
   - Shows before/after examples

2. **Updated script comments** - Inline clarifications

## Summary

✅ **7 name fixes** applied across CAPABILITY_CATEGORIES
✅ **4 ARO references** split into individual datasets
✅ **Sample config updated** with correct names
✅ **Removed "Other" category fallback** - datasets must be explicitly categorized
✅ **Documentation created** to prevent future confusion
✅ **Script ready to run** with BenchmarkSampler

The visualization script now uses the exact dataset names that exist in your codebase!

**Important**: Any dataset not found in CAPABILITY_CATEGORIES will be skipped with a warning message.

---

**Status**: ✅ Ready to use
**Updated**: November 3, 2025
