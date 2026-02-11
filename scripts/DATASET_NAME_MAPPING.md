# Dataset Name Mapping Reference

## Overview

This document clarifies the relationship between **display names** used in papers/results and **actual dataset class names** used in the codebase.

## ARO (Display Name) → Actual Dataset Classes

**ARO does NOT exist as a dataset class.** It's a collection of 4 datasets:

| Display Name | Actual Class Name | Capability Category |
|--------------|-------------------|---------------------|
| ARO (VG_Attribution) | `VG_Attribution` | Attribute Binding |
| ARO (VG_Relation) | `VG_Relation` | Relations |
| ARO (COCO_Order) | `COCO_Order` | Linguistic |
| ARO (Flickr30k_Order) | `Flickr30k_Order` | Linguistic |

### Usage in Code

```python
# ❌ WRONG - ARO doesn't exist as a class
dataset = ARO(...)

# ✅ CORRECT - Use actual class names
dataset = VG_Attribution(...)
dataset = VG_Relation(...)
dataset = COCO_Order(...)
dataset = Flickr30k_Order(...)
```

### Usage in Sample Config

```python
# ❌ WRONG
sample_config = {
    'ARO': 4,  # This won't work!
}

# ✅ CORRECT
sample_config = {
    'VG_Attribution': 4,   # Attribute binding
    'VG_Relation': 4,      # Relations
    'COCO_Order': 3,       # Linguistic
    'Flickr30k_Order': 3,  # Linguistic
}
```

## Other Name Variations

### SPEC vs SPEC_I2T

| Display Name | Actual Class Name | Usage |
|--------------|-------------------|-------|
| SPEC | `SPECImage2TextDataset` | Both names work |
| SPEC_I2T | `SPECImage2TextDataset` | Preferred in code |

```python
# Both work (mapped in get_dataset_class):
sample_config = {
    'SPEC': 8,        # Works
    'SPEC_I2T': 8,    # Also works (preferred)
}
```

### SugarCrepe++ vs SugarCrepe_PP

| Display Name | Actual Class Name | Usage |
|--------------|-------------------|-------|
| SugarCrepe++ | `SugarCrepePPDataset` | Display only |
| SugarCrepe_PP | `SugarCrepePPDataset` | Use in code |

```python
# ❌ WRONG
sample_config = {
    'SugarCrepe++': 4,  # Won't work
}

# ✅ CORRECT
sample_config = {
    'SugarCrepe_PP': 4,
}
```

### COCO-CF vs COCO_Counterfactuals

| Display Name | Actual Class Name | Usage |
|--------------|-------------------|-------|
| COCO-CF | `COCOCounterfactualsDataset` | Display only |
| COCO_Counterfactuals | `COCOCounterfactualsDataset` | Use in code |

```python
# ❌ WRONG
sample_config = {
    'COCO-CF': 3,  # Won't work
}

# ✅ CORRECT
sample_config = {
    'COCO_Counterfactuals': 3,
}
```

## Complete Dataset Name Reference

### Always Use These Names in Code

| Code Name | Display Name | Class |
|-----------|--------------|-------|
| `VG_Attribution` | ARO (Attribution) | `VG_Attribution` |
| `VG_Relation` | ARO (Relation) | `VG_Relation` |
| `COCO_Order` | ARO (COCO Order) | `COCO_Order` |
| `Flickr30k_Order` | ARO (Flickr Order) | `Flickr30k_Order` |
| `SPEC_I2T` | SPEC | `SPECImage2TextDataset` |
| `SugarCrepe` | SugarCrepe | `SugarCrepeDataset` |
| `SugarCrepe_PP` | SugarCrepe++ | `SugarCrepePPDataset` |
| `COCO_Counterfactuals` | COCO-CF | `COCOCounterfactualsDataset` |
| `ControlledImages` | WhatsUp | `Controlled_Images` |
| `VL_CheckList` | VL-CheckList | `VLCheckListDataset` |
| `ColorSwap` | ColorSwap | `ColorSwapDataset` |
| `ColorFoil` | ColorFoil | `ColorFoilDataset` |
| `VisMin` | VisMin | `VisMinDataset` |
| `Winoground` | Winoground | `WinogroundDataset` |
| `BLA` | BLA | `BLADataset` |
| `VALSE` | VALSE | `VALSEDataset` |
| `NegBench` | NegBench | `NegBenchDataset` |
| `COLA` | COLA | `COLAMultiObjectDataset` |
| `CC3M` | CC3M | `CC3MDataset` |
| `LAION400M` | LAION400M | `LAION400MDataset` |

## Capability Mapping (Correct)

```python
CAPABILITY_CATEGORIES = {
    'Attribute Binding': {
        'datasets': {
            'VG_Attribution': ['all'],  # ✅ Correct (not 'ARO')
        }
    },
    'Relations': {
        'datasets': {
            'VG_Relation': ['all'],  # ✅ Correct (not 'ARO')
        }
    },
    'Linguistic': {
        'datasets': {
            'COCO_Order': ['all'],       # ✅ Correct (not 'ARO')
            'Flickr30k_Order': ['all'],  # ✅ Correct (not 'ARO')
        }
    }
}
```

## Example: Correct Sample Configuration

```python
sample_config = {
    # Attribute Recognition
    'ColorFoil': 3,
    'SugarCrepe': 8,
    'VL_CheckList': 6,
    
    # Attribute Binding
    'VG_Attribution': 4,      # ✅ Not 'ARO'
    'ColorSwap': 3,
    'VisMin': 6,
    'Winoground': 3,
    
    # Relations
    'VG_Relation': 4,         # ✅ Not 'ARO'
    'SPEC_I2T': 8,            # ✅ Not 'SPEC'
    'VALSE': 10,
    'ControlledImages': 6,
    'COLA': 3,
    
    # Existence & Negation
    'NegBench': 4,
    
    # Object & Role
    'COCO_Counterfactuals': 3,  # ✅ Not 'COCO-CF'
    
    # Linguistic
    'BLA': 4,
    'COCO_Order': 3,          # ✅ Not 'ARO'
    'Flickr30k_Order': 3,     # ✅ Not 'ARO'
}
```

## Why This Matters

1. **Import Statements**: Classes use actual names
   ```python
   from .aro import VG_Attribution, VG_Relation, COCO_Order, Flickr30k_Order
   # Note: No class called 'ARO'
   ```

2. **get_dataset_class()**: Maps code names to classes
   ```python
   "VG_Attribution": VG_Attribution,  # ✅ Works
   "ARO": ???,                        # ❌ Doesn't exist
   ```

3. **BenchmarkSampler**: Expects actual class names
   ```python
   sampler.sample_from_datasets({'VG_Attribution': 4})  # ✅ Works
   sampler.sample_from_datasets({'ARO': 4})             # ❌ Fails
   ```

## Quick Check: Is My Name Correct?

Ask yourself:
1. ✅ Is there an import for this name? → Use it
2. ❌ Is it a "collection" name (like ARO)? → Use individual datasets
3. ✅ Is it in `get_dataset_class()`? → Use it
4. ❌ Does it have special characters (-,+)? → Replace with underscores

## Summary

**Rule of Thumb**: Use the names that appear in the **import statements** and **get_dataset_class()** mapping.

**ARO is special**: It's a paper grouping, not a dataset class. Always use the 4 individual datasets instead.

---

**Updated**: November 3, 2025
