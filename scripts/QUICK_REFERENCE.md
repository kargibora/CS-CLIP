# 🚀 Quick Reference Card

## Run the Script

```bash
cd scripts
python enhanced_paper_visualization.py
```

**Output**: `./paper_figures_enhanced/` with 16 files (8 PNG + 8 PDF)

## What You Get

📊 **8 Publication-Ready Figures**:

1. `00_overview_summary` - Overview of all capabilities
2. `capability_attribute_recognition` - 🎨 ColorFoil, SugarCrepe, etc.
3. `capability_attribute_binding` - 🔗 ARO, ColorSwap, Winoground, etc.
4. `capability_relations` - 📐 **ControlledImages**, SPEC, VALSE, etc.
5. `capability_quantitative` - 🔢 SPEC, VisMin, VALSE
6. `capability_existence_and_negation` - 🚫 **NegBench**, SPEC, VALSE
7. `capability_object_and_role` - 👤 SugarCrepe, VL_CheckList, etc.
8. `capability_linguistic` - 📝 BLA, ARO, ColorSwap

## Key Features

✅ **Original vs Foil** - Clear ✓/✗ distinction
✅ **Color-Coded** - Each capability has unique color
✅ **All Datasets** - Including NegBench & ControlledImages
✅ **All Subsets** - VG-One, VG-Two, COCO-One, COCO-Two, etc.
✅ **High Quality** - 300 DPI PNG + vector PDF

## Capability Colors

| Capability | Color | Icon |
|------------|-------|------|
| Attribute Recognition | 🟥 Red | 🎨 |
| Attribute Binding | 🟦 Blue | 🔗 |
| Relations | 🟩 Green | 📐 |
| Quantitative | 🟧 Orange | 🔢 |
| Existence & Negation | 🟪 Pink | 🚫 |
| Object & Role | 🟣 Purple | 👤 |
| Linguistic | 🟦 Teal | 📝 |

## Quick Customization

### Change sample counts:
Edit `sample_config` in `main()`:
```python
'NegBench': 4,  # More samples
```

### Change output directory:
```python
visualizer = PaperQualityVisualizer(output_dir="./my_figures")
```

### Quick test (small sample):
```bash
python test_visualization.py
```

## Dataset-Capability Mapping

**Relations** 📐:
- ARO (VG_Relation)
- SPEC (spatial subsets)
- **ControlledImages** (A, B, VG-One, VG-Two, COCO-One, COCO-Two)
- VALSE (relations)
- COLA (multi_objects)

**Existence & Negation** 🚫:
- **NegBench** (msr_vtt, COCO subsets)
- SPEC (existence)
- VALSE (existence)

## File Formats

- **PNG**: For presentations (300 DPI)
- **PDF**: For LaTeX papers (vector)

## Documentation

📖 **Full Docs**: `VISUALIZATION_README.md`
📊 **Changes**: `VISUALIZATION_IMPROVEMENTS.md`
📋 **Summary**: `COMPLETE_VISUALIZATION_SUMMARY.md`

## Example Output

```
paper_figures_enhanced/
├── 00_overview_summary.png ✨ Start here!
├── 00_overview_summary.pdf
├── capability_relations.png 📐 Shows ControlledImages
├── capability_relations.pdf
├── capability_existence_and_negation.png 🚫 Shows NegBench
└── ... (14 more files)
```

## Troubleshooting

**Import Error**: Make sure you're in `scripts/` directory
**No Images**: Check BenchmarkSampler paths
**Want Fewer Samples**: Edit `sample_config` (smaller numbers)
**Want More Subsets**: They're automatically included!

---

**That's it!** Run the script and get beautiful publication-ready figures! 🎉
