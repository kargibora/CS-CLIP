# Enhanced Paper Visualization Script

## Overview

The `enhanced_paper_visualization.py` script creates publication-ready visualizations of your evaluation benchmarks, organized by capability categories.

## Features

✨ **Key Improvements:**

1. **Capability-Based Organization**: Groups datasets by what they test (Attribute Binding, Relations, etc.)
2. **Beautiful Layout**: Modern, clean design with color-coding
3. **Clear Comparison**: Shows original caption vs foil caption side-by-side
4. **Comprehensive Coverage**: Includes all datasets with proper subset handling
5. **Multiple Formats**: Exports to both PNG (high-res) and PDF (vector)
6. **Professional Quality**: Publication-ready for papers

## Capability Categories

The script organizes 7 capability categories:

| Category | Icon | Datasets | Description |
|----------|------|----------|-------------|
| **Attribute Recognition** | 🎨 | ColorFoil, SugarCrepe, VL_CheckList | Single attribute recognition |
| **Attribute Binding** | 🔗 | ARO, ColorSwap, SugarCrepe, VisMin, Winoground | Multi-object attribute binding |
| **Relations** | 📐 | ARO, SPEC, VisMin, SugarCrepe, VALSE, ControlledImages, COLA | Spatial & relational understanding |
| **Quantitative** | 🔢 | SPEC, VisMin, VALSE | Counting, plurality & size |
| **Existence & Negation** | 🚫 | SPEC, VALSE, NegBench | Object existence & negation |
| **Object & Role** | 👤 | VisMin, SugarCrepe, VL_CheckList, VALSE, COCO-CF | Object recognition & roles |
| **Linguistic** | 📝 | ColorSwap, BLA, ARO | Word order, paraphrase & syntax |

## Usage

### Basic Usage

```bash
cd scripts
python enhanced_paper_visualization.py
```

Output will be saved to `./paper_figures_enhanced/`

### What Gets Generated

1. **Overview Summary** (`00_overview_summary.{png,pdf}`)
   - Beautiful overview of all capability categories
   - Shows datasets, sample counts, and descriptions

2. **Capability Plots** (one per category)
   - `capability_attribute_recognition.{png,pdf}`
   - `capability_attribute_binding.{png,pdf}`
   - `capability_relations.{png,pdf}`
   - `capability_quantitative.{png,pdf}`
   - `capability_existence_and_negation.{png,pdf}`
   - `capability_object_and_role.{png,pdf}`
   - `capability_linguistic.{png,pdf}`

Each capability plot shows:
- 6 example samples from various datasets
- Original caption (✓ in green box)
- Foil caption (✗ in red box)
- Dataset and subset labels
- Color-coded borders

## Customization

### Modify Sample Counts

Edit the `sample_config` dictionary in `main()`:

```python
sample_config = {
    'ColorFoil': 3,        # Increase to 5 for more samples
    'SugarCrepe': 8,       # Covers multiple subsets
    'ControlledImages': 6, # VG-One, VG-Two, COCO-One, etc.
    # ... add more datasets
}
```

### Change Output Directory

```python
visualizer = PaperQualityVisualizer(output_dir="./my_custom_figures")
```

### Adjust Figure Size

In `create_capability_category_plot()`:

```python
fig = plt.figure(figsize=(16, 10), facecolor='white')  # Adjust (width, height)
```

### Modify Colors

Edit `CAPABILITY_CATEGORIES` at the top of the file:

```python
'Attribute Recognition': {
    'color': '#E74C3C',  # Change to your preferred color
    'icon': '🎨',        # Change icon
    # ...
}
```

## Dataset-Subset Mapping

The script properly handles datasets with multiple subsets:

```python
CAPABILITY_CATEGORIES = {
    'Relations': {
        'datasets': {
            'ControlledImages': ['A', 'B', 'VG-One', 'VG-Two', 'COCO-One', 'COCO-Two'],
            'VALSE': ['relations'],
            'SPEC': ['relative_spatial', 'absolute_position'],
            # ...
        }
    }
}
```

## Output Format

### File Naming

- Overview: `00_overview_summary.{png,pdf}`
- Capabilities: `capability_{name}.{png,pdf}`

### Resolution

- **PNG**: 300 DPI (high resolution for presentations)
- **PDF**: Vector format (perfect for LaTeX papers)

## Example Output

```
paper_figures_enhanced/
├── 00_overview_summary.png
├── 00_overview_summary.pdf
├── capability_attribute_recognition.png
├── capability_attribute_recognition.pdf
├── capability_attribute_binding.png
├── capability_attribute_binding.pdf
├── capability_relations.png
├── capability_relations.pdf
├── capability_quantitative.png
├── capability_quantitative.pdf
├── capability_existence_and_negation.png
├── capability_existence_and_negation.pdf
├── capability_object_and_role.png
├── capability_object_and_role.pdf
├── capability_linguistic.png
└── capability_linguistic.pdf
```

## Troubleshooting

### Import Error: `generate_sampling_plot`

Make sure you're running from the `scripts/` directory:

```bash
cd scripts
python enhanced_paper_visualization.py
```

### Missing Datasets

Check that your dataset paths are configured correctly in the BenchmarkSampler.

### Image Display Issues

If images don't display correctly, check the image format conversion in `_convert_image()`.

## Advanced Features

### Add New Capability Category

1. Add to `CAPABILITY_CATEGORIES`:

```python
'My New Capability': {
    'description': 'Description here',
    'color': '#FF5733',  # Choose a color
    'icon': '🚀',        # Choose an icon
    'datasets': {
        'MyDataset': ['subset1', 'subset2'],
    }
}
```

2. Run the script - it will automatically create a plot for the new category!

### Filtering by Subset

The script automatically filters samples based on subset matching. If you want specific subsets only:

```python
# In CAPABILITY_CATEGORIES
'Relations': {
    'datasets': {
        'ControlledImages': ['VG-One', 'VG-Two'],  # Only these subsets
    }
}
```

## Tips for Paper Figures

1. **Use PDF for LaTeX**: Vector format scales perfectly
2. **Use PNG for PowerPoint**: High-res raster for presentations
3. **Keep 6 samples per category**: Good balance of coverage and readability
4. **Color-code consistently**: Use the same colors across all figures
5. **Include subset labels**: Helps readers understand what's being tested

## Performance

- Typical runtime: 10-30 seconds
- Depends on number of datasets and samples
- GPU not required (uses CPU for image processing)

## Dependencies

- matplotlib
- numpy
- PIL/Pillow
- torch
- Your `generate_sampling_plot.py` (BenchmarkSampler)

## Support

For issues or questions, check the main repository documentation or create an issue.

---

**Happy Visualizing!** 🎨📊
