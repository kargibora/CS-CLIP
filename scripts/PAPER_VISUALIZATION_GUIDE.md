# Paper-Quality Visualization Recommendations

## 📊 Current Plot Quality Assessment

### ✅ **Strengths of Current Plots**
1. **Clear caption display** - Positive and negative captions are well-differentiated with ✓/✗ symbols
2. **Metadata preservation** - Subset information and sample IDs are included
3. **Multiple format export** - Both PNG and PDF outputs for flexibility
4. **Organized structure** - Individual dataset plots + overview

### ⚠️ **Areas for Improvement**

#### 1. **Organization & Structure**
**Current Issue:** Datasets are listed alphabetically, not by what they test
- ❌ Hard to see relationships between datasets
- ❌ No clear story about evaluation coverage

**Recommendation:** Organize by **capability categories**
```
✅ Attribute Binding (ColorSwap, SugarCrepe swap_att, VG_Attribution)
✅ Spatial Relations (VG_Relation, SPEC spatial, ControlledImages)
✅ Quantitative (SPEC count/size, VisMin counting, VALSE plurality)
✅ Compositional (Winoground, VisMin, SugarCrepe)
```

#### 2. **Visual Clarity**
**Current Issues:**
- Caption text can overflow or be cut off
- No visual distinction between capability types
- White space not optimally used

**Recommendations:**
- **Color-code by capability** (e.g., blue for spatial, green for attributes)
- **Truncate long captions** (60 chars max, add ellipsis)
- **Use capability badges** at the top of each sample
- **Increase font size** for readability in print (9-10pt minimum)

#### 3. **Information Density**
**Current Issue:** Each sample shows limited context

**Recommendations:**
- Add **subset labels** prominently for multi-subset datasets
- Include **capability annotation** (what is being tested)
- Show **difficulty indicators** if available (easy/medium/hard)

#### 4. **Typography & Style**
**Current Issue:** Generic matplotlib styling

**Recommendations for Publication:**
- Use **serif fonts** (Times New Roman, Computer Modern)
- Apply **IEEE/ACM paper style** guidelines
- Set `text.usetex=True` if LaTeX available for math symbols
- Consistent **color palette** (avoid bright/garish colors)

#### 5. **Layout for Paper Appendix**
**Current Issue:** Not optimized for two-column paper format

**Recommendations:**
- **Single column width**: 3.5 inches (88mm) for IEEE papers
- **Double column width**: 7 inches (178mm) for full-page figures
- **Max 3 columns** per row for readability
- **Aspect ratio**: Keep images square or 4:3 for consistency

## 🎨 Enhanced Visualization Script

I've created `enhanced_paper_visualization.py` with these improvements:

### Key Features:

#### 1. **Capability-Based Organization**
```python
CAPABILITY_CATEGORIES = {
    'Attribute Binding': {
        'datasets': ['ColorSwap', 'SugarCrepe', 'VG_Attribution'],
        'color': '#3498DB',  # Distinctive color
        'description': 'Multi-object attribute binding'
    },
    # ... more categories
}
```

#### 2. **Clean, Publication-Ready Styling**
- Serif fonts (Times New Roman)
- IEEE paper dimensions
- Color-coded capability headers
- Clear positive/negative caption formatting

#### 3. **Multiple Output Formats**
- **PNG** at 300 DPI (for Word/PowerPoint)
- **PDF** vector format (for LaTeX papers)
- **SVG** for web (if needed)

#### 4. **Smart Caption Handling**
- Automatic text wrapping at 60 characters
- Ellipsis for long captions
- Clear visual hierarchy (positive in green, negative in red)

#### 5. **Capability Overview Figure**
- Shows dataset organization by capability
- Sample counts per category
- Clear taxonomy for readers

## 📝 Recommended Appendix Structure

### **Appendix A: Dataset Sample Illustrations**

**A.1 Overview of Evaluation Capabilities**
- Figure A1: Capability taxonomy with dataset mapping
- Table A1: Dataset statistics by capability

**A.2 Sample Illustrations by Capability**

**Figure A2: Attribute Binding Samples**
- ColorSwap (2 samples) - color-object binding via swaps
- VG_Attribution (3 samples) - attribute-object binding
- SugarCrepe swap_att (2 samples) - attribute swapping

**Figure A3: Spatial Relation Samples**
- VG_Relation (3 samples) - visual relations
- SPEC_I2T spatial (2 samples) - spatial reasoning
- ControlledImages (2 samples) - controlled spatial changes

**Figure A4: Quantitative Reasoning Samples**
- SPEC_I2T count/size (3 samples)
- VisMin counting (1 sample)
- VALSE plurality (2 samples)

**Figure A5: Compositional Reasoning Samples**
- Winoground (4 samples) - broad compositional
- SugarCrepe (3 samples) - compositional negatives
- VisMin (2 samples) - minimal pair compositions

**Figure A6: Linguistic Reasoning Samples**
- COCO_Order (2 samples) - word order
- BLA (3 samples) - syntax variations
- VALSE linguistic phenomena (3 samples)

## 🚀 Usage Instructions

### Basic Usage (Enhanced Version):
```bash
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally
python scripts/enhanced_paper_visualization.py
```

### Output:
```
paper_figures/
├── 00_capability_overview.{png,pdf}
├── sugarcrepe_samples.{png,pdf}
├── spec_i2t_samples.{png,pdf}
├── vg_attribution_samples.{png,pdf}
└── ... (all datasets)
```

### For LaTeX Papers:
```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{paper_figures/00_capability_overview.pdf}
    \caption{Overview of dataset samples organized by evaluation capability.}
    \label{fig:dataset_overview}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{paper_figures/sugarcrepe_samples.pdf}
    \caption{SugarCrepe dataset samples testing compositional reasoning.}
    \label{fig:sugarcrepe}
\end{figure}
```

## 🎯 Additional Recommendations

### 1. **Add Statistical Information**
Include in figure captions:
- Dataset size
- Number of subsets
- Negative sampling strategy
- Performance baselines

Example:
> *Figure A2: SugarCrepe samples (N=7 subsets, 28K total samples). Shows hard negative generation via attribute swapping. CLIP accuracy: 72.3%.*

### 2. **Highlight Key Differences**
Use annotations to show:
- What makes negative hard (circle differing words)
- Visual features models miss (bounding boxes)
- Capability being tested (labels)

### 3. **Create Comparison Figures**
Show model predictions:
```
Image | Positive | Negative | Model Choice | Result
```

### 4. **Consider Interactive Appendix**
For online papers:
- HTML version with zoom
- Clickable capability categories
- Filterable by dataset/capability

## 📋 Checklist for Paper Submission

- [ ] All figures use consistent fonts (serif, 9-10pt minimum)
- [ ] Color palette is colorblind-friendly
- [ ] Figures work in grayscale (for print)
- [ ] All text is readable at 50% zoom
- [ ] Captions are informative and self-contained
- [ ] File sizes are reasonable (<2MB per figure)
- [ ] Vector formats (PDF) provided for main figures
- [ ] High-res raster (300 DPI PNG) for supplementary
- [ ] Figure numbers are sequential and referenced in text
- [ ] Capability categories are defined in main text

## 🔧 Quick Fixes for Current Plots

If you want to improve existing plots without running new script:

### Fix 1: Better Caption Formatting
```python
# Truncate long captions
caption = caption[:60] + '...' if len(caption) > 60 else caption

# Better wrapping
import textwrap
wrapped = '\n'.join(textwrap.wrap(caption, width=40))
```

### Fix 2: Add Capability Colors
```python
capability_colors = {
    'SugarCrepe': '#3498DB',    # Blue for compositional
    'VG_Attribution': '#E74C3C', # Red for attributes
    'SPEC_I2T': '#2ECC71',      # Green for spatial
}

# Add colored border
ax.add_patch(Rectangle((0, 1.02), 1, 0.03,
                       facecolor=capability_colors.get(dataset, '#95A5A6'),
                       transform=ax.transAxes, clip_on=False))
```

### Fix 3: Export High-Quality PDFs
```python
plt.savefig(path, format='pdf', 
           bbox_inches='tight',
           dpi=300,  # For rasterized elements
           metadata={'Creator': 'Your Paper', 'Author': 'Your Name'})
```

## 📚 References for Good Paper Figures

1. **IEEE Visualization Guidelines**: [link](https://www.computer.org/csdl/magazine/cg/2023/01/10005405/1JWWCpDdOO4)
2. **Nature Figure Guidelines**: Clean, self-explanatory figures
3. **Ten Simple Rules for Better Figures**: PLOS Computational Biology
4. **ColorBrewer**: Colorblind-safe palettes

## 💡 Final Tips

1. **Test print your figures** - Colors look different on screen vs paper
2. **Get feedback early** - Show to colleagues before submission
3. **Version control** - Keep source code for generating figures
4. **Document everything** - Random seeds, hyperparameters, etc.
5. **Make it reproducible** - Include scripts in supplementary materials

---

**Next Steps:**
1. Run `python scripts/enhanced_paper_visualization.py`
2. Review generated figures in `paper_figures/`
3. Adjust capability categories if needed
4. Create LaTeX/Word document with figures
5. Get feedback from co-authors
