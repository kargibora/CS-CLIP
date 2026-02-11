# CLIP Benchmark Testing Guide

## Quick Start

Test MSCOCO (retrieval) and Flowers (classification):

```bash
python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions wds/vtab/flowers
```

**Important:** All webdataset names must have the `wds/` prefix!

## Available Datasets

### 🔍 Retrieval Datasets
- `wds/mscoco_captions` - MSCOCO image-caption pairs
- `wds/flickr8k` - Flickr8k dataset
- `wds/flickr30k` - Flickr30k dataset

### 🎯 Classification Datasets

**ImageNet Variants:**
- `wds/imagenet1k` - ImageNet-1K
- `wds/imagenetv2` - ImageNet-V2
- `wds/imagenet_sketch` - ImageNet-Sketch
- `wds/imagenet-a` - ImageNet-A (natural adversarial)
- `wds/imagenet-r` - ImageNet-R (renditions)
- `wds/imagenet-o` - ImageNet-O (out-of-distribution)

**Standard Benchmarks:**
- `wds/objectnet` - ObjectNet
- `wds/voc2007` - PASCAL VOC 2007
- `wds/voc2007_multilabel` - PASCAL VOC 2007 (multi-label)
- `wds/sun397` - SUN397 scene recognition
- `wds/cars` - Stanford Cars
- `wds/fgvc_aircraft` - FGVC Aircraft
- `wds/fer2013` - FER-2013 emotion recognition

**Small Datasets:**
- `wds/mnist` - MNIST digits
- `wds/stl10` - STL-10
- `wds/gtsrb` - German Traffic Sign Recognition
- `wds/country211` - Country211
- `wds/renderedsst2` - Rendered SST-2

### 🔬 VTAB (Visual Task Adaptation Benchmark)

**Basic:**
- `wds/vtab/caltech101` - Caltech101
- `wds/vtab/cifar10` - CIFAR-10
- `wds/vtab/cifar100` - CIFAR-100
- `wds/vtab/dtd` - Describable Textures
- `wds/vtab/flowers` - Oxford Flowers 102
- `wds/vtab/pets` - Oxford-IIIT Pets
- `wds/vtab/svhn` - Street View House Numbers

**Specialized:**
- `wds/vtab/eurosat` - EuroSAT satellite imagery
- `wds/vtab/resisc45` - RESISC45 remote sensing
- `wds/vtab/pcam` - PatchCamelyon medical imaging
- `wds/vtab/diabetic_retinopathy` - Diabetic retinopathy detection

**3D/Spatial:**
- `wds/vtab/clevr_count_all` - CLEVR object counting
- `wds/vtab/clevr_closest_object_distance` - CLEVR distance
- `wds/vtab/dsprites_label_orientation` - dSprites orientation
- `wds/vtab/dsprites_label_x_position` - dSprites X position
- `wds/vtab/dsprites_label_y_position` - dSprites Y position
- `wds/vtab/smallnorb_label_azimuth` - SmallNORB azimuth
- `wds/vtab/smallnorb_label_elevation` - SmallNORB elevation
- `wds/vtab/kitti_closest_vehicle_distance` - KITTI distance
- `wds/vtab/dmlab` - DeepMind Lab

## Usage Examples

### 1. Test Specific Datasets

```bash
# Test MSCOCO retrieval
python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions

# Test Flowers classification
python scripts/test_clip_benchmark.py --datasets wds/vtab/flowers

# Test multiple datasets
python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions wds/vtab/flowers wds/vtab/cifar10
```

### 2. Quick Test with Subset

Test on only 100 samples for faster iteration:

```bash
python scripts/test_clip_benchmark.py \
    --datasets wds/vtab/flowers \
    --num_samples 100
```

### 3. Custom Model

Test with a different CLIP model:

```bash
# Use ViT-L/14 from LAION
python scripts/test_clip_benchmark.py \
    --datasets wds/imagenet1k \
    --model ViT-L-14 \
    --pretrained laion2b_s32b_b82k

# Use ViT-B/16 from OpenAI
python scripts/test_clip_benchmark.py \
    --datasets wds/vtab/cifar100 \
    --model ViT-B-16 \
    --pretrained openai
```

### 4. Dry Run (Loading Only)

Test that datasets load correctly without running evaluation:

```bash
python scripts/test_clip_benchmark.py \
    --datasets wds/mscoco_captions wds/vtab/flowers \
    --dry_run
```

### 5. List All Datasets

```bash
python scripts/test_clip_benchmark.py --list_datasets
```

### 6. Custom Settings

```bash
python scripts/test_clip_benchmark.py \
    --datasets wds/imagenet1k \
    --data_root /scratch/datasets \
    --batch_size 64 \
    --device cuda:0
```

## Expected Output

### MSCOCO (Retrieval)

```
Dataset: wds/mscoco_captions
  Loading: SUCCESS
  Task: zeroshot_retrieval
  Samples: 5000
  Evaluation: SUCCESS
  Metrics:
    • Image R@1: 0.3856
    • Text R@1: 0.5842
    • Image R@5: 0.6534
    • Text R@5: 0.8123
```

### Flowers (Classification)

```
Dataset: wds/vtab/flowers
  Loading: SUCCESS
  Task: zeroshot_classification
  Samples: 1020
  Evaluation: SUCCESS
  Metrics:
    • Top-1 Accuracy: 0.7451
    • Top-5 Accuracy: 0.9314
    • Mean Per-Class Recall: 0.7389
```

## Performance Tips

### Speed Up Testing

1. **Use subset sampling:**
   ```bash
   --num_samples 100
   ```

2. **Test smaller datasets first:**
   - `wds/vtab/cifar10` (10,000 samples)
   - `wds/vtab/flowers` (1,020 samples)
   - `wds/mnist` (10,000 samples)

3. **Increase batch size:**
   ```bash
   --batch_size 128
   ```

### GPU Memory

For large models or high-resolution images:

1. **Reduce batch size:**
   ```bash
   --batch_size 16
   ```

2. **Use smaller model:**
   ```bash
   --model ViT-B-32  # Instead of ViT-L-14
   ```

## Integration with Your Pipeline

### Use in Python Code

```python
from data_loading.clip_benchmark import CLIPBenchmarkDataset
import open_clip

# Load model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained='openai'
)

# Load dataset
dataset = CLIPBenchmarkDataset(
    dataset_name='wds/mscoco_captions',
    image_preprocess=preprocess,
    verbose=True
)

# Evaluate
results, embeddings = dataset.evaluate(
    embedding_model=model,
    device='cuda',
    batch_size=64
)

print(f"Image R@1: {results['image_retrieval_recall@1']:.4f}")
```

### Batch Test Multiple Datasets

```bash
# Test all VTAB datasets
for dataset in cifar10 cifar100 flowers pets dtd; do
    echo "Testing wds/vtab/$dataset..."
    python scripts/test_clip_benchmark.py \
        --datasets wds/vtab/$dataset \
        --num_samples 500
done
```

## Troubleshooting

### Dataset Download Issues

If download fails:

1. **Check internet connection**
2. **Retry with explicit download flag:**
   ```bash
   # The script uses download=True by default
   ```
3. **Clear cache and retry:**
   ```bash
   rm -rf datasets/clip_benchmark/wds/mscoco_captions
   python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions
   ```

### Memory Issues

1. **Reduce batch size:**
   ```bash
   --batch_size 8
   ```

2. **Use CPU if GPU OOM:**
   ```bash
   --device cpu
   ```

3. **Test on subset first:**
   ```bash
   --num_samples 100
   ```

### Import Errors

Install required packages:

```bash
pip install clip-benchmark
pip install open_clip_torch
pip install datasets  # For HuggingFace datasets
```

## Common Workflows

### 1. Quick Sanity Check

Test that everything works on small datasets:

```bash
python scripts/test_clip_benchmark.py \
    --datasets wds/vtab/cifar10 wds/vtab/flowers \
    --num_samples 100
```

### 2. Full Benchmark Suite

Test comprehensive set of datasets:

```bash
python scripts/test_clip_benchmark.py \
    --datasets \
        wds/mscoco_captions \
        wds/imagenet1k \
        wds/vtab/cifar10 \
        wds/vtab/cifar100 \
        wds/vtab/flowers \
        wds/vtab/pets \
        wds/vtab/dtd \
    --batch_size 64
```

### 3. Model Comparison

Compare different CLIP models:

```bash
# Test OpenAI model
python scripts/test_clip_benchmark.py \
    --datasets wds/imagenet1k \
    --model ViT-B-32 \
    --pretrained openai

# Test LAION model
python scripts/test_clip_benchmark.py \
    --datasets wds/imagenet1k \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k
```

### 4. Domain-Specific Evaluation

Test on medical imaging:

```bash
python scripts/test_clip_benchmark.py \
    --datasets \
        wds/vtab/pcam \
        wds/vtab/diabetic_retinopathy
```

Test on satellite/remote sensing:

```bash
python scripts/test_clip_benchmark.py \
    --datasets \
        wds/vtab/eurosat \
        wds/vtab/resisc45
```

## Tips for Research

1. **Always use same seed** for reproducibility
2. **Report batch size** - affects batch normalization
3. **Test on multiple runs** to estimate variance
4. **Use stratified splits** for classification tasks
5. **Report dataset version** from HuggingFace

## Output Files

The test script doesn't save outputs by default. To save results:

```python
# In your own code
results, embeddings = dataset.evaluate(...)

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save embeddings
import numpy as np
np.save('embeddings.npy', embeddings['image_embeddings'])
```
