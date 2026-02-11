# Swap Negatives Format Update

## 🎯 What Changed

The `swap_negatives` field in JSON files now includes **swap type information** to track which shuffling method was used for each negative.

## 📝 New Format

### Before (Old Format)
```json
{
    "original_caption": "A cat sitting on a mat",
    "swap_negatives": [
        "A mat sitting on a cat",
        "sitting A cat on mat a",
        "sitting A cat mat a on",
        "on a mat A cat sitting"
    ]
}
```

### After (New Format)
```json
{
    "original_caption": "A cat sitting on a mat",
    "swap_negatives": [
        {
            "swap_type": "shuffle_nouns_and_adj",
            "negative": "A mat sitting on a cat"
        },
        {
            "swap_type": "shuffle_allbut_nouns_and_adj",
            "negative": "sitting A cat on mat a"
        },
        {
            "swap_type": "shuffle_within_trigrams",
            "negative": "sitting A cat mat a on"
        },
        {
            "swap_type": "shuffle_trigrams",
            "negative": "on a mat A cat sitting"
        }
    ]
}
```

## 🔍 Swap Types

The four swap types correspond to TextShuffler methods:

1. **`shuffle_nouns_and_adj`**
   - Shuffles nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS)
   - Example: "A red cat on a mat" → "A mat cat on a red"

2. **`shuffle_allbut_nouns_and_adj`**
   - Shuffles everything EXCEPT nouns and adjectives
   - Example: "A red cat on a mat" → "on A red cat a mat"

3. **`shuffle_within_trigrams`**
   - Groups words into trigrams (3-word chunks) and shuffles within each trigram
   - Example: "The quick brown fox jumps over" → "brown The quick jumps fox over"

4. **`shuffle_trigrams`**
   - Groups words into trigrams and shuffles the trigrams themselves
   - Example: "The quick brown fox jumps over" → "fox jumps over The quick brown"

## 💡 Benefits

### 1. Analysis by Swap Type
```python
# Analyze which swap types are most effective
from collections import Counter

swap_type_counts = Counter()
for sample in dataset:
    for neg in sample['swap_negatives']:
        swap_type_counts[neg['swap_type']] += 1

print("Swap type distribution:", swap_type_counts)
# Output: {'shuffle_nouns_and_adj': 2500, 'shuffle_trigrams': 2300, ...}
```

### 2. Selective Sampling
```python
# Use only specific types during training
def get_negative(sample, preferred_types=['shuffle_nouns_and_adj']):
    swap_negatives = sample.get('swap_negatives', [])
    
    # Filter by preferred types
    filtered = [
        neg for neg in swap_negatives 
        if neg['swap_type'] in preferred_types
    ]
    
    if filtered:
        return random.choice(filtered)['negative']
    elif swap_negatives:
        return random.choice(swap_negatives)['negative']
    else:
        return sample['original_caption']
```

### 3. Curriculum Learning
```python
# Start with simple swaps, gradually use harder ones
def get_negative_curriculum(sample, epoch):
    swap_negatives = sample.get('swap_negatives', [])
    
    if epoch < 5:
        # Early training: only noun/adj swaps (easier)
        types = ['shuffle_nouns_and_adj']
    elif epoch < 10:
        # Mid training: add trigram shuffling
        types = ['shuffle_nouns_and_adj', 'shuffle_within_trigrams']
    else:
        # Late training: all types
        types = None  # Use all
    
    if types:
        filtered = [neg for neg in swap_negatives if neg['swap_type'] in types]
        if filtered:
            return random.choice(filtered)['negative']
    
    # Fallback: any negative
    if swap_negatives:
        return random.choice(swap_negatives)['negative']
    return sample['original_caption']
```

### 4. Debugging and Analysis
```python
# Track which swap types cause model failures
failures_by_type = defaultdict(int)

for sample in test_set:
    prediction = model.predict(sample['image'])
    
    for neg in sample['swap_negatives']:
        if model.confuses(sample['original_caption'], neg['negative']):
            failures_by_type[neg['swap_type']] += 1

print("Most confusing swap types:", failures_by_type)
# Helps identify which linguistic perturbations are hardest for the model
```

## 🔄 Migration Guide

### If You Had Old Format Data

The old format is still compatible for reading, but you won't have swap type info:

```python
# Backward compatible code
def get_negative_safe(sample):
    swap_negatives = sample.get('swap_negatives', [])
    
    if not swap_negatives:
        return sample['original_caption']
    
    # Check format
    first_neg = swap_negatives[0]
    
    if isinstance(first_neg, dict):
        # New format: has swap_type
        return random.choice(swap_negatives)['negative']
    else:
        # Old format: just strings
        return random.choice(swap_negatives)
```

### Regenerate With New Format

To update existing files:

```bash
# Process again with the new script
python scripts/add_swap_negatives.py \
    --input_dir /path/to/old/files \
    --output_dir /path/to/new/files \
    --num_processes 8
```

## 📊 Example Usage in Training

```python
class MyDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = json.load(f)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get positive caption
        positive = sample['original_caption']
        
        # Get negative with type info
        swap_negatives = sample.get('swap_negatives', [])
        if swap_negatives:
            neg_obj = random.choice(swap_negatives)
            negative = neg_obj['negative']
            swap_type = neg_obj['swap_type']
            
            # Optional: weight loss by swap type difficulty
            if swap_type == 'shuffle_nouns_and_adj':
                weight = 1.0  # Easier, normal weight
            elif swap_type == 'shuffle_trigrams':
                weight = 1.5  # Harder, higher weight
            else:
                weight = 1.2
        else:
            negative = positive
            weight = 1.0
        
        return {
            'image': self.load_image(sample),
            'positive': positive,
            'negative': negative,
            'weight': weight,
            'swap_type': swap_type if swap_negatives else None
        }
```

## ✅ Testing

Run the test suite to verify the new format:

```bash
python scripts/test_swap_negatives.py
```

Expected output will now show swap types:
```
📝 Original: Green Grass Texture - Foto Stock
   ✓ [shuffle_nouns_and_adj]: Texture Green Grass - Stock Foto
   ✓ [shuffle_allbut_nouns_and_adj]: Foto Stock - Green Grass Texture
   ✓ [shuffle_within_trigrams]: Green Texture Grass - Foto Stock
   ✓ [shuffle_trigrams]: Stock Foto - Texture Grass Green
   Generated: 4/4 unique negatives
```

## 📈 Benefits Summary

✅ **Track which methods work best** - Analyze effectiveness by swap type  
✅ **Selective training** - Use only certain swap types based on training stage  
✅ **Curriculum learning** - Start easy (noun swaps), progress to hard (trigrams)  
✅ **Better debugging** - Identify which perturbations confuse your model  
✅ **Research insights** - Study which linguistic changes matter most  
✅ **Backward compatible** - Old code still works with simple updates  

## 🚀 Processing

Generate swap negatives with type info:

```bash
# Same command as before, but output includes swap types
python scripts/add_swap_negatives.py \
    --num_processes 8 \
    --batch_size 1000
```

The format change is automatic - no special flags needed!
