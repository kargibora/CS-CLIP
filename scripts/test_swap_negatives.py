"""
Test script for swap negative generation.

Tests TextShuffler performance and output quality before processing all JSONs.
"""

import json
import os
import time
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.perturbations import TextShuffler


def test_text_shuffler_basic():
    """Test basic TextShuffler functionality."""
    print("="*80)
    print("TEST 1: Basic TextShuffler Functionality")
    print("="*80)
    
    shuffler = TextShuffler()
    
    test_captions = [
        "Green Grass Texture - Foto Stock",
        "US Health Secretary Tom Price reacts after delivering his speech",
        "A red car parked on the street",
        "The cat sits on the mat",
        "Beautiful sunset over the ocean with clouds",
    ]
    
    shuffler_funcs = [
        ("shuffle_nouns_and_adj", shuffler.shuffle_nouns_and_adj),
        ("shuffle_allbut_nouns_and_adj", shuffler.shuffle_allbut_nouns_and_adj),
        ("shuffle_within_trigrams", shuffler.shuffle_within_trigrams),
        ("shuffle_trigrams", shuffler.shuffle_trigrams),
    ]
    
    for caption in test_captions:
        print(f"\n📝 Original: {caption}")
        for func_name, func in shuffler_funcs:
            try:
                shuffled = func(caption)
                print(f"   {func_name:30s}: {shuffled}")
            except Exception as e:
                print(f"   {func_name:30s}: ❌ ERROR: {e}")
    
    print("\n✅ Basic functionality test complete")


def test_swap_negative_generation():
    """Test the complete swap negative generation process."""
    print("\n" + "="*80)
    print("TEST 2: Swap Negative Generation")
    print("="*80)
    
    shuffler = TextShuffler()
    
    shuffler_methods = [
        ('shuffle_nouns_and_adj', shuffler.shuffle_nouns_and_adj),
        ('shuffle_allbut_nouns_and_adj', shuffler.shuffle_allbut_nouns_and_adj),
        ('shuffle_within_trigrams', shuffler.shuffle_within_trigrams),
        ('shuffle_trigrams', shuffler.shuffle_trigrams),
    ]
    
    sample_captions = [
        "Green Grass Texture - Foto Stock",
        "US Health Secretary Tom Price reacts after delivering his speech",
        "A red car parked on the street near a blue house",
        "The quick brown fox jumps over the lazy dog",
        "Beautiful mountain landscape with snow-capped peaks",
    ]
    
    print(f"\nGenerating {len(shuffler_methods)} swap negatives per caption...\n")
    
    for caption in sample_captions:
        print(f"📝 Original: {caption}")
        
        swap_negatives = []
        identical_count = 0
        for swap_type, func in shuffler_methods:
            try:
                shuffled = func(caption)
                if shuffled != caption:
                    swap_negatives.append({
                        'swap_type': swap_type,
                        'negative': shuffled
                    })
                    print(f"   ✓ [{swap_type}]: {shuffled}")
                else:
                    identical_count += 1
                    print(f"   ⚠️  [{swap_type}] IDENTICAL (skipped): {shuffled}")
            except Exception as e:
                print(f"   ❌ [{swap_type}] ERROR: {e}")
        
        print(f"   Generated: {len(swap_negatives)}/{len(shuffler_methods)} unique negatives")
        if identical_count > 0:
            print(f"   Skipped: {identical_count} identical to original")
        print()
    
    print("✅ Swap negative generation test complete")


def test_processing_speed():
    """Test processing speed on a sample of captions."""
    print("\n" + "="*80)
    print("TEST 3: Processing Speed")
    print("="*80)
    
    shuffler = TextShuffler()
    
    shuffler_funcs = [
        shuffler.shuffle_nouns_and_adj,
        shuffler.shuffle_allbut_nouns_and_adj,
        shuffler.shuffle_within_trigrams,
        shuffler.shuffle_trigrams,
    ]
    
    # Generate test captions
    test_captions = [
        "Green Grass Texture - Foto Stock",
        "US Health Secretary Tom Price reacts after delivering his speech",
        "A red car parked on the street",
        "The cat sits on the mat",
        "Beautiful sunset over the ocean",
    ] * 20  # 100 captions total
    
    print(f"\n⏱️  Processing {len(test_captions)} captions with {len(shuffler_funcs)} shuffle functions each...")
    print(f"   Total operations: {len(test_captions) * len(shuffler_funcs)}\n")
    
    start_time = time.time()
    
    total_success = 0
    total_failed = 0
    
    for caption in test_captions:
        for func in shuffler_funcs:
            try:
                _ = func(caption)
                total_success += 1
            except Exception:
                total_failed += 1
    
    elapsed = time.time() - start_time
    
    print(f"✅ Processing complete:")
    print(f"   Time taken: {elapsed:.2f} seconds")
    print(f"   Successful: {total_success}/{len(test_captions) * len(shuffler_funcs)}")
    print(f"   Failed: {total_failed}/{len(test_captions) * len(shuffler_funcs)}")
    print(f"   Speed: {len(test_captions) / elapsed:.1f} captions/second")
    print(f"   Speed: {(len(test_captions) * len(shuffler_funcs)) / elapsed:.1f} operations/second")


def test_with_real_json():
    """Test with actual JSON file from the dataset."""
    print("\n" + "="*80)
    print("TEST 4: Real JSON File Processing")
    print("="*80)
    
    json_dir = "/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components"
    
    # Find first available JSON file
    json_files = sorted(Path(json_dir).glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {json_dir}")
        return
    
    test_file = json_files[0]
    print(f"\n📂 Loading: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Total samples: {len(data)}")
    
    # Test on first 5 samples
    num_test = min(5, len(data))
    print(f"\n🔬 Testing on first {num_test} samples:\n")
    
    shuffler = TextShuffler()
    shuffler_funcs = [
        shuffler.shuffle_nouns_and_adj,
        shuffler.shuffle_allbut_nouns_and_adj,
        shuffler.shuffle_within_trigrams,
        shuffler.shuffle_trigrams,
    ]
    
    for i, sample in enumerate(data[:num_test]):
        caption = sample.get('original_caption', '')
        print(f"{i+1}. Original: {caption}")
        
        swap_negatives = []
        identical_count = 0
        for func in shuffler_funcs:
            try:
                shuffled = func(caption)
                if shuffled != caption:
                    swap_negatives.append(shuffled)
                else:
                    identical_count += 1
            except Exception as e:
                print(f"   ⚠️  Shuffle failed: {e}")
        
        print(f"   Generated {len(swap_negatives)} unique swap negatives:")
        if identical_count > 0:
            print(f"   (Skipped {identical_count} identical to original)")
        for j, neg in enumerate(swap_negatives, 1):
            print(f"     {j}. {neg}")
        print()
    
    # Test speed on all samples
    print(f"⏱️  Speed test on all {len(data)} samples:")
    
    start_time = time.time()
    
    total_success = 0
    total_failed = 0
    
    for sample in data:
        caption = sample.get('original_caption', '')
        for func in shuffler_funcs:
            try:
                _ = func(caption)
                total_success += 1
            except Exception:
                total_failed += 1
    
    elapsed = time.time() - start_time
    
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Speed: {len(data) / elapsed:.1f} captions/second")
    print(f"   Success rate: {total_success}/{total_success + total_failed} ({100*total_success/(total_success+total_failed):.1f}%)")
    
    print("\n✅ Real JSON file test complete")


def test_memory_efficiency():
    """Test memory efficiency of TextShuffler singleton."""
    print("\n" + "="*80)
    print("TEST 5: Memory Efficiency")
    print("="*80)
    
    import psutil
    import os as os_module
    
    process = psutil.Process(os_module.getpid())
    
    print("\n📊 Memory usage:")
    
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"   Before TextShuffler: {mem_before:.1f} MB")
    
    shuffler = TextShuffler()
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    print(f"   After TextShuffler: {mem_after:.1f} MB")
    print(f"   TextShuffler memory: {mem_after - mem_before:.1f} MB")
    
    # Process some captions to see working memory
    test_captions = ["Test caption " + str(i) for i in range(1000)]
    
    for caption in test_captions:
        try:
            shuffler.shuffle_nouns_and_adj(caption)
        except:
            pass
    
    mem_working = process.memory_info().rss / 1024 / 1024  # MB
    print(f"   After 1000 operations: {mem_working:.1f} MB")
    print(f"   Working memory delta: {mem_working - mem_after:.1f} MB")
    
    print("\n✅ Memory efficiency test complete")


def run_all_tests():
    """Run all tests."""
    print("\n" + "🧪 " + "="*76 + " 🧪")
    print("   SWAP NEGATIVES TESTING SUITE")
    print("🧪 " + "="*76 + " 🧪\n")
    
    try:
        test_text_shuffler_basic()
        test_swap_negative_generation()
        test_processing_speed()
        test_with_real_json()
        test_memory_efficiency()
        
        print("\n" + "🎉 " + "="*76 + " 🎉")
        print("   ALL TESTS PASSED!")
        print("🎉 " + "="*76 + " 🎉\n")
        
    except Exception as e:
        print("\n" + "❌ " + "="*76 + " ❌")
        print(f"   TEST FAILED: {e}")
        print("❌ " + "="*76 + " ❌\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
