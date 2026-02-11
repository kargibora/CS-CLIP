"""
Simplified dataset evaluation utilities for training pipeline.
Uses the existing dataset loading system with default paths.
"""

import os
import csv
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import argparse
import signal
import logging
import traceback
import wandb
import numpy as np
import clip

# Import the existing dataset loading system
from data_loading import build_dataset_from_args

# Import the benchmark visualization logger
from utils.benchmark_visualization_logger import BenchmarkVisualizationLogger


class DatasetEvaluator:
    """Evaluates multiple benchmark datasets during training using existing dataset loading system."""
    
    def __init__(
        self, 
        datasets=None, 
        csv_path="dataset_evaluation_results.csv",
        enable_visualization=True,
        num_viz_samples=5,
        device=None,
        is_ft=False
    ):
        """
        Initialize the DatasetEvaluator.
        
        Args:
            datasets (list): List of dataset names to evaluate (e.g., ['VALSE', 'BLA', 'ColorFoil', 'CLIPBench_wds_imagenet1k'])
            csv_path (str): Path to save CSV results
            enable_visualization (bool): Whether to enable benchmark visualization logging
            num_viz_samples (int): Number of samples to visualize per dataset
            device: Device for visualization computation (if None, will be inferred from model)
            is_ft (bool): Whether using fine-tuning mode
        """
        # Parse CLIPBench datasets: group them together
        self.datasets = self._parse_datasets(datasets or [])
        self.csv_path = csv_path
        self.csv_lock = threading.Lock()  # For thread-safe CSV writing
        self.enable_visualization = enable_visualization
        self.device = device
        self.is_ft = is_ft
        
        # Initialize visualization logger if enabled
        self.viz_logger = None
        if enable_visualization:
            # Device will be set later when we have access to the model
            self.viz_logger = BenchmarkVisualizationLogger(
                device=device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                num_samples_per_dataset=num_viz_samples,
                is_ft=is_ft
            )
            print(f"✅ Visualization logger initialized: {num_viz_samples} samples per dataset")
        else:
            print("ℹ️ Visualization logging disabled (enable_visualization=False)")
        
        # Define dataset subsets based on eval_datasets.sh
        self.dataset_subsets = {
            'VisMin': ['all'],
            'SPEC_I2T': ['count', 'relative_spatial', 'relative_size', 'absolute_size', 'absolute_spatial', 'existence'],
            'ControlledImages': ['A', 'B', 'COCO-One', 'COCO-Two', 'VG-One', 'VG-Two'],
            'COCO_Order': ['all'],
            'Flickr30k_Order': ['all'],
            'VG_Relation': ['all'],
            'VG_Attribution': ['all'],
            'SugarCrepe': ['add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj'],
            'SugarCrepe_PP': ['swap_object', 'swap_atribute', 'replace_object', 'replace_attribute', 'replace_relation'],
            'CC3M': ['all'],
            'Winoground': ['all'],
            'BLA': ['ap', 'co', 'rc'],
            "VALSE": ["existence", "plurals", "counting", "relations", "actions", "coreference", "noun phrases"],
            'VL_CheckList': [
                # Standard VL-CheckList evaluation grouping (as used in papers)
                # Each test aggregates ALL datasets and subtypes for that category
                
                # Attribute tests (5 types)
                'attr_color',      # All color: vaw + vg
                'attr_material',   # All material: vaw + vg
                'attr_size',       # All size: vaw + vg
                'attr_state',      # All state: vaw + vg
                'attr_action',     # All action: vaw + vg
                
                # Object tests (2 types)
                'obj_location',    # All location: center+margin+mid × hake+4×swig+2×vg = 21 datasets
                'obj_size',        # All size: large+medium+small × hake+4×swig+2×vg = 21 datasets
                
                # Relation tests (2 types)
                'rel_action',      # All action: hake + swig + vg
                'rel_spatial',     # All spatial: vg
            ],
            'ColorSwap': ['all'],
            'ColorFoil': ['all'],
            'COCO_Counterfactuals': ['all'],
            'COLA': ['multi_objects'],
            'NegBench': [
                'COCO_val_mcq_llama3.1_rephrased', 
                'COCO_val_retrieval', 
                'COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true',
                'msr_vtt_retrieval',
                'msr_vtt_retrieval_rephrased_llama',
                'msr_vtt_mcq_rephrased_llama',
                "VOC2007_mcq_llama3.1_rephrased"
            ],
            # SVO Probes dataset - tests verb understanding with subject/verb/object negatives
            'SVOProbes': ['subj_neg', 'verb_neg', 'obj_neg'],
            # MMVP-VLM Benchmark - tests visual pattern understanding
            # Pattern types: Camera Perspective, Color, Orientation, Presence, Quantity, Shape, Size, State, Texture
            'MMVP': [
                "Camera Perspective",
                "Color",
                "Orientation", 
                "Presence",
                "Quantity",
                "Spatial",
                "State",
                "Structural Character",
                "Text",
            ]
        }
        
        # Add CLIPBench datasets dynamically if parsed
        if hasattr(self, '_clipbench_subsets') and self._clipbench_subsets:
            # Add CLIPBench with the parsed subsets
            self.dataset_subsets['CLIPBench'] = self._clipbench_subsets
        
        # Initialize CSV file with headers
        self._initialize_csv()
    
    def _parse_datasets(self, datasets: List[str]) -> List[str]:
        """
        Parse dataset list and group CLIPBench datasets together.
        
        Example:
            Input: ['NegBench', 'CLIPBench_wds_imagenet1k', 'CLIPBench_wds_cifar10', 'VALSE']
            Output: ['NegBench', 'CLIPBench', 'VALSE']
            
        Also populates self.dataset_subsets with CLIPBench subsets.
        
        Args:
            datasets: List of dataset names
            
        Returns:
            Parsed list with CLIPBench datasets grouped
        """
        parsed_datasets = []
        clipbench_subsets = []
        
        for dataset in datasets:
            if dataset.startswith('CLIPBench_'):
                # Extract the actual CLIP Benchmark dataset name (e.g., 'wds_imagenet1k')
                clip_dataset_name = dataset[len('CLIPBench_'):]
                clipbench_subsets.append(clip_dataset_name)
            else:
                # Regular dataset
                parsed_datasets.append(dataset)
        
        # Add CLIPBench as a single dataset if we found any CLIPBench datasets
        if clipbench_subsets:
            parsed_datasets.append('CLIPBench')
            # Store the subsets for later use
            self._clipbench_subsets = clipbench_subsets
            print(f"📦 Grouped {len(clipbench_subsets)} CLIP Benchmark datasets:")
            for subset in clipbench_subsets:
                print(f"   • {subset}")
        else:
            self._clipbench_subsets = []
        
        return parsed_datasets
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with self.csv_lock:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'step', 'epoch', 'dataset', 'subset', 
                        'metric', 'value', 'total_samples'
                    ])
    
    def _create_args_for_dataset(self, dataset_name: str, subset_name: str) -> argparse.Namespace:
        """Create args object for dataset loading using existing system."""
        args = argparse.Namespace()
        
        # Check if this is a CLIP Benchmark dataset
        if dataset_name == 'CLIPBench':
            # New format: dataset_name='CLIPBench', subset_name='wds_imagenet1k'
            # subset_name already contains the full HuggingFace dataset name (e.g., 'wds_imagenet1k')
            args.dataset = 'CLIPBenchmark'
            args.clip_benchmark_name = subset_name  # Pass the full name including wds_ prefix
            args.subset_name = 'all'  # CLIPBench datasets don't have subsets
        elif dataset_name.startswith('CLIPBench_'):
            # Old format (for backward compatibility): 'CLIPBench_wds_imagenet1k'
            # Extract the actual dataset name (remove CLIPBench_ prefix)
            actual_dataset_name = dataset_name[len('CLIPBench_'):]
            args.dataset = 'CLIPBenchmark'
            args.clip_benchmark_name = actual_dataset_name
            args.subset_name = subset_name
        else:
            # Regular dataset
            args.dataset = dataset_name
            args.subset_name = subset_name
        
        args.data_path = None  # Will use defaults from build_dataset_from_args
        return args
    
    def _is_labclip_model(self, model):
        """Check if the model is a LabCLIP model that computes similarities directly."""
        if hasattr(model, 'module'):  # Handle DDP wrapped models
            model = model.module
        
        # Check for standalone LabCLIP model
        if model.__class__.__name__ == "LabCLIPFlexibleAlignment":
            return True
        
        # Check for LabCLIP FT model (CLIPMultiLayerFTAlignment with use_labclip=True)
        if (model.__class__.__name__ == "CLIPMultiLayerFTAlignment" and 
            hasattr(model, 'use_labclip') and 
            getattr(model, 'use_labclip', False)):
            return True
        
        return False
    
    def _process_labclip_batch(self, batch, model, clip_model, device):
        """Process a batch of samples for LabCLIP evaluation."""
        import clip
        import logging
        from utils.align import extract_intermediate_features
        
        images = []
        texts = []
        labels = []
        
        logging.debug(f"🔄 Processing batch of {len(batch)} samples")
        
        # Extract data from different formats
        for i, sample in enumerate(batch):
            try:
                img = None
                txt = None
                lbl = 1  # Default to positive
                
                if isinstance(sample, dict):
                    # SugarCrepe format
                    if 'image_options' in sample and 'caption_options' in sample:
                        img = sample['image_options']
                        # For SugarCrepe, we'll use the positive caption (index 0)
                        caption_options = sample['caption_options']
                        if isinstance(caption_options, list) and len(caption_options) > 0:
                            txt = caption_options[0]  # Use positive caption
                        lbl = sample.get('label', 1)
                    # Standard dictionary format
                    elif 'image' in sample or 'img' in sample:
                        img = sample.get('image') or sample.get('img')
                        txt = sample.get('caption') or sample.get('text') or sample.get('txt')
                        lbl = sample.get('label', 1)
                    else:
                        logging.warning(f"⚠️ Unknown dict sample format at index {i}: keys={list(sample.keys())}")
                        continue
                        
                elif isinstance(sample, (tuple, list)):
                    # Tuple format: (image, text, label) or (image, text)
                    img = sample[0]
                    txt = sample[1]
                    lbl = sample[2] if len(sample) > 2 else 1
                else:
                    logging.warning(f"⚠️ Unknown sample format at index {i}: {type(sample)}")
                    continue
                
                if img is not None and txt is not None:
                    images.append(img)
                    texts.append(txt)
                    labels.append(lbl)
                else:
                    logging.warning(f"⚠️ Sample {i} missing image or text: img={img is not None}, txt={txt is not None}")
                    if isinstance(sample, dict):
                        logging.debug(f"   Sample keys: {list(sample.keys())}")
                    
            except Exception as e:
                logging.error(f"❌ Error processing sample {i}: {e}")
                continue
        
        if not images or not texts:
            logging.warning(f"⚠️ No valid samples in batch: {len(images)} images, {len(texts)} texts")
            return None
        
        logging.debug(f"✅ Extracted {len(images)} valid samples from batch")
        
        try:
            # Prepare images - handle both PIL and tensor inputs
            if hasattr(images[0], 'mode'):  # PIL Image
                # Apply preprocessing if needed
                if hasattr(self, 'preprocess') and self.preprocess is not None:
                    images_tensor = torch.stack([self.preprocess(img) for img in images]).to(device)
                else:
                    # Convert PIL to tensor manually
                    import torchvision.transforms as transforms
                    to_tensor = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    images_tensor = torch.stack([to_tensor(img) for img in images]).to(device)
                logging.debug(f"📷 Preprocessed PIL images to tensor: {images_tensor.shape}")
            else:
                # Already tensors
                images_tensor = torch.stack(images).to(device)
                logging.debug(f"📷 Stacked tensor images: {images_tensor.shape}")
                
            # Tokenize texts with proper error handling
            texts_tensor = clip.tokenize(texts, truncate=True).to(device)
            logging.debug(f"📝 Tokenized texts: {texts_tensor.shape}")
            
            # Extract features
            with torch.no_grad():
                # Get CLIP features
                image_features = extract_intermediate_features(
                    images_tensor, clip_model, device=device,
                    layer_names=['final'], is_image=True, dtype=clip_model.dtype
                )
                logging.debug(f"🖼️ Extracted image features: {image_features['final'].shape}")
                
                text_features = extract_intermediate_features(
                    texts_tensor, clip_model, device=device,
                    layer_names=['final'], is_image=False, dtype=clip_model.dtype
                )
                logging.debug(f"📝 Extracted text features: {text_features['final'].shape}")
                
                # Get LabCLIP scores
                scores = model(image_features, text_features)  # Shape: (batch_size,) or (batch_size, 1)
                logging.debug(f"🎯 Model output scores shape: {scores.shape}")
                
                if scores.dim() > 1:
                    scores = scores.squeeze(-1)  # Remove last dimension if present
                    logging.debug(f"🎯 Squeezed scores shape: {scores.shape}")
                
                # Convert scores to predictions (threshold at 0)
                predictions = (scores > 0).long().cpu().tolist()
                scores_list = scores.cpu().tolist()
                
                logging.debug(f"✅ Batch processed: {len(predictions)} predictions, score range [{min(scores_list):.3f}, {max(scores_list):.3f}]")
                
                return {
                    'predictions': predictions,
                    'labels': labels,
                    'scores': scores_list
                }
                
        except Exception as e:
            logging.error(f"❌ Error in batch processing: {e}")
            import traceback
            logging.error(f"📍 Traceback:\n{traceback.format_exc()}")
            return None
    
    def _evaluate_with_labclip(self, model, clip_model, preprocess, dataset, device):
        """
        Evaluate LabCLIP model that computes similarities directly.
        This bypasses the dataset's .evaluate() method and implements custom logic.
        """
        import logging
        
        logging.info("🚀 Starting LabCLIP evaluation")
        
        # Try to use dataset's built-in evaluation if it supports LabCLIP
        if hasattr(dataset, 'evaluate') and hasattr(dataset, 'supports_labclip_evaluation'):
            try:
                logging.info("📊 Using dataset's built-in LabCLIP evaluation")
                results, _ = dataset.evaluate(
                    embedding_model=clip_model,
                    aligning_model=model,
                    labclip_mode=True
                )
                logging.info(f"✅ Built-in evaluation succeeded: {list(results.keys())}")
                return results
            except Exception as e:
                logging.warning(f"⚠️ Built-in evaluation failed, falling back to standard: {e}")
        
        # Check if dataset has standard evaluate method that might work
        if hasattr(dataset, 'evaluate'):
            try:
                logging.info("📊 Trying dataset's standard evaluate method with LabCLIP model")
                
                # Check if this is SugarCrepe dataset
                dataset_name = getattr(dataset, '__class__', type(dataset)).__name__
                logging.info(f"📋 Dataset type: {dataset_name}")

                # The dataset expects intermediate layer names for feature extraction
                intermediate_image_layer_names = ['final']  # Use final layer
                intermediate_text_layer_names = ['final']   # Use final layer
                
                logging.info(f"🔧 Calling dataset.evaluate with parameters:")
                logging.info(f"   embedding_model: {type(clip_model).__name__}")
                logging.info(f"   aligning_model: {type(model).__name__}")
                logging.info(f"   device: {device}")
                logging.info(f"   intermediate_image_layer_names: {intermediate_image_layer_names}")
                logging.info(f"   intermediate_text_layer_names: {intermediate_text_layer_names}")
                
                results, embeddings = dataset.evaluate(
                    embedding_model=clip_model,
                    aligning_model=model,  # Pass LabCLIP model as aligning model
                    intermediate_image_layer_names=intermediate_image_layer_names,
                    intermediate_text_layer_names=intermediate_text_layer_names,
                    device=device
                )
                
                logging.info(f"✅ Dataset evaluation succeeded!")
                logging.info(f"   Results keys: {list(results.keys()) if results else 'None'}")
                logging.info(f"   Embeddings keys: {list(embeddings.keys()) if embeddings else 'None'}")
                
                if results and len(results) > 0:
                    logging.info(f"✅ Standard evaluation with LabCLIP model succeeded: {list(results.keys())}")
                    
                    # Make sure to include text_contrastive_accuracy
                    if 'text_contrastive_accuracy' not in results and 'accuracy' in results:
                        results['text_contrastive_accuracy'] = results['accuracy']
                    
                    # Log final results
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            logging.info(f"   {key}: {value:.4f}")
                    
                    return results
                else:
                    logging.warning("⚠️ Standard evaluation returned empty results")
                    
            except Exception as e:
                logging.error(f"❌ Standard evaluation failed with error: {e}")
                import traceback
                logging.error(f"📍 Full traceback:\n{traceback.format_exc()}")
        else:
            logging.warning("⚠️ Dataset does not have an evaluate method")
        
        # Custom LabCLIP evaluation implementation (only as last resort)
        logging.warning("� Falling back to custom LabCLIP evaluation - this should not happen for SugarCrepe!")
        
        # For SugarCrepe, we should never reach here - force an error to debug
        dataset_name = getattr(dataset, '__class__', type(dataset)).__name__
        if 'SugarCrepe' in dataset_name:
            logging.error("❌ SugarCrepe dataset should use built-in evaluation, not custom fallback!")
            return {'accuracy': 0.0, 'total_samples': 0, 'num_samples': 0, 'error': 'sugarcrepe_fallback_error'}
        
        logging.info("🔧 Using custom LabCLIP evaluation implementation")
        
        if hasattr(dataset, 'get_evaluation_samples'):
            # Some datasets have special evaluation sample extraction
            evaluation_data = dataset.get_evaluation_samples()
            total_samples = len(evaluation_data)
            logging.info(f"📋 Using dataset's evaluation samples: {total_samples} samples")
        else:
            # For efficiency, limit evaluation to a reasonable subset
            total_samples = min(len(dataset), 500)  # Reduce to 500 samples max for speed
            evaluation_data = None  # We'll load on demand
            logging.info(f"📋 Using custom sampling: {total_samples} samples (limited from {len(dataset)})")
        
        # Process samples in batches
        batch_size = 8  # Even smaller batch size for memory efficiency
        all_predictions = []
        all_labels = []
        all_scores = []
        
        # Create progress bar
        num_batches = (total_samples + batch_size - 1) // batch_size
        logging.info(f"🔄 Processing {num_batches} batches of size {batch_size}")
        
        successful_batches = 0
        failed_batches = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            
            try:
                # Load batch on demand
                if evaluation_data is not None:
                    batch = evaluation_data[start_idx:end_idx]
                else:
                    batch = []
                    for i in range(start_idx, end_idx):
                        try:
                            sample = dataset[i]
                            batch.append(sample)
                        except Exception as e:
                            logging.error(f"❌ Error getting sample {i}: {e}")
                            continue
                
                if not batch:
                    logging.warning(f"⚠️ Batch {batch_idx} is empty, skipping")
                    continue
                
                # Extract batch data
                batch_results = self._process_labclip_batch(batch, model, clip_model, device)
                
                if batch_results:
                    all_predictions.extend(batch_results['predictions'])
                    all_labels.extend(batch_results['labels'])
                    all_scores.extend(batch_results['scores'])
                    successful_batches += 1
                    
                    if batch_idx % 10 == 0:  # Log progress every 10 batches
                        logging.info(f"📊 Batch {batch_idx}/{num_batches}: {len(batch_results['predictions'])} samples processed")
                else:
                    failed_batches += 1
                    logging.warning(f"⚠️ Batch {batch_idx} processing failed")
                    
            except Exception as e:
                failed_batches += 1
                logging.error(f"❌ Error processing batch {batch_idx}: {e}")
                continue
        
        logging.info(f"📈 Processing complete: {successful_batches} successful, {failed_batches} failed batches")
        
        # Calculate metrics
        if not all_predictions:
            logging.error("❌ No predictions generated - evaluation failed completely")
            return {'accuracy': 0.0, 'total_samples': 0, 'num_samples': 0, 'error': 'no_predictions'}
        
        logging.info(f"🔢 Computing metrics from {len(all_predictions)} predictions")
        logging.info(f"📊 Score range: min={min(all_scores):.3f}, max={max(all_scores):.3f}, mean={sum(all_scores)/len(all_scores):.3f}")
        
        # Calculate label and prediction distributions
        import numpy as np
        try:
            label_unique, label_counts = np.unique(all_labels, return_counts=True)
            pred_unique, pred_counts = np.unique(all_predictions, return_counts=True)
            logging.info(f"🏷️ Label distribution: {dict(zip(label_unique, label_counts))}")
            logging.info(f"🎯 Prediction distribution: {dict(zip(pred_unique, pred_counts))}")
        except Exception as e:
            logging.warning(f"⚠️ Could not compute distributions: {e}")
        
        # Calculate accuracy
        correct = sum(1 for pred, label in zip(all_predictions, all_labels) if pred == label)
        accuracy = correct / len(all_predictions)
        
        # Additional metrics
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        score_std = torch.tensor(all_scores).std().item() if all_scores else 0.0
        
        results = {
            'accuracy': accuracy,
            'avg_score': avg_score,
            'score_std': score_std,
            'total_samples': len(all_predictions),
            'num_samples': len(all_predictions),
            'text_contrastive_accuracy': accuracy,  # Add this for consistency with expected metrics
        }
        
        logging.info("✅ Final metrics computed:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logging.info(f"   {key}: {value:.4f}")
        
        # Add dataset-specific metrics if available
        if hasattr(dataset, 'compute_additional_metrics'):
            try:
                logging.info("🔧 Computing additional dataset-specific metrics")
                additional_metrics = dataset.compute_additional_metrics(all_predictions, all_labels, all_scores)
                results.update(additional_metrics)
                logging.info(f"✅ Added {len(additional_metrics)} additional metrics")
            except Exception as e:
                logging.error(f"❌ Error computing additional metrics: {e}")
        
        return results
    
    
    def _evaluate_with_standard_clip(self, model, clip_model, preprocess, dataset, alignment_type, is_initial_eval, device):
        """
        Evaluate using standard CLIP approach with separate embeddings.
        This uses the dataset's original .evaluate() method.
        """
        # Determine embedding_model and aligning_model based on context
        if is_initial_eval:
            # Initial evaluation: Only CLIP model, no alignment model for all types
            embedding_model = clip_model
            aligning_model = None
        elif alignment_type in ["HNB", "SB"]:
            # Standard CLIP training: CLIP as embedding model, trained model as aligning model
            embedding_model = clip_model
            aligning_model = model
        elif alignment_type == "FT":
            # Fine-tuning: trained model is embedding model, no aligning model
            embedding_model = model
            aligning_model = None
        else:
            # Default to standard CLIP setup
            embedding_model = clip_model
            aligning_model = model
        
        # Call the dataset's evaluate method directly
        results, _ = dataset.evaluate(
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
            batch_size=64,
            indices=None,
            intermediate_text_layer_names=['final'],
            intermediate_image_layer_names=['final']
        )
        
        return results

    def _is_tqa_model(self, model) -> bool:
        """Check if model is a TextQueryAggregator pipeline."""
        base_model = model.module if hasattr(model, 'module') else model
        # Check for encode_multimodal method (TQA feature)
        if hasattr(base_model, 'encode_multimodal'):
            return True
        # Check class name
        class_name = base_model.__class__.__name__
        return 'TextQueryAggregator' in class_name
    
    # List of compositional benchmarks that benefit from TQA caption-conditioned evaluation
    COMPOSITIONAL_BENCHMARKS = {
        'SugarCrepe', 'SugarCrepe_PP', 'Winoground', 'VALSE', 'BLA', 
        'ColorSwap', 'ColorFoil', 'VG_Attribution', 'VG_Relation',
        'COCO_Counterfactuals', 'ControlledImages', 'NegBench'
    }
    
    def _evaluate_compositional_with_tqa(self, model, clip_model, preprocess, dataset, device) -> Optional[Dict[str, float]]:
        """
        Evaluate TQA model on compositional benchmarks using caption-conditioned embeddings.
        
        For each (image, caption_pos, caption_neg) triplet:
        1. Compute image embedding conditioned on caption_pos
        2. Compute image embedding conditioned on caption_neg
        3. Compare which has higher similarity to its corresponding text
        
        This leverages TQA's cross-attention: v_cls' = v_cls + Attn(q=text, k=patches, v=patches)
        
        Returns:
            Dict with accuracy metrics, or None if evaluation fails
        """
        from tqdm import tqdm
        
        logging.info("🔍 Using TQA caption-conditioned evaluation for compositional benchmark")
        
        # Get base model (handle DDP wrapper)
        base_model = model.module if hasattr(model, 'module') else model
        
        # Check if model supports encode_multimodal
        if not hasattr(base_model, 'encode_multimodal'):
            logging.warning("Model doesn't support encode_multimodal, falling back to standard evaluation")
            return None
        
        correct = 0
        total = 0
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="TQA Compositional Eval", disable=len(dataset) > 500):
                try:
                    sample = dataset[i]
                    
                    # Handle different dataset formats
                    if isinstance(sample, dict):
                        # Get image
                        image = sample.get('image') or sample.get('img') or sample.get('image_tensor')
                        
                        # SugarCrepe format: caption_options = [positive, negative]
                        if 'caption_options' in sample:
                            captions = sample['caption_options']
                            if len(captions) >= 2:
                                caption_pos = captions[0]
                                caption_neg = captions[1]
                            else:
                                continue
                        # Winoground/BLA format
                        elif 'caption' in sample and 'foil' in sample:
                            caption_pos = sample['caption']
                            caption_neg = sample['foil']
                        # Standard contrastive format
                        elif 'positive_caption' in sample and 'negative_caption' in sample:
                            caption_pos = sample['positive_caption']
                            caption_neg = sample['negative_caption']
                        elif 'caption' in sample and 'negative' in sample:
                            caption_pos = sample['caption']
                            caption_neg = sample['negative']
                        # VALSE/VG format
                        elif 'true_caption' in sample and 'false_caption' in sample:
                            caption_pos = sample['true_caption']
                            caption_neg = sample['false_caption']
                        else:
                            # Skip if we can't find the caption pair
                            continue
                        
                        if image is None:
                            continue
                            
                    elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                        # Tuple format: (image, pos_caption, neg_caption)
                        image = sample[0]
                        caption_pos = sample[1]
                        caption_neg = sample[2]
                    else:
                        continue
                    
                    # Preprocess image
                    if hasattr(image, 'mode'):  # PIL Image
                        image_tensor = preprocess(image).unsqueeze(0).to(device)
                    elif isinstance(image, torch.Tensor):
                        if image.dim() == 3:
                            image_tensor = image.unsqueeze(0).to(device)
                        else:
                            image_tensor = image.to(device)
                    else:
                        continue
                    
                    # Tokenize captions
                    pos_tokens = clip.tokenize([caption_pos], truncate=True).to(device)
                    neg_tokens = clip.tokenize([caption_neg], truncate=True).to(device)
                    
                    # Get caption-conditioned image embeddings using TQA cross-attention
                    out_pos = base_model.encode_multimodal(image=image_tensor, text_tokens=pos_tokens)
                    out_neg = base_model.encode_multimodal(image=image_tensor, text_tokens=neg_tokens)
                    
                    img_emb_pos = out_pos['image_embeds']  # Image conditioned on positive caption
                    img_emb_neg = out_neg['image_embeds']  # Image conditioned on negative caption
                    text_emb_pos = out_pos['text_embeds']
                    text_emb_neg = out_neg['text_embeds']
                    
                    # Normalize embeddings
                    img_emb_pos = img_emb_pos / img_emb_pos.norm(dim=-1, keepdim=True)
                    img_emb_neg = img_emb_neg / img_emb_neg.norm(dim=-1, keepdim=True)
                    text_emb_pos = text_emb_pos / text_emb_pos.norm(dim=-1, keepdim=True)
                    text_emb_neg = text_emb_neg / text_emb_neg.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    # For compositional benchmarks: positive image-text should have higher similarity
                    # than negative image-text
                    sim_pos = (img_emb_pos @ text_emb_pos.T).item()
                    sim_neg = (img_emb_neg @ text_emb_neg.T).item()
                    
                    # Correct if positive similarity > negative similarity
                    if sim_pos > sim_neg:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logging.debug(f"Error processing sample {i}: {e}")
                    continue
        
        if total == 0:
            logging.warning("TQA evaluation processed 0 samples")
            return None
        
        accuracy = correct / total
        logging.info(f"✅ TQA Compositional Eval: {correct}/{total} = {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'text_contrastive_accuracy': accuracy,
            'tqa_conditioned_accuracy': accuracy,
            'total_samples': total,
            'num_samples': total
        }

    def _evaluate_single_dataset(self, model, clip_model, preprocess, dataset_name: str, subset_name: str, alignment_type: str = "HNB", is_initial_eval: bool = False) -> Dict[str, float]:
        """Evaluate model on a single dataset/subset with automatic method detection."""
        import signal
        import logging
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Dataset evaluation timed out")
        # Set a timeout of 2 hours per dataset (7200 seconds)
        # Large datasets like VL_CheckList/obj_location with ViT-L/14 need more time
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(7200)  # 2 hours
        try:
            logging.info(f"🔍 Starting evaluation: {dataset_name}/{subset_name} (alignment: {alignment_type}, initial: {is_initial_eval})")
            
            # Create args for this dataset
            args = self._create_args_for_dataset(dataset_name, subset_name)
            
            # Build dataset using existing system
            dataset = build_dataset_from_args(args, preprocess=preprocess)
            logging.info(f"✅ Dataset loaded: {len(dataset)} samples")
            
            # Sample dataset for visualization if enabled
            if self.viz_logger is not None:
                logging.info(f"📸 Attempting to sample {dataset_name}/{subset_name} for visualization...")
                try:
                    self.viz_logger.sample_dataset(
                        dataset=dataset,
                        dataset_name=dataset_name,
                        subset_name=subset_name,
                        preprocess=preprocess
                    )
                    key = f"{dataset_name}/{subset_name}"
                    if key in self.viz_logger.dataset_samples:
                        num_samples = len(self.viz_logger.dataset_samples[key])
                        logging.info(f"✅ Successfully sampled {num_samples} items from {key}")
                    else:
                        logging.warning(f"⚠️ Sampling completed but no samples stored for {key}")
                except Exception as e:
                    logging.error(f"❌ Failed to sample {dataset_name}/{subset_name}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
            else:
                logging.debug(f"ℹ️ Visualization logger is None, skipping sampling for {dataset_name}/{subset_name}")
            
            # Get device from model
            device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'
            logging.info(f"📱 Using device: {device}")
            
            # Update viz_logger device if needed
            if self.viz_logger is not None and self.viz_logger.device != device:
                self.viz_logger.device = device
            
            # Determine evaluation method based on model type
            if self._is_labclip_model(model) and not is_initial_eval:
                # Use LabCLIP evaluation (direct similarity computation)
                logging.info(f"🚀 Using LabCLIP evaluation for {dataset_name}/{subset_name}")
                results = self._evaluate_with_labclip(model, clip_model, preprocess, dataset, device)
            elif self._is_tqa_model(model) and dataset_name in self.COMPOSITIONAL_BENCHMARKS and not is_initial_eval:
                # Use TQA caption-conditioned evaluation for compositional benchmarks
                logging.info(f"🎯 Using TQA caption-conditioned evaluation for {dataset_name}/{subset_name}")
                results = self._evaluate_compositional_with_tqa(model, clip_model, preprocess, dataset, device)
                if results is None:
                    # Fall back to standard evaluation if TQA eval fails
                    logging.info(f"📊 Falling back to standard CLIP evaluation for {dataset_name}/{subset_name}")
                    results = self._evaluate_with_standard_clip(
                        model, clip_model, preprocess, dataset, alignment_type, is_initial_eval, device
                    )
            else:
                # Use standard CLIP evaluation (separate embeddings)
                logging.info(f"📊 Using standard CLIP evaluation for {dataset_name}/{subset_name}")
                results = self._evaluate_with_standard_clip(
                    model, clip_model, preprocess, dataset, alignment_type, is_initial_eval, device
                )
            
            # Log results
            if results:
                logging.info(f"✅ {dataset_name}/{subset_name} evaluation complete:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"   {key}: {value:.4f}")
                    else:
                        logging.info(f"   {key}: {value}")
            else:
                logging.warning(f"⚠️ {dataset_name}/{subset_name} returned empty results")
            
            # Cancel the timeout
            signal.alarm(0)
            return results
            
        except TimeoutError:
            error_msg = f"Evaluation of {dataset_name}/{subset_name} timed out after 2 hours"
            logging.error(f"⏰ {error_msg}")
            signal.alarm(0)
            return {'accuracy': 0.0, 'total_samples': 0, 'num_samples': 0, 'error': 'timeout'}
            
        except Exception as e:
            error_msg = f"Error evaluating {dataset_name}/{subset_name}: {e}"
            logging.error(f"❌ {error_msg}")
            import traceback
            logging.error(f"📍 Traceback:\n{traceback.format_exc()}")
            signal.alarm(0)
            return {'accuracy': 0.0, 'total_samples': 0, 'num_samples': 0, 'error': str(e)}
    
    def evaluate_all(
        self, 
        model: torch.nn.Module, 
        clip_model: torch.nn.Module,
        preprocess,
        step: int,
        epoch: int = None,
        alignment_type: str = "HNB",
        wandb_log: bool = True,
        is_initial_eval: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on all configured datasets.
        
        Args:
            model: Trained model to evaluate
            clip_model: Original CLIP model (for LabCLIP) or None (for FT)
            preprocess: CLIP preprocessing function
            step: Current training step (global_step for step-based eval, or epoch number for epoch-based eval)
            epoch: Current training epoch (None for step-based evaluation, epoch number for epoch-based)
            alignment_type: Type of alignment ("HNB", "SB", "FT")
            wandb_log: Whether to log to wandb (uses step for x-axis)
            is_initial_eval: Whether this is initial evaluation (before training)
            
        Returns:
            Dict with flattened results in {dataset_name}/{subset_name}/{metric} format
        """
        all_results = {}
        
        # Check if we're in distributed training
        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        
        # Log what we're evaluating with
        import logging
        if epoch is not None:
            logging.info(f"📊 [Rank {rank}/{world_size}] Starting dataset evaluation at step={step}, epoch={epoch}")
        else:
            logging.info(f"📊 [Rank {rank}/{world_size}] Starting dataset evaluation at step={step} (step-based evaluation)")
        
        for dataset_name in self.datasets:
            if dataset_name not in self.dataset_subsets:
                print(f"Warning: Unknown dataset {dataset_name}, skipping...")
                continue
            
            # Regular processing for all datasets (including optimized VALSE)
            subsets = self.dataset_subsets[dataset_name]
            
            for subset_name in subsets:
                print(f"Evaluating {dataset_name}/{subset_name}...")
                
                # Evaluate this dataset/subset
                with torch.no_grad():
                    subset_results = self._evaluate_single_dataset(
                        model, clip_model, preprocess, dataset_name, subset_name, alignment_type, is_initial_eval
                    )
                
                # Store results in hierarchical format
                for metric, value in subset_results.items():
                    key = f"eval/{dataset_name}/{subset_name}/{metric}"
                    all_results[key] = value
                    
                    # Skip num_samples metric - it's not a meaningful metric to log
                    if metric == 'num_samples':
                        continue
                    
                    # Get total_samples for CSV logging (default to 0 if not present)
                    total_samples = subset_results.get('total_samples', 0)
                    
                    # Save to CSV (skip total_samples metric itself to avoid redundancy)
                    if metric != 'total_samples':
                        self._save_to_csv(
                            step=step,
                            epoch=epoch,
                            dataset=dataset_name,
                            subset=subset_name,
                            metric=metric,
                            value=value,
                            total_samples=total_samples
                        )
        
        # Generate visualization images (independent of wandb)
        # Only generate on main process (rank 0) in distributed training
        viz_results = None
        
        # Check if we're in distributed training
        is_main_process = True
        if torch.distributed.is_initialized():
            is_main_process = torch.distributed.get_rank() == 0
            if not is_main_process:
                logging.info(f"ℹ️ Skipping visualization generation on rank {torch.distributed.get_rank()} (not main process)")
        
        if is_main_process and self.viz_logger is not None:
            num_sampled = len(self.viz_logger.dataset_samples)
            if num_sampled > 0:
                logging.info(f"📸 Generating benchmark visualizations for {num_sampled} dataset/subset pairs...")
                logging.info(f"   Sampled datasets: {list(self.viz_logger.dataset_samples.keys())}")
                try:
                    viz_results = self.viz_logger.log_all_datasets(
                        model=model,
                        clip_model=clip_model,
                        epoch=epoch if epoch is not None else step,
                        wandb_prefix="benchmark_viz"
                    )
                    if viz_results:
                        logging.info(f"✅ Generated {len(viz_results.get('images', []))} visualization images")
                    else:
                        logging.warning("⚠️ Visualization generation returned empty results")
                except Exception as e:
                    logging.error(f"❌ Failed to generate visualizations: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logging.warning("⚠️ No dataset samples collected for visualization")
                logging.warning("   This might indicate sampling failed during evaluation")
        elif not is_main_process:
            # Non-main process, skip visualization
            pass
        elif self.viz_logger is None:
            logging.info("ℹ️ Visualization logger not initialized (enable_visualization=False)")
        
        # Compute average text contrastive accuracy across all datasets
        text_contrastive_values = []
        for key, value in all_results.items():
            if key.endswith('/text_contrastive_accuracy') and isinstance(value, (int, float)):
                text_contrastive_values.append(value)
        
        if text_contrastive_values:
            avg_text_contrastive = sum(text_contrastive_values) / len(text_contrastive_values)
            all_results['eval/average_text_contrastive_accuracy'] = avg_text_contrastive
            all_results['eval/num_datasets_evaluated'] = len(text_contrastive_values)
            logging.info(f"📊 Average text contrastive accuracy: {avg_text_contrastive:.4f} (over {len(text_contrastive_values)} dataset/subsets)")
            
        
        # Log to wandb if enabled (also only on main process)
        if wandb_log and is_main_process:
            try:
                import wandb
                wandb.log(all_results, step=step)
                
                # Log visualization images to wandb if available
                if viz_results is not None:
                    if viz_results.get('images'):
                        wandb.log({
                            f"{viz_results['wandb_prefix']}/samples": viz_results['images']
                        }, step=step)
                        logging.info(f"📸 Logged {len(viz_results['images'])} visualization images to wandb")
                    
                    # Log visualization metrics
                    if viz_results.get('metrics'):
                        viz_metrics = {f"eval/{k}": v for k, v in viz_results['metrics'].items()}
                        wandb.log(viz_metrics, step=step)
                        logging.info(f"📊 Logged {len(viz_metrics)} visualization metrics to wandb")
                else:
                    logging.warning("⚠️ No visualization results to log to wandb")
                        
            except ImportError:
                print("wandb not available, skipping wandb logging")
        
        return all_results
    
    def _save_to_csv(
        self, 
        step: int, 
        epoch: Optional[int], 
        dataset: str, 
        subset: str, 
        metric: str, 
        value: float, 
        total_samples: int
    ):
        """Save results to CSV file with thread safety."""
        with self.csv_lock:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    step,
                    epoch if epoch is not None else '',
                    dataset,
                    subset,
                    metric,
                    value,
                    total_samples
                ])


def setup_dataset_evaluation(
    datasets: List[str],
    csv_path: str = "dataset_evaluation_results.csv",
    enable_visualization: bool = True,
    num_viz_samples: int = 5,
    device=None,
    is_ft: bool = False
) -> DatasetEvaluator:
    """
    Setup dataset evaluation with specified datasets.
    
    Args:
        datasets: List of dataset names to evaluate
        csv_path: Path to save CSV results
        enable_visualization: Whether to enable benchmark visualization logging
        num_viz_samples: Number of samples to visualize per dataset
        device: Device for visualization computation
        is_ft: Whether using fine-tuning mode
        
    Returns:
        DatasetEvaluator instance
    """
    return DatasetEvaluator(
        datasets=datasets, 
        csv_path=csv_path,
        enable_visualization=enable_visualization,
        num_viz_samples=num_viz_samples,
        device=device,
        is_ft=is_ft
    )


def evaluate_and_log_results_with_datasets(
    model,
    clip_model,
    preprocess,
    dataset_evaluator: DatasetEvaluator,
    step: int,
    epoch: int = None,
    alignment_type: str = "HNB",
    wandb_log: bool = True,
    is_initial_eval: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model on configured datasets and log results.
    
    Args:
        model: Trained model to evaluate
        clip_model: Original CLIP model (for LabCLIP) or None (for FT)
        preprocess: CLIP preprocessing function
        dataset_evaluator: DatasetEvaluator instance
        step: Current training step (for this use case, step = epoch for consistency)
        epoch: Current training epoch
        alignment_type: Type of alignment ("HNB", "SB", "FT")
        wandb_log: Whether to log to wandb
        is_initial_eval: Whether this is initial evaluation (before training)
        
    Returns:
        Dict with evaluation results
    """
    if dataset_evaluator is None:
        return {}
    
    return dataset_evaluator.evaluate_all(
        model=model,
        clip_model=clip_model,
        preprocess=preprocess,
        alignment_type=alignment_type,
        step=step,
        epoch=epoch,
        wandb_log=wandb_log,
        is_initial_eval=is_initial_eval
    )
