"""Dataset evaluation utilities."""

import os
import csv
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import argparse
import signal
import logging
import clip

from data_loading import build_dataset_from_args


class DatasetEvaluator:
    """Evaluates multiple benchmark datasets during training using existing dataset loading system."""
    
    def __init__(
        self, 
        datasets=None, 
        csv_path="dataset_evaluation_results.csv",
        dataset_root=None,
        device=None,
        is_ft=False
    ):
        """
        Initialize the DatasetEvaluator.
        
        Args:
            datasets (list): List of dataset names to evaluate (e.g., ['VALSE', 'BLA', 'ColorFoil', 'CLIPBench_wds_imagenet1k'])
            csv_path (str): Path to save CSV results
            dataset_root: Root directory containing evaluation datasets
            device: Device for evaluation
            is_ft (bool): Whether using fine-tuning mode
        """
        # Parse CLIPBench datasets: group them together
        self.datasets = self._parse_datasets(datasets or [])
        self.csv_path = csv_path
        self.csv_lock = threading.Lock()  # For thread-safe CSV writing
        self.dataset_root = dataset_root
        self.device = device
        self.is_ft = is_ft
        
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
            print(f"Grouped {len(clipbench_subsets)} CLIP Benchmark datasets:")
            for subset in clipbench_subsets:
                print(f"   - {subset}")
        else:
            self._clipbench_subsets = []
        
        return parsed_datasets

    def _resolve_dataset_path(self, dataset_name: str) -> Optional[str]:
        if not self.dataset_root:
            return None

        subdirs = {
            "BLA": "BLA_Benchmark",
            "VALSE": "VALSE",
            "VL_CheckList": "VL-CheckList",
            "ColorSwap": "ColorSwap",
            "ColorFoil": "ColorFoil",
            "COCO_Counterfactuals": "COCO-Counterfactuals",
            "NegBench": "NegBench",
            "ControlledImages": "WhatsUp",
            "VG_Attribution": "WhatsUp",
            "VG_Relation": "WhatsUp",
            "COCO_Order": "WhatsUp",
            "Flickr30k_Order": "WhatsUp",
            "CC3M": "CC3M",
            "VisMin": "VisMin",
            "SugarCrepe": "SugarCrepe",
            "SugarCrepe_PP": "SugarCrepe",
            "Winoground": "Winoground",
            "SPEC_I2T": "SPEC",
            "COLA": "cola",
            "CLIPBenchmark": "clip_benchmark",
            "SVOProbes": "svo_probes",
            "MMVP": "MMVP",
        }
        return os.path.join(self.dataset_root, subdirs.get(dataset_name, dataset_name))
    
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
        
        args.data_path = self._resolve_dataset_path(args.dataset)
        return args
    
    
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
            logging.info(f"Starting evaluation: {dataset_name}/{subset_name} (alignment: {alignment_type}, initial: {is_initial_eval})")
            
            # Create args for this dataset
            args = self._create_args_for_dataset(dataset_name, subset_name)
            
            # Build dataset using existing system
            dataset = build_dataset_from_args(args, preprocess=preprocess)
            logging.info(f"Dataset loaded: {len(dataset)} samples")
            
            # Get device from model
            device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'
            logging.info(f"Using device: {device}")
            
            # Use standard CLIP evaluation (separate embeddings)
            logging.info(f"Using standard CLIP evaluation for {dataset_name}/{subset_name}")
            results = self._evaluate_with_standard_clip(
                model, clip_model, preprocess, dataset, alignment_type, is_initial_eval, device
            )
            
            # Log results
            if results:
                logging.info(f"{dataset_name}/{subset_name} evaluation complete:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"   {key}: {value:.4f}")
                    else:
                        logging.info(f"   {key}: {value}")
            else:
                logging.warning(f"{dataset_name}/{subset_name} returned empty results")
            
            # Cancel the timeout
            signal.alarm(0)
            return results
            
        except TimeoutError:
            error_msg = f"Evaluation of {dataset_name}/{subset_name} timed out after 2 hours"
            logging.error(error_msg)
            signal.alarm(0)
            return {'accuracy': 0.0, 'total_samples': 0, 'num_samples': 0, 'error': 'timeout'}
            
        except Exception as e:
            error_msg = f"Error evaluating {dataset_name}/{subset_name}: {e}"
            logging.error(error_msg)
            import traceback
            logging.error(f"Traceback:\n{traceback.format_exc()}")
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
        is_initial_eval: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on all configured datasets.
        
        Args:
            model: Trained model to evaluate
            clip_model: Original CLIP model or None (for FT)
            preprocess: CLIP preprocessing function
            step: Current training step (global_step for step-based eval, or epoch number for epoch-based eval)
            epoch: Current training epoch (None for step-based evaluation, epoch number for epoch-based)
            alignment_type: Type of alignment ("HNB", "SB", "FT")
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
            logging.info(f"[Rank {rank}/{world_size}] Starting dataset evaluation at step={step}, epoch={epoch}")
        else:
            logging.info(f"[Rank {rank}/{world_size}] Starting dataset evaluation at step={step}")
        
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
        
        # Compute average text contrastive accuracy across all datasets
        text_contrastive_values = []
        for key, value in all_results.items():
            if key.endswith('/text_contrastive_accuracy') and isinstance(value, (int, float)):
                text_contrastive_values.append(value)
        
        if text_contrastive_values:
            avg_text_contrastive = sum(text_contrastive_values) / len(text_contrastive_values)
            all_results['eval/average_text_contrastive_accuracy'] = avg_text_contrastive
            all_results['eval/num_datasets_evaluated'] = len(text_contrastive_values)
            logging.info(f"Average text contrastive accuracy: {avg_text_contrastive:.4f} (over {len(text_contrastive_values)} dataset/subsets)")
        
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
    dataset_root: Optional[str] = None,
    device=None,
    is_ft: bool = False
) -> DatasetEvaluator:
    """
    Setup dataset evaluation with specified datasets.
    
    Args:
        datasets: List of dataset names to evaluate
        csv_path: Path to save CSV results
        dataset_root: Root directory containing evaluation datasets
        device: Device for evaluation
        is_ft: Whether using fine-tuning mode
        
    Returns:
        DatasetEvaluator instance
    """
    return DatasetEvaluator(
        datasets=datasets, 
        csv_path=csv_path,
        dataset_root=dataset_root,
        device=device,
        is_ft=is_ft
    )
