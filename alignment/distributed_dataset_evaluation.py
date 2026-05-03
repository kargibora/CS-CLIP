"""Distributed dataset evaluation utilities."""

import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional
import logging


class DistributedDatasetEvaluator:
    """
    Wrapper around DatasetEvaluator that distributes evaluation across multiple GPUs.
    
    Strategy:
    1. Split dataset/subset combinations across available GPUs
    2. Each GPU evaluates its assigned datasets in parallel
    3. Collect results on main process using all_gather
    4. Combine and return complete results
    """
    
    def __init__(self, base_evaluator):
        """
        Initialize distributed evaluator.
        
        Args:
            base_evaluator: DatasetEvaluator instance to wrap
        """
        self.base_evaluator = base_evaluator
        self.is_distributed = dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.rank = dist.get_rank() if self.is_distributed else 0
        
        logging.info(f"DistributedDatasetEvaluator initialized: rank={self.rank}, world_size={self.world_size}")
    
    def _get_dataset_subset_list(self) -> List[tuple]:
        """
        Get flat list of (dataset_name, subset_name) tuples to evaluate.
        
        Returns:
            List of (dataset_name, subset_name) tuples
        """
        dataset_subset_list = []
        
        for dataset_name in self.base_evaluator.datasets:
            if dataset_name not in self.base_evaluator.dataset_subsets:
                logging.warning(f"Unknown dataset {dataset_name}, skipping...")
                continue
            
            subsets = self.base_evaluator.dataset_subsets[dataset_name]
            for subset_name in subsets:
                dataset_subset_list.append((dataset_name, subset_name))
        
        return dataset_subset_list
    
    def _split_workload(self, dataset_subset_list: List[tuple]) -> List[tuple]:
        """
        Split dataset/subset combinations across GPUs in round-robin fashion.
        
        Args:
            dataset_subset_list: Full list of (dataset_name, subset_name) tuples
            
        Returns:
            List of (dataset_name, subset_name) tuples for this GPU
        """
        # Round-robin assignment: GPU i gets items [i, i+world_size, i+2*world_size, ...]
        my_datasets = dataset_subset_list[self.rank::self.world_size]
        
        logging.info(f"Rank {self.rank}: Assigned {len(my_datasets)} datasets out of {len(dataset_subset_list)} total")
        for dataset_name, subset_name in my_datasets:
            logging.info(f"   - {dataset_name}/{subset_name}")
        
        return my_datasets
    
    def _evaluate_my_datasets(
        self,
        my_datasets: List[tuple],
        model: torch.nn.Module,
        clip_model: torch.nn.Module,
        preprocess,
        alignment_type: str,
        is_initial_eval: bool
    ) -> Dict[str, Any]:
        """
        Evaluate datasets assigned to this GPU.
        
        Args:
            my_datasets: List of (dataset_name, subset_name) tuples for this GPU
            model: Model to evaluate
            clip_model: CLIP model
            preprocess: Preprocessing function
            alignment_type: Type of alignment
            is_initial_eval: Whether this is initial evaluation
            
        Returns:
            Dict of results for datasets evaluated by this GPU
        """
        my_results = {}
        
        for dataset_name, subset_name in my_datasets:
            logging.info(f"Rank {self.rank}: Evaluating {dataset_name}/{subset_name}...")
            
            try:
                # Evaluate this dataset/subset using base evaluator
                with torch.no_grad():
                    subset_results = self.base_evaluator._evaluate_single_dataset(
                        model=model,
                        clip_model=clip_model,
                        preprocess=preprocess,
                        dataset_name=dataset_name,
                        subset_name=subset_name,
                        alignment_type=alignment_type,
                        is_initial_eval=is_initial_eval
                    )
                
                # Store results with full key path
                for metric, value in subset_results.items():
                    key = f"eval/{dataset_name}/{subset_name}/{metric}"
                    my_results[key] = value
                
                logging.info(f"Rank {self.rank}: {dataset_name}/{subset_name} completed")
                
            except Exception as e:
                logging.error(f"Rank {self.rank}: Error evaluating {dataset_name}/{subset_name}: {e}")
                import traceback
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                
                # Store error placeholder
                key = f"eval/{dataset_name}/{subset_name}/accuracy"
                my_results[key] = 0.0
                my_results[f"eval/{dataset_name}/{subset_name}/error"] = str(e)
        
        return my_results
    
    def _gather_results(self, my_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather results from all GPUs to rank 0.
        
        Args:
            my_results: Results from this GPU
            
        Returns:
            Combined results from all GPUs (only valid on rank 0)
        """
        if not self.is_distributed:
            # Single GPU: just return my results
            return my_results
        
        # Synchronize before gathering
        dist.barrier()
        
        # Convert results to lists of (key, value) tuples for gathering
        # We'll gather the keys and values separately
        my_keys = list(my_results.keys())
        my_values = [my_results[k] for k in my_keys]
        
        # Gather all keys and values on rank 0
        if self.rank == 0:
            all_keys = [None] * self.world_size
            all_values = [None] * self.world_size
        else:
            all_keys = None
            all_values = None
        
        # Use object list gathering for arbitrary python objects
        dist.gather_object(my_keys, all_keys, dst=0)
        dist.gather_object(my_values, all_values, dst=0)
        
        # Combine results on rank 0
        if self.rank == 0:
            combined_results = {}
            for keys, values in zip(all_keys, all_values):
                for key, value in zip(keys, values):
                    if key not in combined_results:
                        combined_results[key] = value
                    else:
                        # Handle duplicate keys (shouldn't happen with proper workload split)
                        logging.warning(f"Duplicate key {key} found during gather, keeping first value")
            
            logging.info(f"Gathered {len(combined_results)} results from {self.world_size} GPUs")
            return combined_results
        else:
            # Non-main processes return empty dict
            return {}
    
    def _save_results_to_csv(
        self,
        all_results: Dict[str, Any],
        step: int,
        epoch: Optional[int]
    ):
        """
        Save combined results to CSV (only on rank 0).
        
        Args:
            all_results: Combined results from all GPUs
            step: Current step
            epoch: Current epoch
        """
        if self.rank != 0:
            return  # Only main process saves to CSV
        
        # Parse results and save to CSV
        for key, value in all_results.items():
            # Parse key: "eval/dataset_name/subset_name/metric"
            parts = key.split('/')
            if len(parts) >= 4 and parts[0] == 'eval':
                dataset_name = parts[1]
                subset_name = parts[2]
                metric = '/'.join(parts[3:])  # Metric might contain slashes
                
                # Get total samples if available
                total_samples_key = f"eval/{dataset_name}/{subset_name}/num_samples"
                total_samples = all_results.get(total_samples_key, 0)
                
                # Save to CSV
                self.base_evaluator._save_to_csv(
                    step=step,
                    epoch=epoch,
                    dataset=dataset_name,
                    subset=subset_name,
                    metric=metric,
                    value=value,
                    total_samples=total_samples
                )
    
    def evaluate_all(
        self,
        model: torch.nn.Module,
        clip_model: torch.nn.Module,
        preprocess,
        step: int,
        epoch: Optional[int] = None,
        alignment_type: str = "HNB",
        is_initial_eval: bool = False
    ) -> Dict[str, Any]:
        """
        Distributed evaluation across all GPUs.
        
        Args:
            model: Model to evaluate
            clip_model: CLIP model
            preprocess: Preprocessing function
            step: Current training step
            epoch: Current training epoch
            alignment_type: Type of alignment ("HNB", "SB", "FT")
            is_initial_eval: Whether this is initial evaluation
            
        Returns:
            Dict with all evaluation results (on rank 0), empty dict on other ranks
        """
        logging.info(f"Rank {self.rank}: Starting distributed dataset evaluation")
        
        # Step 1: Get full list of datasets to evaluate (all ranks need this)
        dataset_subset_list = self._get_dataset_subset_list()
        
        if not dataset_subset_list:
            logging.warning(f"Rank {self.rank}: No datasets to evaluate")
            return {}
        
        # Step 2: Split workload across GPUs
        my_datasets = self._split_workload(dataset_subset_list)
        
        # Step 3: Each GPU evaluates its assigned datasets
        my_results = self._evaluate_my_datasets(
            my_datasets=my_datasets,
            model=model,
            clip_model=clip_model,
            preprocess=preprocess,
            alignment_type=alignment_type,
            is_initial_eval=is_initial_eval
        )
        
        logging.info(f"Rank {self.rank}: Local evaluation complete, {len(my_results)} results")
        
        # Step 4: Gather all results to rank 0
        all_results = self._gather_results(my_results)
        
        if self.rank == 0:
            self._save_results_to_csv(all_results, step, epoch)
            logging.info(f"Distributed dataset evaluation complete: {len(all_results)} total results")
        
        # Synchronize before returning
        if self.is_distributed:
            dist.barrier()
        
        return all_results if self.rank == 0 else {}


def setup_distributed_dataset_evaluation(
    datasets: List[str],
    csv_path: str = "dataset_evaluation_results.csv"
):
    """
    Setup distributed dataset evaluation.
    
    Args:
        datasets: List of dataset names to evaluate
        csv_path: Path to save CSV results
        
    Returns:
        DistributedDatasetEvaluator instance
    """
    from alignment.simple_dataset_evaluation import setup_dataset_evaluation
    
    # Create base evaluator
    base_evaluator = setup_dataset_evaluation(datasets=datasets, csv_path=csv_path)
    
    # Wrap with distributed evaluator
    distributed_evaluator = DistributedDatasetEvaluator(base_evaluator)
    
    return distributed_evaluator
