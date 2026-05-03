import os
from .aro import VG_Attribution, VG_Relation, COCO_Order, Flickr30k_Order, ARONeg
from .base import BaseEmbeddingsDataset, BaseNegEmbeddingsDataset
from .cc3m import CC3MDataset, CC3MNeg
from .spec import SPECImage2TextDataset, SPECNeg, GroupUniqueBatchSampler
from .vismin import VisMinDataset, VisMinNeg
from .whatsup import Controlled_Images, ControlledImagesNeg
from .sugarcrepe_pp import SugarCrepePPDataset, SugarCrepeNeg
from .sugarcrepe import SugarCrepeDataset
from .winoground import WinogroundDataset, WinogroundNeg
from .bla import BLADataset
from .valse import VALSEDataset
from .vl_checklist import VLCheckListDataset
from .colorswap import ColorSwapDataset
from .colorfoil import ColorFoilDataset
from .coco_counterfactuals import COCOCounterfactualsDataset
from .negbench import NegBenchDataset
from .cola import COLAMultiObjectDataset, COLASingleObjectDataset
from .clip_benchmark import CLIPBenchmarkDataset
from .svo_probes import SVOProbesDataset, SVOProbesNeg
from .mmvp import MMVPDataset, MMVPNeg
from .coco_neg import COCODataset, COCONeg, COCONegDataset

def get_dataset_class(name):
    """
    Returns the dataset class based on the provided name.
    
    Args:
        name (str): The name of the dataset class to retrieve.
        
    Returns:
        type: The dataset class corresponding to the provided name.
    """
    dataset_classes = {
        "VG_Attribution": VG_Attribution,
        "VG_Relation": VG_Relation,
        "COCO_Order": COCO_Order,
        "Flickr30k_Order": Flickr30k_Order,
        "CC3M": CC3MDataset,
        "SPEC": SPECImage2TextDataset,
        "SPEC_I2T": SPECImage2TextDataset,
        "VisMin": VisMinDataset,
        "ControlledImages": Controlled_Images,
        "SugarCrepe": SugarCrepeDataset,
        "SugarCrepe_PP": SugarCrepePPDataset,
        "Winoground": WinogroundDataset,
        "BLA": BLADataset,
        "VALSE": VALSEDataset,
        "VL_CheckList": VLCheckListDataset,
        "ColorSwap": ColorSwapDataset,
        "ColorFoil": ColorFoilDataset,
        "COCO_Counterfactuals": COCOCounterfactualsDataset,
        "COLA": COLAMultiObjectDataset,  # Default to multi-object setting
        "NegBench": NegBenchDataset,
        "CLIPBenchmark": CLIPBenchmarkDataset,
        "SVOProbes": SVOProbesDataset,
        "MMVP": MMVPDataset,
    }
    return dataset_classes.get(name, None)  # Return None if not found

def get_dataset_embedding_class(name):
    """
    Returns the embedding dataset class based on the provided name.
    
    Args:
        name (str): The name of the embedding dataset class to retrieve
    """
    embedding_classes = {
        "BaseEmbeddingsDataset": BaseEmbeddingsDataset,
        "BaseNegEmbeddingsDataset": BaseNegEmbeddingsDataset,
        "CC3M": CC3MNeg,
        "SPEC": SPECNeg,
        "VisMin": VisMinNeg,
        "Controlled_Images": ControlledImagesNeg,
        "VG_Attribution": ARONeg,
        "VG_Relation": ARONeg,
        "COCO_Order": ARONeg,
        "Flickr30k_Order": ARONeg,
        "SugarCrepe": SugarCrepeNeg,
        "SugarCrepe_PP": SugarCrepeNeg,
        "Winoground": WinogroundNeg,
        "SVOProbes": SVOProbesNeg,
        "MMVP": MMVPNeg,
        "COCONeg": COCONeg,
    }
    
    return embedding_classes.get(name, None)  # Return None if not found


def build_sampler(name, **sampler_kwargs):
    """
    Returns the sampler class based on the provided name.
    
    Args:
        name (str): The name of the sampler class to retrieve.
        dataset (Dataset): The dataset to be used with the sampler.
        indices (list, optional): Specific indices to sample from. Defaults to None.
        
    Returns:
        type: The sampler class corresponding to the provided name.
    """
    sampler_classes = {
        "SPEC_I2T": GroupUniqueBatchSampler,
    }
    
    sampler_class = sampler_classes.get(name, None)
    if sampler_class is not None:
        return sampler_class(**sampler_kwargs)
    
    return None  # Return None if not found


def build_dataset_from_args(args, preprocess=None):
    """
    Builds a dataset based on the provided arguments.
    The main process detection is now handled automatically by the datasets.
    Supports dataset_kwargs for dataset-specific configuration.
    """
    
    # Get dataset-specific kwargs if available
    dataset_kwargs = {}
    if hasattr(args, 'dataset_kwargs') and args.dataset_kwargs is not None:
        dataset_kwargs = args.dataset_kwargs
    
    # Set default data_path for each dataset if not provided
    if args.data_path is None:
        if args.dataset == 'BLA':
            args.data_path = "./datasets/BLA_Benchmark"
        elif args.dataset == 'VALSE':
            args.data_path = "./datasets/VALSE"
        elif args.dataset == 'VL_CheckList':
            args.data_path = "./datasets/VL-CheckList"
        elif args.dataset == 'ColorSwap':
            args.data_path = "./datasets/ColorSwap"
        elif args.dataset == 'ColorFoil':
            args.data_path = "./datasets/ColorFoil"
        elif args.dataset == 'COCO_Counterfactuals':
            args.data_path = "./datasets/COCO-Counterfactuals"
        elif args.dataset == 'NegBench':
            args.data_path = "./datasets/NegBench"
        elif args.dataset == 'ControlledImages':
            args.data_path = "./datasets/WhatsUp"
        elif args.dataset == 'CC3M':
            args.data_path = "./datasets/CC3M"
        elif args.dataset == 'VG_Attribution':
            args.data_path = "./datasets/WhatsUp"
        elif args.dataset == 'VG_Relation':
            args.data_path = "./datasets/WhatsUp"
        elif args.dataset == 'COCO_Order':
            args.data_path = "./datasets/WhatsUp"
        elif args.dataset == 'Flickr30k_Order':
            args.data_path = "./datasets/WhatsUp"
        elif args.dataset == 'VisMin':
            args.data_path = "./datasets/VisMin"
        elif args.dataset == 'SugarCrepe':
            args.data_path = "./datasets/SugarCrepe"
        elif args.dataset == 'SugarCrepe_PP':
            args.data_path = "./datasets/SugarCrepe"
        elif args.dataset == 'Winoground':
            args.data_path = "./datasets/Winoground"
        elif args.dataset == 'SPEC_I2T':
            args.data_path = './datasets/SPEC'
        elif args.dataset == 'COLA':
            args.data_path = './datasets/cola'
        elif args.dataset == 'CLIPBenchmark':
            args.data_path = './datasets/clip_benchmark'
        elif args.dataset == 'SVOProbes':
            args.data_path = './datasets/svo_probes'
        elif args.dataset == 'MMVP':
            args.data_path = './datasets/MMVP'
        else:
            args.data_path = "./datasets"

    if args.dataset == 'BLA':
        # Default kwargs with override capability
        default_kwargs = {'split': 'test'}
        default_kwargs.update(dataset_kwargs)
        dataset = BLADataset(args.data_path, subset=args.subset_name, image_preprocess=preprocess, **default_kwargs)
        
    elif args.dataset == 'VALSE':
        dataset = VALSEDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'VL_CheckList':
        dataset = VLCheckListDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'ColorSwap':
        dataset = ColorSwapDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'ColorFoil':
        dataset = ColorFoilDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'COCO_Counterfactuals':
        dataset = COCOCounterfactualsDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'ControlledImages':
        dataset = Controlled_Images(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'CC3M':
        default_kwargs = {'combine_by_caption_id': False}
        default_kwargs.update(dataset_kwargs)
        dataset = CC3MDataset(args.data_path, args.subset_name, preprocess, **default_kwargs)
        
    elif args.dataset == 'VG_Attribution':
        default_kwargs = {'download': False}
        default_kwargs.update(dataset_kwargs)
        dataset = VG_Attribution(args.data_path, args.subset_name, preprocess, **default_kwargs)
        
    elif args.dataset == 'VG_Relation':
        default_kwargs = {'download': False}
        default_kwargs.update(dataset_kwargs)
        dataset = VG_Relation(args.data_path, args.subset_name, preprocess, **default_kwargs)
        
    elif args.dataset == 'COCO_Order':
        default_kwargs = {'download': False}
        default_kwargs.update(dataset_kwargs)
        dataset = COCO_Order(args.data_path, args.subset_name, preprocess, **default_kwargs)
        
    elif args.dataset == 'Flickr30k_Order':
        default_kwargs = {'download': False}
        default_kwargs.update(dataset_kwargs)
        dataset = Flickr30k_Order(args.data_path, args.subset_name, preprocess, **default_kwargs)
        
    elif args.dataset == 'VisMin':
        dataset = VisMinDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'SugarCrepe':
        default_kwargs = {'coco_root': args.data_path}
        default_kwargs.update(dataset_kwargs)
        dataset = SugarCrepeDataset(args.data_path, args.subset_name, image_preprocess=preprocess, **default_kwargs)
        
    elif args.dataset == 'SugarCrepe_PP':
        default_kwargs = {'coco_root': args.data_path}
        default_kwargs.update(dataset_kwargs)
        dataset = SugarCrepePPDataset(args.subset_name, image_preprocess=preprocess, **default_kwargs)
        
    elif args.dataset == 'SPEC_I2T':
        dataset = SPECImage2TextDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
        
    elif args.dataset == 'Winoground':
        default_kwargs = {'use_auth_token': "your_token"}
        default_kwargs.update(dataset_kwargs)
        dataset = WinogroundDataset(args.data_path, args.subset_name, preprocess, **default_kwargs)
        
    elif args.dataset == 'NegBench':
        dataset = NegBenchDataset(args.data_path, args.subset_name, preprocess, **dataset_kwargs)
    
    elif args.dataset == 'COCONeg':
        default_kwargs = {
            'num_entity_captions': 3,
            'use_structured_sampling': True,
            'structured_relation_prob': 0.5,
            'use_context_in_entity_pairs': True,
            'swap_negative_prob': 0.5,
            'inplace_replacement_prob': 0.7,
        }
        default_kwargs.update(dataset_kwargs)
        json_folder = default_kwargs.pop('json_folder', None)
        image_root = default_kwargs.pop('image_root', None)
        if not json_folder:
            raise ValueError("COCONeg requires dataset.dataset_kwargs.json_folder to be set.")
        if not image_root:
            raise ValueError("COCONeg requires dataset.dataset_kwargs.image_root to be set.")
        dataset = COCODataset(
            json_folder=json_folder,
            image_root=image_root,
            image_preprocess=preprocess,
            subset_name=args.subset_name,
            **default_kwargs
        )
    
    elif args.dataset == 'COLA':
        # Handle both multi-object and single-object subsets
        default_kwargs = {'download': True}
        default_kwargs.update(dataset_kwargs)
        
        if args.subset_name == 'multi_objects':
            dataset = COLAMultiObjectDataset(
                data_root=args.data_path,
                subset_name='multi_objects',
                image_preprocess=preprocess,
                **default_kwargs
            )
        elif args.subset_name.startswith('single_'):
            # Extract the actual subset name (e.g., 'single_GQA' -> 'GQA')
            single_subset = args.subset_name.replace('single_', '')
            dataset = COLASingleObjectDataset(
                data_root=args.data_path,
                subset_name=single_subset,
                image_preprocess=preprocess,
                **default_kwargs
            )
        else:
            raise ValueError(f"Invalid COLA subset: {args.subset_name}. Choose from ['multi_objects', 'single_GQA', 'single_CLEVR', 'single_PACO']")
    
    elif args.dataset == 'CLIPBenchmark':
        # CLIP Benchmark datasets (ImageNet, CIFAR, COCO, Flickr, etc.)
        default_kwargs = {
            'task': 'auto',  # Auto-detect from dataset name
            'split': 'test',
            'download': True,
        }
        default_kwargs.update(dataset_kwargs)
        
        # Get the actual dataset name from args.clip_benchmark_name
        # This is set by simple_dataset_evaluation.py when parsing CLIPBench_ prefix
        clip_benchmark_name = getattr(args, 'clip_benchmark_name', args.subset_name)
        
        dataset = CLIPBenchmarkDataset(
            dataset_name=clip_benchmark_name,
            data_root=args.data_path,
            image_preprocess=preprocess,
            **default_kwargs
        )
    
    elif args.dataset == 'SVOProbes':
        # SVO Probes dataset
        default_kwargs = {
            'download': True,
            'verbose': False,
        }
        default_kwargs.update(dataset_kwargs)
        dataset = SVOProbesDataset(
            data_root=args.data_path,
            subset_name=args.subset_name,
            image_preprocess=preprocess,
            **default_kwargs
        )
    elif args.dataset == 'MMVP':
        # MMVP dataset
        default_kwargs = {
            'download': True,
            'verbose': False,
        }
        default_kwargs.update(dataset_kwargs)
        dataset = MMVPDataset(
            data_root=args.data_path,
            subset_name=args.subset_name,
            image_preprocess=preprocess,
            **default_kwargs
        )
    else:
        raise ValueError("Invalid dataset. Choose from ['SPEC_I2T', 'ControlledImages', 'CC3M', 'VG_Attribution', 'VG_Relation', 'COCO_Order', 'Flickr30k_Order', 'VisMin', 'SugarCrepe', 'SugarCrepe_PP', 'Winoground', 'BLA', 'VALSE', 'VL_CheckList', 'ColorSwap', 'ColorFoil', 'COCO_Counterfactuals', 'COLA', 'NegBench', 'COCONeg', 'CLIPBenchmark', 'SVOProbes']")

    return dataset

def get_dataset_cache_name(args, cache_folder = None, is_image=True, is_dict=True):
    """
    Returns the cache name for the dataset based on the provided arguments.
    
    Args:
        args (argparse.Namespace): The parsed command line arguments.
        cache_folder (str): The folder where the cache is stored. If None, uses args.cache_folder or default.
        
    Returns:
        str: The cache name for the dataset.
    """
    if cache_folder is None:
        cache_folder = getattr(args, 'cache_folder', 'cache')
    if is_image:
        name = "img_emb"
    else:
        name = "text_emb"

    if is_dict:
        dict_postfix = '_dict'
    else:
        dict_postfix = ''

    return os.path.join(cache_folder, f"{args.dataset}_{name}_{args.subset_name}_embeddings{dict_postfix}.pt")
