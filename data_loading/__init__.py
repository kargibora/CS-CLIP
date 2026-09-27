"""COCO training loader; benchmark dependencies are loaded only for evaluation."""
from .coco_neg import COCODataset, COCONeg, COCONegDataset


def build_dataset_from_args(args, preprocess=None):
    if args.dataset == "COCONeg":
        kwargs = dict(getattr(args, "dataset_kwargs", {}) or {})
        for field in ("json_folder", "image_root"):
            if not kwargs.get(field):
                raise ValueError(f"COCONeg requires dataset.dataset_kwargs.{field}.")
        return COCODataset(image_preprocess=preprocess, subset_name=args.subset_name, **kwargs)
    from .benchmarks import build_dataset_from_args as build_benchmark
    return build_benchmark(args, preprocess)


def get_dataset_embedding_class(name):
    if name == "COCONeg":
        return COCONeg
    from .benchmarks import get_dataset_embedding_class as get_benchmark
    return get_benchmark(name)


def get_dataset_class(name):
    from .benchmarks import get_dataset_class as get_benchmark
    return get_benchmark(name)


def build_sampler(name, **kwargs):
    from .benchmarks import build_sampler as build_benchmark_sampler
    return build_benchmark_sampler(name, **kwargs)
