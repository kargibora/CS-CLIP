# CS-CLIP: Component-Supervised CLIP

> **Code is currently being refactored. Full implementation will be released soon.**

## Paper

**[Half-Truths Break Similarity-Based Retrieval](https://arxiv.org/abs/2602.23906)**  
Bora Kargi, Arnas Uselis, Seong Joon Oh  
*arXiv preprint arXiv:2602.23906, 2026*

When a text description is extended with an additional detail, image-text similarity should drop if that detail is wrong. We show that CLIP-style dual encoders often violate this intuition: appending a plausible but incorrect object or relation to an otherwise correct description can increase the similarity score. We call such cases *half-truths*. On COCO, CLIP prefers the correct shorter description only 40.6% of the time, and performance drops to 32.9% when the added detail is a relation. We trace this vulnerability to weak supervision on caption parts: contrastive training aligns full sentences but does not explicitly enforce that individual entities and relations are grounded. We propose CS-CLIP (Component-Supervised CLIP), which decomposes captions into entity and relation units, constructs a minimally edited foil for each unit, and fine-tunes the model to score the correct unit above its foil while preserving standard dual-encoder inference. CS-CLIP raises half-truth accuracy to 69.3% and improves average performance on established compositional benchmarks by 5.7 points, suggesting that reducing half-truth errors aligns with broader gains in compositional understanding.

## Coming Soon

- [x] CS-CLIP COCO Checkpoint
- [x] Pre-extracted negatives
- [ ] Training and evaluation code
- [ ] Unit extraction pipeline
- [ ] Dataset setup instructions

## Pre-trained Checkpoints

| Model | Dataset | Download |
|-------|---------|----------|
| CS-CLIP-ViT-B/32 | MSCOCO | [link](https://drive.google.com/file/d/14IBgBgKhCDhRHJfnDaFxsTo6ocoS4I6W/view?usp=drive_link) |

## Datasets

Pre-extracted caption units (entities and relations) used for CS-CLIP experiments.

| Dataset | Description | Download |
|--------|-------------|----------|
| MSCOCO | Caption samples with extracted entities and relations | [link](https://drive.google.com/file/d/1DpthIA-5zT_m1GKfqvHUWWH_z2XyEOMP/view?usp=drive_link) |

## Citation

```bibtex
@article{kargi2026halftruths,
  title   = {Half-Truths Break Similarity-Based Retrieval},
  author  = {Kargi, Bora and Uselis, Arnas and Oh, Seong Joon},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.23906},
}
```
