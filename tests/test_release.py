"""Release checks: paper loss, real archive schema, checkpoints, and HT ties."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from alignment.ft_experiment import _save_run_artifacts
from alignment.losses import _compute_negclip_hard_loss_for_components
from data_loading import build_dataset_from_args, get_dataset_embedding_class
from scripts.evaluate_half_truth import summarize
from utils.dist import create_distributed_dataloader
from models import CLIPEndToEndPipeline
from open_clip import CLIP
from scripts.batch_evaluate_checkpoints import BatchCheckpointEvaluator

ROOT = Path(__file__).resolve().parents[1]


class ReleaseChecks(unittest.TestCase):
    def test_unit_loss_matches_paper_and_gradients(self):
        torch.manual_seed(42)
        images = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
        positives = F.normalize(torch.randn(4, 2, 8), dim=-1).requires_grad_()
        negatives = F.normalize(torch.randn(4, 2, 8), dim=-1).requires_grad_()
        temperature = torch.tensor(.07, requires_grad=True)
        actual, _ = _compute_negclip_hard_loss_for_components(
            images, positives, negatives, temperature, torch.device('cpu'), 2)
        losses = []
        labels = torch.arange(4)
        for k in range(2):
            pair_logits = images @ positives[:, k].T / temperature
            foil_logits = (images * negatives[:, k]).sum(-1, keepdim=True) / temperature
            losses.append((F.cross_entropy(torch.cat([pair_logits, foil_logits], dim=1), labels)
                           + F.cross_entropy(pair_logits.T, labels)) / 2)
        expected = torch.stack(losses).mean()
        torch.testing.assert_close(actual, expected)
        for a, b in zip(torch.autograd.grad(actual, (images, positives, negatives, temperature), retain_graph=True),
                        torch.autograd.grad(expected, (images, positives, negatives, temperature))):
            torch.testing.assert_close(a, b)

    def test_archive_fields_produce_valid_tokenized_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (32, 32)).save(root / 'train2014.jpg')
            sample = {'sample_id': '1', 'image_path': 'train2014.jpg', 'original_caption': 'A person rides a horse.',
                      'positive_components': ['person', 'horse'],
                      'negative_components': {'horse': [{'negative': 'cow', 'change_type': 'object_change'}]},
                      'relations': [{'subject': 'person', 'relation_type': 'rides', 'object': 'horse',
                                     'negatives': [{'relation_type': 'carries', 'change_type': 'antonym'}]}],
                      'swap_negatives': [{'negative': 'A horse rides a person.'}]}
            (root / 'samples.json').write_text(json.dumps([sample]))
            args = SimpleNamespace(dataset='COCONeg', subset_name='train', dataset_kwargs={
                'json_folder': str(root), 'image_root': str(root), 'num_entity_captions': 2,
                'structured_relation_prob': 1., 'swap_negative_prob': 1.})
            dataset = build_dataset_from_args(args, ToTensor())
            wrapped = get_dataset_embedding_class('COCONeg')(dataset, [0])
            loader, _ = create_distributed_dataloader(wrapped, 1, num_workers=0, pin_memory=False)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch['pos_tokens'].shape), (1, 3, 77))
            self.assertTrue(batch['caption_valid_mask'].all())
            self.assertFalse(torch.equal(batch['pos_tokens'], batch['neg_tokens']))

    def test_saved_last_best_and_full_config_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(1, 1, bias=False)
            model.weight.data.fill_(2.)
            cfg = OmegaConf.create({'training': {'epochs': 25}, 'loss': {'lambda_entities': .5}})
            args = SimpleNamespace(save_path=directory, exp_name='run')
            _save_run_artifacts(args, cfg, model, {'best_model_state_dict': {'weight': torch.ones(1, 1)}})
            out = Path(directory) / 'run'
            self.assertEqual(torch.load(out/'last_checkpoint.pt', weights_only=True)['weight'].item(), 2.)
            self.assertEqual(torch.load(out/'best_checkpoint.pt', weights_only=True)['weight'].item(), 1.)
            self.assertEqual(json.loads((out/'config.json').read_text()), OmegaConf.to_container(cfg))

    def test_ties_are_incorrect_and_cor_uses_same_denominator(self):
        scores = [{'triplet_type': 'entity', 's_anchor': .3, 's_half_truth': .3, 's_full_truth': .4},
                  {'triplet_type': 'relation', 's_anchor': .3, 's_half_truth': .2, 's_full_truth': .4}]
        result = summarize(scores)
        self.assertEqual(result['all']['ht_accuracy'], 50.)
        self.assertEqual(result['all']['cor_accuracy'], 50.)
        self.assertEqual(result['all']['n'], 2)

    def test_final_clip_checkpoints_load_directly_and_keep_best_distinct(self):
        def small_clip():
            return CLIP(embed_dim=8,
                        vision_cfg={'layers': 1, 'width': 64, 'patch_size': 16, 'image_size': 32},
                        text_cfg={'context_length': 8, 'vocab_size': 32, 'width': 64, 'heads': 1, 'layers': 1})

        with tempfile.TemporaryDirectory() as directory:
            backbone = small_clip()
            head = torch.nn.Module()
            head.register_parameter('logit_scale', torch.nn.Parameter(torch.tensor(1.)))
            model = CLIPEndToEndPipeline(backbone, head)
            best = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                backbone.visual.conv1.weight.add_(1.)
            cfg = OmegaConf.create({'alignment': {'align_image': False, 'align_text': False}})
            args = SimpleNamespace(save_path=directory, exp_name='run')
            _save_run_artifacts(args, cfg, torch.nn.DataParallel(model), {'best_model_state_dict': best})
            out = Path(directory) / 'run'
            for filename in ('last_checkpoint.pt', 'best_checkpoint.pt'):
                state = torch.load(out / filename, weights_only=True)
                restored = small_clip()
                restored.load_state_dict(state, strict=True)
                expected = backbone.state_dict() if filename.startswith('last') else {
                    k.removeprefix('model.'): v for k, v in best.items() if k.startswith('model.')}
                for key in state:
                    torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)
                evaluator = BatchCheckpointEvaluator.__new__(BatchCheckpointEvaluator)
                loaded = evaluator._load_ft_checkpoint(restored, state, cfg)
                self.assertIs(loaded, restored)
                self.assertFalse(loaded.training)

            cfg.alignment.align_text = True
            _save_run_artifacts(args, cfg, model, {'best_model_state_dict': best})
            wrapped = torch.load(out / 'last_checkpoint.pt', weights_only=True)
            self.assertIn('head.logit_scale', wrapped)
            self.assertIn('model.visual.conv1.weight', wrapped)
            with self.assertRaisesRegex(ValueError, 'alignment'):
                evaluator._load_ft_checkpoint(small_clip(), state, cfg)



if __name__ == '__main__':
    unittest.main()
