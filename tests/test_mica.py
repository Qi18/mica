import json
import tempfile
import unittest
from pathlib import Path

import torch
from mica import MicaConfig, MicaForCausalLM
from model.runtime import train, load_model, load_data, import_legacy, evaluate

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "tests/fixtures/model-v0.2-logits.json").read_text())


class MicaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, intermediate_size=64, vocab_size=64,
                           max_position_embeddings=128, flash_attn=False)
        self.data = self.root / "tokens.jsonl"
        self.data.write_text('{"input_ids":[1,3,4,5,2]}\n{"input_ids":[1,9,8,2]}\n')

    def tearDown(self):
        self.temp.cleanup()

    def recipe(self, name, steps):
        path = self.root / name
        path.write_text(json.dumps(dict(model=self.config, data=str(self.data), steps=steps,
                                       batch_size=2, seed=42, device="cpu")))
        return path

    def test_dense_and_moe_match_frozen_baseline(self):
        for moe in (False, True):
            cfg = dict(self.config, use_moe=moe)
            torch.manual_seed(42)
            new = MicaForCausalLM(MicaConfig(**cfg)).eval()
            ids = torch.tensor([[1, 5, 6, 2]])
            with torch.no_grad():
                torch.testing.assert_close(torch.tensor(BASELINE[str(moe)]), new(ids).logits, rtol=1e-5, atol=1e-6)

    def test_resume_matches_uninterrupted_training(self):
        a, b, c = [self.root / x for x in ("partial", "resumed", "full")]
        train(self.recipe("two.json", 2), a)
        train(self.recipe("four.json", 4), b, resume=a)
        train(self.root / "four.json", c)
        for key, value in load_model(b).state_dict().items():
            torch.testing.assert_close(value, load_model(c).state_dict()[key], rtol=0, atol=0)
        self.assertTrue(evaluate(c, self.data)["nll"] > 0)

    def test_empty_supervision_is_rejected(self):
        self.data.write_text('{"input_ids":[1,2],"labels":[-100,-100]}\n')
        with self.assertRaisesRegex(ValueError, "supervised"):
            load_data(self.data, 64, 128)

    def test_import_and_cache_equivalence(self):
        old = MicaForCausalLM(MicaConfig(**self.config)).eval()
        checkpoint = self.root / "legacy.pth"
        torch.save(old.state_dict(), checkpoint)
        config = self.root / "config.json"
        config.write_text(json.dumps(self.config))
        import_legacy(checkpoint, config, self.root / "imported")
        model = load_model(self.root / "imported").eval()
        self.assertEqual(model.config.model_type, "mica")
        ids = torch.tensor([[1, 3, 6, 7]])
        with torch.no_grad():
            prefix = model(ids[:, :-1], use_cache=True)
            cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits
            torch.testing.assert_close(cached, model(ids).logits[:, -1:], rtol=1e-5, atol=1e-6)

    def test_resume_rejects_changed_data(self):
        train(self.recipe("two.json", 2), self.root / "partial")
        self.data.write_text('{"input_ids":[1,4,6,2]}\n')
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            train(self.recipe("four.json", 4), self.root / "resumed", self.root / "partial")

    def test_existing_output_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "keep").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            train(self.recipe("two.json", 2), output)
        self.assertEqual((output / "keep").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
