"""Training invariants across accumulation, DDP and interruption."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import torch
from model.runtime import train, load_model, read_json
from dataset.indexed import JsonlDataset


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "tokens.jsonl"
        rows = [{"input_ids": [1, 3, 4, 2]},
                {"input_ids": [1, 5, 6, 7, 8, 2], "labels": [-100, -100, -100, 7, 8, 2]},
                {"input_ids": [1, 9, 2]}]
        self.data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.cfg = dict(model=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, intermediate_size=64, vocab_size=64,
                                   max_position_embeddings=128, flash_attn=False),
                        data=str(self.data), steps=4, batch_size=2, device="cpu",
                        cpu_threads=1, seed=42, save_every=2, gradient_accumulation_steps=2)

    def tearDown(self):
        self.temp.cleanup()

    def recipe(self, name, **changes):
        config = dict(self.cfg, **changes)
        path = self.root / (name + ".json")
        path.write_text(json.dumps(config))
        return path

    def assert_weights(self, left, right, tolerance=0):
        a, b = load_model(left).state_dict(), load_model(right).state_dict()
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=tolerance, rtol=tolerance, msg=key)

    def run_ddp(self, recipe, output, resume=None):
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   "-m", "mica", "train", "--recipe", str(recipe), "--output", str(output)]
        if resume:
            command += ["--resume", str(resume)]
        subprocess.run(command, check=True, timeout=90, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=dict(os.environ, OMP_NUM_THREADS="1"))

    def test_token_weighted_accumulation(self):
        train(self.recipe("acc"), self.root / "acc")
        train(self.recipe("batch", batch_size=4, gradient_accumulation_steps=1), self.root / "batch")
        self.assert_weights(self.root / "acc", self.root / "batch", 2e-5)

    def test_ddp_matches_global_batch_and_resume(self):
        partial = self.recipe("ddp2", steps=2)
        full = self.recipe("ddp4")
        self.run_ddp(partial, self.root / "ddp2")
        self.run_ddp(full, self.root / "resumed", self.root / "ddp2")
        self.run_ddp(full, self.root / "ddp4")
        self.assert_weights(self.root / "resumed", self.root / "ddp4")
        train(self.recipe("single", batch_size=8, gradient_accumulation_steps=1), self.root / "single")
        self.assert_weights(self.root / "ddp4", self.root / "single", 2e-5)

    def test_signal_checkpoint_resumes_exactly(self):
        recipe = self.recipe("long", steps=10000)
        output = self.root / "interrupted"
        command = [sys.executable, "-m", "mica", "train", "--recipe", str(recipe), "--output", str(output)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 30
            while not (output / "latest.json").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError("worker did not reach a checkpoint")
                time.sleep(0.02)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 130, stderr.decode())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        run = read_json(output / "run.json")
        self.assertEqual(run["status"], "interrupted")
        final_steps = run["steps"] + 2
        continued = self.recipe("continued", steps=final_steps)
        train(continued, self.root / "continued", output)
        train(continued, self.root / "full")
        self.assert_weights(self.root / "continued", self.root / "full")

    def test_index_stores_offsets_not_tokens(self):
        dataset = JsonlDataset(self.data, 64, 128)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.offsets.itemsize, 8)
        self.assertEqual(dataset[2][0], [1, 9, 2])

    def test_failed_save_preserves_last_complete_checkpoint(self):
        from unittest.mock import patch
        original_save = torch.save
        calls = [0]

        def fail_third(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 3:
                raise OSError("simulated storage failure")
            return original_save(*args, **kwargs)

        output = self.root / "save-failure"
        with patch("trainer.common.engine.torch.save", side_effect=fail_third):
            with self.assertRaisesRegex(RuntimeError, "simulated storage failure"):
                train(self.recipe("failure"), output)
        self.assertEqual(read_json(output / "latest.json")["checkpoint"], "checkpoints/step-00000002")
        self.assertFalse((output / "checkpoints/step-00000004").exists())
        self.assertIsNotNone(load_model(output))

    def test_moe_ddp_trains_with_unused_experts(self):
        model = dict(self.cfg["model"], use_moe=True)
        recipe = self.recipe("moe", model=model, steps=2)
        self.run_ddp(recipe, self.root / "moe")
        metrics = [json.loads(line) for line in (self.root / "moe/metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics))


if __name__ == "__main__":
    unittest.main()
