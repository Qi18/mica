import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dataset.prepare import prepare


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3] * len(text)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 4, 5]


class DataTests(unittest.TestCase):
    def test_streamed_sft_mask_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.jsonl", root / "tokens.jsonl"
            source.write_text(json.dumps({"prompt": "hi", "response": "ok"}) + "\n")
            with patch("dataset.prepare.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                prepare(source, "fixture", output)
                row = json.loads(output.read_text())
                self.assertEqual(row["labels"], [-100, -100, -100, 3, 3, 2])
                source.write_text('{"text":"ok"}\n{"prompt":"hi","response":"ok"}\n')
                with self.assertRaisesRegex(ValueError, "supervised"):
                    prepare(source, "fixture", root / "invalid.jsonl", max_length=2)
                self.assertFalse((root / "invalid.jsonl").exists())
                self.assertEqual(list(root.glob(".tokens-*")), [])
                with self.assertRaises(FileExistsError):
                    prepare(source, "fixture", output)
                self.assertEqual(json.loads(output.read_text()), row)
