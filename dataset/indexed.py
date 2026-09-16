"""Seekable JSONL with compact offsets; tokens are decoded only on demand."""
from array import array
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_row(row, number, vocab_size, max_length):
    ids = row["input_ids"]
    labels = row.get("labels", ids)
    if len(ids) != len(labels) or not 2 <= len(ids) <= max_length:
        raise ValueError(f"invalid lengths at row {number}")
    if any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
        raise ValueError(f"invalid token at row {number}")
    if any(type(x) is not int or (x != -100 and not 0 <= x < vocab_size) for x in labels):
        raise ValueError(f"invalid label at row {number}")
    if all(x == -100 for x in labels[1:]):
        raise ValueError(f"no supervised next-token targets at row {number}")
    return ids, labels


class JsonlDataset:
    def __init__(self, path, vocab_size, max_length):
        self.path = Path(path)
        self.vocab_size, self.max_length = vocab_size, max_length
        self.offsets = array("Q")
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                digest.update(line)
                if line.strip():
                    validate_row(json.loads(line), len(self.offsets) + 1, vocab_size, max_length)
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("dataset is empty")
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        # A new handle avoids shared cursor state across workers.
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return validate_row(json.loads(handle.readline()), index + 1, self.vocab_size, self.max_length)

    def __iter__(self):
        with self.path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield validate_row(json.loads(line), number, self.vocab_size, self.max_length)
