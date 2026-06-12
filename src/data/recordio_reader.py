"""
Pure-Python reader for MXNet Indexed RecordIO (.idx + .rec).

Avoids the mxnet package, which breaks on NumPy 2.x / Python 3.12.
Format reference: Apache MXNet python/mxnet/recordio.py and dmlc-core recordio.
"""

from __future__ import annotations

import struct
from collections import namedtuple
from io import BytesIO
from pathlib import Path

from PIL import Image

MAGIC = 0xCED7230A
_IR_FORMAT = "<IfQQ"
_IR_SIZE = struct.calcsize(_IR_FORMAT)
IRHeader = namedtuple("IRHeader", ["flag", "label", "id", "id2"])


def load_record_index(idx_path: str | Path) -> dict[int, int]:
    """
    Parse train.idx text file: one 'record_key<TAB>byte_offset' entry per line.
    """
    index: dict[int, int] = {}
    with open(idx_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key_str, offset_str = line.split("\t", 1)
            index[int(key_str)] = int(offset_str)
    return index


def _read_record_body(rec_file, offset: int) -> bytes:
    """Read one RecordIO payload at a byte offset inside train.rec."""
    rec_file.seek(offset)
    header = rec_file.read(8)
    if len(header) != 8:
        raise ValueError(f"Unexpected EOF at offset {offset}")

    magic, lrec = struct.unpack("<II", header)
    if magic != MAGIC:
        raise ValueError(f"Invalid RecordIO magic at offset {offset}: {magic:#x}")

    length = lrec & ((1 << 29) - 1)
    body = rec_file.read(length)
    if len(body) != length:
        raise ValueError(f"Short read at offset {offset}: expected {length} bytes")

    pad = ((length + 3) // 4) * 4 - length
    if pad:
        rec_file.read(pad)
    return body


def unpack_record(body: bytes) -> tuple[IRHeader, bytes]:
    """Split IRHeader metadata from JPEG/raw image bytes."""
    header = IRHeader(*struct.unpack(_IR_FORMAT, body[:_IR_SIZE]))
    content = body[_IR_SIZE:]
    if header.flag > 0:
        import numpy as np

        label = np.frombuffer(content, dtype=np.float32, count=header.flag)
        content = content[header.flag * 4 :]
        header = header._replace(label=label)
    return header, content


def decode_image_bytes(content: bytes) -> Image.Image:
    """Decode packed image bytes to RGB PIL Image."""
    try:
        return Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        pass

    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as exc:
        raise ValueError("Could not decode image bytes") from exc


class MXIndexedRecordReader:
    """Minimal random-access reader for MXNet Indexed RecordIO files."""

    def __init__(self, idx_path: str | Path, rec_path: str | Path):
        self.index = load_record_index(idx_path)
        self.rec_path = Path(rec_path)
        self._rec_file = open(self.rec_path, "rb")

    def close(self):
        self._rec_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def read_idx(self, key: int) -> bytes:
        if key not in self.index:
            raise KeyError(f"Record key {key} not found in index")
        body = _read_record_body(self._rec_file, self.index[key])
        _header, content = unpack_record(body)
        return content

    def read_image(self, key: int) -> Image.Image:
        return decode_image_bytes(self.read_idx(key))
