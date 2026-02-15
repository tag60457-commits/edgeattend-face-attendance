from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class FaceRecord:
    name: str
    embedding: np.ndarray


class FaceDB:
    """Very small on-disk face embedding DB.

    Stores one or more embeddings per person as .npy arrays:
      data/known_faces/<name>/<ts>.npy
    """

    def __init__(self, root: str = "data/known_faces"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def add(self, name: str, emb: np.ndarray, *, tag: str = "embed") -> str:
        name_dir = os.path.join(self.root, name)
        os.makedirs(name_dir, exist_ok=True)
        path = os.path.join(name_dir, f"{tag}.npy")
        np.save(path, emb.astype(np.float32))
        return path

    def load_all(self) -> List[FaceRecord]:
        out: List[FaceRecord] = []
        for name in sorted(os.listdir(self.root)):
            name_dir = os.path.join(self.root, name)
            if not os.path.isdir(name_dir):
                continue
            for f in os.listdir(name_dir):
                if not f.endswith(".npy"):
                    continue
                emb = np.load(os.path.join(name_dir, f)).astype(np.float32)
                out.append(FaceRecord(name=name, embedding=emb))
        return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / (na * nb))
