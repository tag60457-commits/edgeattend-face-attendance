from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

from .face_db import FaceDB, cosine_sim


@dataclass
class Match:
    name: str
    score: float


class FaceRecognizer:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=self.device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.db = FaceDB()

    @torch.inference_mode()
    def embed_bgr(self, frame_bgr) -> Optional[np.ndarray]:
        # frame_bgr is OpenCV BGR ndarray
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face = self.mtcnn(rgb)
        if face is None:
            return None
        if isinstance(face, torch.Tensor):
            face = face.unsqueeze(0)  # [1,3,160,160]
        emb = self.model(face.to(self.device))
        emb = emb[0].detach().cpu().numpy().astype(np.float32)
        return emb

    def best_match(self, emb: np.ndarray, *, threshold: float = 0.6) -> Optional[Match]:
        records = self.db.load_all()
        best: Optional[Match] = None
        for r in records:
            s = cosine_sim(emb, r.embedding)
            if best is None or s > best.score:
                best = Match(name=r.name, score=s)
        if best is None:
            return None
        if best.score < threshold:
            return None
        return best
