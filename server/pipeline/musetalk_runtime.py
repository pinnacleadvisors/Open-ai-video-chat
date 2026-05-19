"""Runtime adapter for MuseTalk (https://github.com/TMElyralab/MuseTalk).

The upstream repo is cloned into `third_party/MuseTalk` by `scripts/setup.sh`,
which also downloads the model checkpoints into `models/checkpoints/musetalk/`.
This module imports the upstream pipeline and exposes a single
`MuseTalkRuntime.infer(audio, ref)` entrypoint.

Implemented as a separate file so that `avatar.py` stays importable on
machines that don't have MuseTalk installed (Wav2Lip-only fallback).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger


THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "MuseTalk"
if THIRD_PARTY.exists() and str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))


class MuseTalkRuntime:
    def __init__(self, checkpoint_dir: Path, device: str, fps: int, batch: int):
        if not THIRD_PARTY.exists():
            raise RuntimeError(
                f"MuseTalk source missing at {THIRD_PARTY}. Run scripts/setup.sh."
            )
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.whisper.audio2feature import Audio2Feature

        self.device = device
        self.fps = fps
        self.batch = batch
        self._get_bbox = get_landmark_and_bbox

        logger.info("loading musetalk vae + unet + pe")
        self.audio_processor = Audio2Feature(
            model_path=str(checkpoint_dir / "whisper" / "tiny.pt")
        )
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=str(checkpoint_dir / "musetalkV15" / "unet.pth"),
            vae_type="sd-vae",
            unet_config=str(checkpoint_dir / "musetalkV15" / "musetalk.json"),
            device=device,
        )
        self.unet = self.unet.half() if device == "cuda" else self.unet
        self.timesteps = torch.tensor([0], device=device)

    @torch.no_grad()
    def prepare_reference(self, image_bgr: np.ndarray) -> dict:
        from musetalk.utils.preprocessing import coord_placeholder
        from musetalk.utils.blending import get_image_prepare_material

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bbox, _ = self._get_bbox([rgb])
        if bbox[0] == coord_placeholder:
            raise ValueError("MuseTalk could not detect a face in the reference image")
        x1, y1, x2, y2 = bbox[0]
        face = rgb[y1:y2, x1:x2]
        face_256 = cv2.resize(face, (256, 256))
        latent = self.vae.get_latents_for_unet(face_256)
        mask, mask_box = get_image_prepare_material(rgb, bbox[0])
        return {
            "full_frame": rgb,
            "bbox": (x1, y1, x2, y2),
            "face_latent": latent,
            "mask": mask,
            "mask_box": mask_box,
        }

    @torch.no_grad()
    def infer(self, audio16: np.ndarray, ref: dict) -> list[np.ndarray]:
        from musetalk.utils.blending import get_image

        feats = self.audio_processor.audio2feat(audio16)
        chunks = self.audio_processor.feature2chunks(feature_array=feats, fps=self.fps)

        latents = ref["face_latent"].to(self.device)
        if self.device == "cuda":
            latents = latents.half()

        out: list[np.ndarray] = []
        full = ref["full_frame"]
        for i in range(0, len(chunks), self.batch):
            audio_batch = torch.from_numpy(np.stack(chunks[i:i + self.batch])).to(self.device)
            if self.device == "cuda":
                audio_batch = audio_batch.half()
            audio_emb = self.pe(audio_batch)
            lat_batch = latents.repeat(audio_emb.shape[0], 1, 1, 1)
            pred = self.unet.model(lat_batch, self.timesteps, encoder_hidden_states=audio_emb).sample
            decoded = self.vae.decode_latents(pred)
            for face in decoded:
                composed = get_image(full, face, ref["bbox"], ref["mask"], ref["mask_box"])
                out.append(cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
        return out
