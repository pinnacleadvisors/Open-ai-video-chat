from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

# (magic_bytes, extension). Matches PNG, JPEG, GIF, WEBP, WAV, MP3, OGG, FLAC, MP4.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"RIFF", ".webp_or_wav"),     # disambiguated below
    (b"OggS", ".ogg"),
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"\xff\xf3", ".mp3"),
    (b"\xff\xf2", ".mp3"),
    (b"fLaC", ".flac"),
)

IMAGE_EXTS = {".png", ".jpg", ".gif", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def sniff_extension(head: bytes) -> str | None:
    for sig, ext in _MAGIC_SIGNATURES:
        if head.startswith(sig):
            if ext == ".webp_or_wav":
                # RIFF: 4 bytes magic, 4 bytes size, 4 bytes form-type.
                form = head[8:12] if len(head) >= 12 else b""
                if form == b"WEBP":
                    return ".webp"
                if form == b"WAVE":
                    return ".wav"
                return None
            return ext
    # MP4 container (ftyp at offset 4)
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return ".m4a"
    return None


def save_upload(
    file: UploadFile,
    dest_dir: Path,
    basename: str,
    *,
    kind: str,
    max_bytes: int,
) -> Path:
    """Save an upload to disk, validating size + magic bytes.

    `kind` is 'image' or 'audio'. Raises HTTPException on rejection.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    head = file.file.read(64)
    ext = sniff_extension(head)
    expected = IMAGE_EXTS if kind == "image" else AUDIO_EXTS
    if ext not in expected:
        raise HTTPException(415, f"unsupported {kind} format")

    out_path = dest_dir / f"{basename}{ext}"
    written = 0
    with out_path.open("wb") as out:
        out.write(head)
        written += len(head)
        while True:
            chunk = file.file.read(1 << 16)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out.close()
                out_path.unlink(missing_ok=True)
                raise HTTPException(413, f"{kind} too large (max {max_bytes // (1 << 20)} MB)")
            out.write(chunk)
    return out_path
