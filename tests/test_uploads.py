import io

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from server.utils.uploads import save_upload, sniff_extension


def _upload(data: bytes, filename: str = "x.bin", content_type: str = "application/octet-stream") -> UploadFile:
    file = io.BytesIO(data)
    headers = Headers({"content-type": content_type})
    return UploadFile(file=file, filename=filename, headers=headers)


def test_sniff_png():
    assert sniff_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) == ".png"


def test_sniff_jpeg():
    assert sniff_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 8) == ".jpg"


def test_sniff_webp():
    # RIFF + size + WEBP
    assert sniff_extension(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4) == ".webp"


def test_sniff_wav():
    assert sniff_extension(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 4) == ".wav"


def test_sniff_unknown_returns_none():
    assert sniff_extension(b"\x00\x01\x02\x03random") is None


def test_save_upload_rejects_unknown_format(tmp_path):
    file = _upload(b"\x00\x01\x02\x03not-an-image")
    with pytest.raises(HTTPException) as ei:
        save_upload(file, tmp_path, "x", kind="image", max_bytes=1024)
    assert ei.value.status_code == 415


def test_save_upload_writes_correct_extension(tmp_path):
    png_head = b"\x89PNG\r\n\x1a\n"
    file = _upload(png_head + b"junk")
    path = save_upload(file, tmp_path, "img", kind="image", max_bytes=1024)
    assert path.suffix == ".png"
    assert path.read_bytes() == png_head + b"junk"


def test_save_upload_enforces_size_limit(tmp_path):
    png_head = b"\x89PNG\r\n\x1a\n"
    body = png_head + (b"x" * 10_000)
    file = _upload(body)
    with pytest.raises(HTTPException) as ei:
        save_upload(file, tmp_path, "img", kind="image", max_bytes=128)
    assert ei.value.status_code == 413
