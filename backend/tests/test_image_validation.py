from io import BytesIO

import pytest
from app.services.image_validation import read_validated_image
from fastapi import UploadFile
from PIL import Image


@pytest.mark.anyio
async def test_valid_image_is_read():
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    upload = UploadFile(
        filename="test.png",
        file=BytesIO(buffer.getvalue()),
        headers={"content-type": "image/png"},
    )
    data, width, height = await read_validated_image(upload)
    assert data and (width, height) == (64, 64)


@pytest.mark.anyio
async def test_corrupt_image_is_rejected():
    upload = UploadFile(
        filename="bad.png",
        file=BytesIO(b"not an image"),
        headers={"content-type": "image/png"},
    )
    with pytest.raises(Exception) as error:
        await read_validated_image(upload)
    assert getattr(error.value, "status_code", None) == 400
