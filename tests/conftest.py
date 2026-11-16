"""
Shared fixtures for the test suite.

Place this file at:  tests/conftest.py
Run tests from the repository root with:  pytest tests/ -v
"""

import io
import os
import sys

import pytest
import torch
from PIL import Image

# Make app.py importable when pytest runs from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402  (app.py at the repository root)


@pytest.fixture
def client():
    """Flask test client. Lets us call the routes without starting a server."""
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def valid_jpeg_bytes():
    """A real, openable JPEG held in memory."""
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), color=(120, 80, 200)).save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def corrupt_jpeg_bytes():
    """A text file wearing a .jpg extension. Pillow cannot open this."""
    return io.BytesIO(b"this is plain text, not image data at all")


@pytest.fixture
def pdf_bytes():
    """Something that is clearly not an image."""
    return io.BytesIO(b"%PDF-1.4\n% fake pdf for testing\n")


class SpyModel:
    """
    Stand-in for the trained CNN.

    Counts its calls so a test can assert the model was never reached, and
    returns a tensor shaped like the real model's output so the route's
    softmax / argmax code runs unchanged.
    """

    def __init__(self, raises=False):
        self.calls = 0
        self.raises = raises

    def __call__(self, tensor):
        self.calls += 1
        if self.raises:
            raise RuntimeError("simulated model failure")
        return torch.tensor([[0.1, 2.5]])  # shape (1, 2), as ImageCNN returns


@pytest.fixture
def spy_model():
    return SpyModel()


@pytest.fixture
def failing_model():
    return SpyModel(raises=True)
