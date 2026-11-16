"""
Integration Test: Error Handling

Place this file at:  tests/test_integration_errors.py
Run with:            pytest tests/test_integration_errors.py -v

These tests drive the real /upload-photo and /upload-video routes through
Flask's test client. Only the trained CNN is replaced, so the tests run
without the .pth weight files.

Unlike the Photo Upload unit tests, which check that bad input produces SOME
error, these check that the error is the RIGHT one, that bad input is stopped
before it reaches the model, and that temporary files are cleaned up even when
inference fails.
"""

import io

import pytest

import app as app_module


# ===================================================================== E1
# Input rejected before any processing begins
class TestRequestLevelRejection:
    """
    The two checks that happen before the try block: an empty request and
    more than one file. These guard the single-upload requirement (UR-11).
    """

    def test_request_with_no_file_returns_400(self, client):
        response = client.post("/upload-photo", data={}, content_type="multipart/form-data")
        assert response.status_code == 400
        assert b"No selected file" in response.data

    def test_empty_filename_returns_400(self, client):
        response = client.post(
            "/upload-photo",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_multiple_files_rejected(self, client, valid_jpeg_bytes):
        second = io.BytesIO(valid_jpeg_bytes.getvalue())
        response = client.post(
            "/upload-photo",
            data={"file": [(valid_jpeg_bytes, "a.jpg"), (second, "b.jpg")]},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        assert b"one file at a time" in response.data


# ===================================================================== E2
# Invalid content must not reach the model
class TestInvalidContentStopsBeforeInference:
    """
    This is the part the unit tests cannot check. A file that Pillow cannot
    open must be rejected BEFORE the CNN is called -- a prediction made from
    garbage input is worse than an error message, because the user is shown a
    verdict that means nothing.
    """

    def test_corrupt_image_never_reaches_the_model(self, client, corrupt_jpeg_bytes, spy_model, monkeypatch):
        monkeypatch.setattr(app_module, "photo_model", spy_model)

        client.post(
            "/upload-photo",
            data={"file": (corrupt_jpeg_bytes, "corrupt.jpg")},
            content_type="multipart/form-data",
        )

        assert spy_model.calls == 0, "corrupt input was passed to the CNN"

    def test_pdf_never_reaches_the_model(self, client, pdf_bytes, spy_model, monkeypatch):
        monkeypatch.setattr(app_module, "photo_model", spy_model)

        client.post(
            "/upload-photo",
            data={"file": (pdf_bytes, "document.pdf")},
            content_type="multipart/form-data",
        )

        assert spy_model.calls == 0


# ===================================================================== E3
# Correct status code for the kind of failure
class TestStatusCodesMatchTheFailure:
    """
    HTTP 4xx means the client sent something wrong; HTTP 5xx means the server
    itself failed. Returning 500 for a bad upload tells the user the site is
    broken when in fact their file was the problem.

    NOTE: these two tests FAIL against the current app.py, which returns 500
    for unreadable files. See the fix in the report.
    """

    def test_corrupt_image_returns_400_not_500(self, client, corrupt_jpeg_bytes):
        response = client.post(
            "/upload-photo",
            data={"file": (corrupt_jpeg_bytes, "corrupt.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_pdf_returns_400_not_500(self, client, pdf_bytes):
        response = client.post(
            "/upload-photo",
            data={"file": (pdf_bytes, "document.pdf")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_error_message_is_readable_and_leaks_no_internals(self, client, corrupt_jpeg_bytes):
        """
        Our users have limited technical experience. A raw exception string
        tells them nothing and exposes internal detail.
        """
        response = client.post(
            "/upload-photo",
            data={"file": (corrupt_jpeg_bytes, "corrupt.jpg")},
            content_type="multipart/form-data",
        )
        body = response.get_data(as_text=True).lower()
        assert "traceback" not in body
        assert "cannot identify image file" not in body  # raw Pillow message
        assert "_io.bytesio" not in body


# ===================================================================== E4
# A genuine server-side failure is handled, not crashed
class TestModelFailureIsHandled:
    """
    If the CNN itself raises -- missing weights, out of memory -- the app must
    return an error response rather than dying. Here 500 IS the correct code,
    because the fault is on our side, not the user's.
    """

    def test_model_exception_returns_500_without_crashing(self, client, valid_jpeg_bytes, failing_model, monkeypatch):
        monkeypatch.setattr(app_module, "photo_model", failing_model)

        response = client.post(
            "/upload-photo",
            data={"file": (valid_jpeg_bytes, "holiday.jpg")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 500
        assert failing_model.calls == 1, "the model should have been reached"

    def test_app_still_works_after_a_failed_request(self, client, valid_jpeg_bytes, spy_model, failing_model, monkeypatch):
        """One bad request must not leave the app in a broken state."""
        monkeypatch.setattr(app_module, "photo_model", failing_model)
        client.post(
            "/upload-photo",
            data={"file": (io.BytesIO(valid_jpeg_bytes.getvalue()), "x.jpg")},
            content_type="multipart/form-data",
        )

        monkeypatch.setattr(app_module, "photo_model", spy_model)
        response = client.post(
            "/upload-photo",
            data={"file": (io.BytesIO(valid_jpeg_bytes.getvalue()), "y.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200


# ===================================================================== E5
# Temporary files are removed even on failure
class TestTemporaryVideoFileIsDeleted:
    """
    The video route writes the upload to a temporary file so OpenCV can read
    it. The SRS promises uploads are never retained, so that file must be
    removed on every path, including when processing fails.

    The photo route never writes to disk at all -- it reads the upload into
    memory -- so there is nothing to clean up there.
    """

    def test_temp_file_removed_after_invalid_video(self, client, monkeypatch, tmp_path):
        created = []

        real_mkstemp = app_module.tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            created.append(path)
            return fd, path

        monkeypatch.setattr(app_module.tempfile, "mkstemp", tracking_mkstemp)

        client.post(
            "/upload-video",
            data={"file": (io.BytesIO(b"not a real video"), "broken.mp4")},
            content_type="multipart/form-data",
        )

        assert created, "no temporary file was created, test needs updating"
        for path in created:
            assert not app_module.os.path.exists(path), f"temp file left behind: {path}"
