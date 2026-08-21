"""An upload's filename is client-supplied text that becomes a real path."""
import io
import types

import pytest

from src.unstructured.ingestion.service import IngestionManager

safe_name = IngestionManager._safe_upload_name


class _Upload:
    """Stands in for starlette's UploadFile: a name and a stream."""

    def __init__(self, filename, data=b"pdf bytes"):
        self.filename = filename
        self.file = io.BytesIO(data)


@pytest.mark.parametrize(
    "sent, expected",
    [
        ("report.pdf", "report.pdf"),
        # The folder picker sends webkitRelativePath, which is what broke
        # every upload made by choosing a directory.
        ("corpus10/irs_p501_dependents.pdf", "irs_p501_dependents.pdf"),
        ("a/b/c/deep.pdf", "deep.pdf"),
        # Windows clients send backslashes; PurePosixPath alone would keep them.
        (r"C:\Users\me\Desktop\report.pdf", "report.pdf"),
        (r"folder\sub\report.pdf", "report.pdf"),
        # Names that resolve to no name at all.
        ("", "upload"),
        (None, "upload"),
        (".", "upload"),
        ("..", "upload"),
        # A trailing slash yields the last component, which is still a safe
        # bare name -- the guarantee is "no directories, no climbing out",
        # not that a degenerate name is detected as one.
        ("some/dir/", "dir"),
    ],
)
def test_only_the_file_name_survives(sent, expected):
    assert safe_name(sent) == expected


@pytest.mark.parametrize(
    "attack",
    [
        "../../../etc/cron.d/job",
        "../escape.pdf",
        "..\\..\\windows\\system32\\evil.dll",
        "/etc/passwd",
        "/../../root/.ssh/authorized_keys",
    ],
)
def test_a_filename_cannot_climb_out_of_the_ingest_directory(attack, tmp_path):
    """`temp_dir / f"{job_id}_{filename}"` looks like string interpolation but
    builds a path: a `..` in the name walks up out of it."""
    service = types.SimpleNamespace(
        temp_dir=tmp_path / "tmp_ingest",
        _safe_upload_name=IngestionManager._safe_upload_name,
    )
    written = IngestionManager._save_upload(service, _Upload(attack), "job123")

    assert written.resolve().parent == (tmp_path / "tmp_ingest").resolve()
    assert ".." not in written.parts
    assert written.read_bytes() == b"pdf bytes"


def test_a_folder_upload_lands_in_the_ingest_directory(tmp_path):
    """The regression itself: the save path named a directory nothing created."""
    service = types.SimpleNamespace(
        temp_dir=tmp_path / "tmp_ingest",
        _safe_upload_name=IngestionManager._safe_upload_name,
    )
    written = IngestionManager._save_upload(
        service, _Upload("corpus10/nist_csf_2.pdf"), "abc123"
    )

    assert written.name == "abc123_nist_csf_2.pdf"
    assert written.exists()


def test_the_job_id_still_keeps_two_uploads_of_one_name_apart(tmp_path):
    """Two files called report.pdf from different folders must not collide."""
    service = types.SimpleNamespace(
        temp_dir=tmp_path / "tmp_ingest",
        _safe_upload_name=IngestionManager._safe_upload_name,
    )
    first = IngestionManager._save_upload(service, _Upload("q1/report.pdf", b"one"), "job1")
    second = IngestionManager._save_upload(service, _Upload("q2/report.pdf", b"two"), "job2")

    assert first != second
    assert first.read_bytes() == b"one" and second.read_bytes() == b"two"
