"""Tests unitarios para src/data/s3_utils.py. No se conectan a AWS."""

import pytest

from src.data import s3_utils


class ClienteFalso:
    """Registra los pedidos en lugar de enviarlos a S3."""

    def __init__(self):
        self.subidas = []

    def upload_file(self, local, bucket, clave):
        self.subidas.append((bucket, clave))


@pytest.fixture
def cliente(monkeypatch):
    falso = ClienteFalso()
    monkeypatch.setattr(s3_utils, "get_s3_client", lambda region=None: falso)
    return falso


@pytest.mark.parametrize("prefijo", ["raw", "raw/"])
def test_upload_directory_conserva_las_carpetas(cliente, tmp_path, prefijo):
    for clase in ("glioma", "notumor"):
        (tmp_path / clase).mkdir()
        (tmp_path / clase / "a.jpg").write_bytes(b"x")

    assert s3_utils.upload_directory(tmp_path, "bucket", prefijo) == 2
    assert cliente.subidas == [("bucket", "raw/glioma/a.jpg"), ("bucket", "raw/notumor/a.jpg")]


def test_upload_directory_falla_si_no_existe_la_carpeta(cliente, tmp_path):
    with pytest.raises(NotADirectoryError):
        s3_utils.upload_directory(tmp_path / "no_existe", "bucket", "raw")
