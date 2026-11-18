"""Offline regression tests for Kaggle auto-download certificate recovery."""
from __future__ import annotations

from pathlib import Path
import requests
import pytest

from scripts.prepare_kvasir_seg import _maybe_download


class _Reply:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"PK\x03\x04download-fixture"


def test_auto_simula_ssl_error_falls_back_once(tmp_path, monkeypatch):
    monkeypatch.setenv('BT_AUTO_DATASET_TLS_FALLBACK', '1')
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs['verify'])
        if kwargs['verify']:
            raise requests.exceptions.SSLError('certificate verify failed')
        return _Reply()

    monkeypatch.setattr(requests, 'get', fake_get)
    dst = tmp_path / 'kvasir.zip'
    assert _maybe_download('https://datasets.simula.no/downloads/kvasir-seg.zip', dst) == dst
    assert dst.read_bytes().startswith(b'PK')
    assert calls == [True, False]


def test_no_fallback_without_auto_flag(tmp_path, monkeypatch):
    monkeypatch.delenv('BT_AUTO_DATASET_TLS_FALLBACK', raising=False)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs['verify'])
        raise requests.exceptions.SSLError('certificate verify failed')

    monkeypatch.setattr(requests, 'get', fake_get)
    with pytest.raises(RuntimeError, match='Failed to download'):
        _maybe_download('https://datasets.simula.no/downloads/kvasir-seg.zip', tmp_path/'x.zip')
    assert calls == [True, True, True]


def test_no_fallback_for_untrusted_domain(tmp_path, monkeypatch):
    monkeypatch.setenv('BT_AUTO_DATASET_TLS_FALLBACK', '1')
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs['verify'])
        raise requests.exceptions.SSLError('certificate verify failed')

    monkeypatch.setattr(requests, 'get', fake_get)
    with pytest.raises(RuntimeError, match='Failed to download'):
        _maybe_download('https://example.com/data.zip', tmp_path/'x.zip')
    assert calls == [True, True, True]


def test_non_ssl_failure_does_not_disable_cert_validation(tmp_path, monkeypatch):
    monkeypatch.setenv('BT_AUTO_DATASET_TLS_FALLBACK', '1')
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs['verify'])
        raise requests.exceptions.HTTPError('404')

    monkeypatch.setattr(requests, 'get', fake_get)
    with pytest.raises(RuntimeError, match='Failed to download'):
        _maybe_download('https://datasets.simula.no/downloads/kvasir-seg.zip', tmp_path/'x.zip')
    assert calls == [True, True, True]
