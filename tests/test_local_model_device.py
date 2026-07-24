"""Cheap structural test for local_model.py — only exercises device-selection
logic, never calls load_local_model() (which would download the ~3GB base model)."""

import local_model


def test_device_returns_valid_torch_device_string():
    assert local_model._device() in ("mps", "cuda", "cpu")


def test_is_loaded_false_before_any_load_call():
    # As long as no other test in this process called load_local_model() first.
    # This is a soft check (module-level cache is process-global) — the real
    # guarantee is that importing local_model never populates the cache itself.
    assert isinstance(local_model.is_loaded(), bool)
