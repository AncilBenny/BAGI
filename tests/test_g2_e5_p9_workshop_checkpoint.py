"""Synthetic-only P9 checkpoint safety tests; zero native actions or fits."""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest
import torch

from bagi.agents.dqn import QNetwork
from bagi.experiments import g2_e5_p9_workshop_checkpoint as p9


def fake_source(*, poison_eval: bool = False) -> bytes:
    rng = np.random.default_rng(12)
    arrays = {name: rng.normal(size=shape).astype(np.float32)
              for name, shape in p9.WEIGHTS.items()}
    for name in p9.SOURCE_FIELDS:
        arrays[name] = np.asarray([99], dtype=np.int8)
    if poison_eval:
        arrays["seed17_net_0_weight"][0, 0] = np.nan
    numeric = io.BytesIO()
    np.savez_compressed(numeric, **arrays)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("summary.json", b"never loaded")
        z.writestr("public_policy_numeric.npz", numeric.getvalue())
    return outer.getvalue()


def test_frozen_source_and_tensor_contract():
    assert p9.SOURCE_SHA256 == "b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a"
    assert len(p9.WEIGHTS) == 12
    assert p9.WEIGHTS["seed29_net_4_weight"] == (4, 128)
    assert p9.SOURCE_FIELDS.isdisjoint(p9.WEIGHTS)


def test_safe_extraction_is_deterministic_and_contains_no_eval(tmp_path):
    source = tmp_path / "mixed.zip"
    source.write_bytes(fake_source())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    first = p9.extract_checkpoint(source, a, expected_source_sha256=sha,
                                  expected_checkpoint_sha256=None)
    p9.extract_checkpoint(source, b, expected_source_sha256=sha,
                          expected_checkpoint_sha256=None)
    assert a.read_bytes() == b.read_bytes()
    assert first["source_eval_arrays_materialized"] == 0
    with zipfile.ZipFile(a) as z:
        assert len(z.namelist()) == 13
        assert all("observation" not in name and "action" not in name
                   for name in z.namelist())
    loaded = p9.load_checkpoint(a, expected_sha256=first["checkpoint_sha256"],
                                expected_source_sha256=sha)
    assert set(loaded) == set(p9.WEIGHTS)
    assert all(v.dtype == np.float32 for v in loaded.values())


def test_wrong_source_hash_rejected_before_npz_access(tmp_path):
    source, output = tmp_path / "mixed.zip", tmp_path / "weights.zip"
    source.write_bytes(fake_source())
    with pytest.raises(ValueError, match="source ZIP SHA256"):
        p9.extract_checkpoint(source, output)
    assert not output.exists()


def test_corrupt_or_unexpected_tensors_rejected(tmp_path):
    source, output = tmp_path / "mixed.zip", tmp_path / "weights.zip"
    source.write_bytes(fake_source(poison_eval=True))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Invalid E2 frozen DQN tensor"):
        p9.extract_checkpoint(source, output, expected_source_sha256=sha,
                              expected_checkpoint_sha256=None)
    assert not output.exists()


def test_checkpoint_hash_and_extra_member_rejected(tmp_path):
    source, output = tmp_path / "mixed.zip", tmp_path / "weights.zip"
    source.write_bytes(fake_source())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    record = p9.extract_checkpoint(source, output, expected_source_sha256=sha,
                                   expected_checkpoint_sha256=None)
    with pytest.raises(ValueError, match="checkpoint ZIP SHA256"):
        p9.load_checkpoint(output, expected_sha256="0" * 64)
    with zipfile.ZipFile(output, "a") as z:
        z.writestr("observation.npy", b"FORBIDDEN")
    with pytest.raises(ValueError, match="Unexpected checkpoint member"):
        p9.load_checkpoint(output, expected_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                           expected_source_sha256=sha)
    assert record["frozen_tensors"] == 12


def test_tensor_names_reconstruct_original_qnetwork_exactly():
    model = QNetwork((5, 5), 4)
    expected = set(model.state_dict())
    names = {name.replace("_", ".") for name in p9.SHAPES}
    assert names == expected
    for name, shape in p9.SHAPES.items():
        assert tuple(model.state_dict()[name.replace("_", ".")].shape) == shape
    with torch.no_grad():
        q = model(torch.zeros(1, 5, 5))
    assert q.shape == (1, 4)


def test_source_eval_fields_are_never_materialized(monkeypatch, tmp_path):
    source, output = tmp_path / "mixed.zip", tmp_path / "weights.zip"
    source.write_bytes(fake_source())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    original_getitem = np.lib.npyio.NpzFile.__getitem__

    def forbid_eval(self, key):
        if key in p9.SOURCE_FIELDS:
            raise AssertionError(f"Source EVAL-bearing array materialized: {key}")
        return original_getitem(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbid_eval)
    p9.extract_checkpoint(source, output, expected_source_sha256=sha,
                          expected_checkpoint_sha256=None)
    assert output.exists()


def test_frozen_actor_reuses_original_dqn_without_training(monkeypatch, tmp_path):
    source, output = tmp_path / "mixed.zip", tmp_path / "weights.zip"
    source.write_bytes(fake_source())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    record = p9.extract_checkpoint(source, output, expected_source_sha256=sha,
                                   expected_checkpoint_sha256=None)
    weights = p9.load_checkpoint(output, expected_sha256=record["checkpoint_sha256"],
                                 expected_source_sha256=sha)
    monkeypatch.setattr(p9, "load_checkpoint", lambda path: weights)
    model = p9.frozen_q_network(output, 17)
    expected = QNetwork((5, 5), 4)
    expected.load_state_dict({name.replace("_", "."): torch.from_numpy(weights[f"seed17_{name}"])
                              for name in p9.SHAPES})
    views = torch.zeros((3, 5, 5), dtype=torch.int8)
    with torch.no_grad():
        assert torch.equal(model(views), expected(views))
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(ValueError, match="Unregistered"):
        p9.frozen_q_network(output, 999)
