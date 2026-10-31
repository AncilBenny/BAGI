"""Synthetic hash and schema safety tests; zero native or expert calls."""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest

from bagi.experiments.g2_e5_p9_xland_source_preflight import (
    checked_npz,
    record_digest,
    verified_members,
)


def _npz(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _outer(tmp_path, values):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in values:
            archive.writestr(key, value)
    raw = path.read_bytes()
    return path, hashlib.sha256(raw).hexdigest()


def test_reject_tampered_outer_before_unpacking(tmp_path):
    path, digest = _outer(tmp_path, [("verified.npz", b"binary")])
    with pytest.raises(ValueError, match="outer"):
        verified_members(path, "0" * 64, {"verified.npz": hashlib.sha256(b"binary").hexdigest()})
    assert verified_members(path, digest, {"verified.npz": hashlib.sha256(b"binary").hexdigest()}) == {"verified.npz": b"binary"}


def test_reject_extra_unreviewed_and_duplicate_members(tmp_path):
    path, digest = _outer(tmp_path, [("ok.npz", b"a"), ("eval.npz", b"z")])
    with pytest.raises(ValueError, match="archive members"):
        verified_members(path, digest, {"ok.npz": hashlib.sha256(b"a").hexdigest()})
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate, d_hash = _outer(tmp_path, [("ok.npz", b"a"), ("ok.npz", b"a")])
    with pytest.raises(ValueError, match="duplicate"):
        verified_members(duplicate, d_hash, {"ok.npz": hashlib.sha256(b"a").hexdigest()})


def test_reject_valid_outer_but_tampered_inner(tmp_path):
    path, digest = _outer(tmp_path, [("a.npz", b"altered")])
    with pytest.raises(ValueError, match="member"):
        verified_members(path, digest, {"a.npz": hashlib.sha256(b"original").hexdigest()})


def test_refuse_eval_fields_and_object_pickle(tmp_path):
    good = _npz(obs=np.zeros((2, 3), dtype=np.float32))
    schema = {"obs": ((2, 3), "float32")}
    np.testing.assert_array_equal(checked_npz(good, schema)["obs"], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="numeric fields"):
        checked_npz(_npz(obs=np.zeros((2, 3)), eval_observation=np.ones((2, 3))), schema)
    with pytest.raises(ValueError):
        checked_npz(_npz(obs=np.asarray([{"private": 1}], dtype=object)), schema)


def test_reject_wrong_shapes_and_nonnumeric_nan():
    with pytest.raises(ValueError, match="Invalid"):
        checked_npz(_npz(obs=np.zeros((3, 3), dtype=np.float32)),
                    {"obs": ((2, 3), "float32")})
    with pytest.raises(ValueError, match="Invalid"):
        checked_npz(_npz(obs=np.asarray([[float("nan")]], dtype=np.float32)),
                    {"obs": ((1, 1), "float32")})


def test_record_digest_order_independent_but_data_sensitive():
    x = {"action": np.array([1, 2], dtype=np.int64),
         "observation": np.zeros((2, 3), dtype=np.float32)}
    assert record_digest(x) == record_digest(dict(reversed(list(x.items()))))
    y = {"action": x["action"].copy(), "observation": x["observation"].copy()}
    y["action"][0] = 0
    assert record_digest(x) != record_digest(y)
