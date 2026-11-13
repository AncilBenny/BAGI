"""Static checkpoint auditing must never unpickle model contents."""
import hashlib
import io
import json
import zipfile

import pytest

from bagi.experiments.pretrained_checkpoint_audit import (
    probe_flat_semantics,
    verify_and_inspect_archive,
)


def specimen(*, traversal=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("data", json.dumps({
            "observation_space": {":serialized:": "NOT_A_REAL_PICKLE"},
            "policy_class": "ActorCriticPolicy",
        }))
        archive.writestr("policy.pth", b"untrusted tensor pickle NEVER loaded")
        archive.writestr("_stable_baselines3_version", "2.0.0")
        if traversal:
            archive.writestr("../escape.txt", "bad")
    blob = output.getvalue()
    return blob, hashlib.sha256(blob).hexdigest()


def test_safe_audit_only_reads_json_not_serialized_payloads():
    blob, digest = specimen()
    report = verify_and_inspect_archive(blob, digest)
    assert report["archive_sha256_verified"] == digest
    assert report["execution_or_deserialization_performed"] is False
    assert report["serialized_python_fields_not_loaded"] == ["observation_space"]
    assert report["original_sb3_version"] == "2.0.0"


def test_invalid_hash_fails_closed():
    blob, _ = specimen()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_and_inspect_archive(blob, "0" * 64)


def test_path_traversal_fails_closed():
    blob, digest = specimen(traversal=True)
    with pytest.raises(ValueError, match="Unsafe"):
        verify_and_inspect_archive(blob, digest)



def test_upstream_flat_wrapper_contract_not_reinvented():
    # Uses Farama's own flat-observation encoder, not BAGI replacement logic.
    report = probe_flat_semantics("MiniGrid-DoorKey-5x5-v0")
    assert report["all_checks_passed"]
    assert len(report["examples"]) == 3
    for result in report["examples"]:
        assert result["image_features"] == 147
        assert result["mission_features"] == 2688
        assert result["mission_encoding_dims"] == [96, 28]
        assert result["flat_output_features"] == 2835
        assert result["image_prefix_equal"]
        assert result["mission_one_hot_suffix_equal"]
