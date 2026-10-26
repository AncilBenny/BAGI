"""Pure torch synthetic BC checkpoint tests; never import imitation/native XLand."""
from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bagi.experiments.g2_e5_p9_xland_bc_checkpoint import (
    ARM,
    _sha,
    encode_reconstructed,
    load_reconstructed,
    public_probe_inputs,
    save_reconstructed,
)


class DummyActor(torch.nn.Module):
    """Mimics only documented SB3 get_distribution/categorical logits API."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(115, 12), torch.nn.ReLU(),
            torch.nn.Linear(12, 6),
        )

    def get_distribution(self, obs):
        distribution = torch.distributions.Categorical(logits=self.actor(obs))
        return SimpleNamespace(distribution=distribution)


def _models(seeds=(17, 29)):
    return {seed: DummyActor(seed) for seed in seeds}


def _rewrite(payload: bytes, mutate) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as old:
        bodies = [(x.filename, old.read(x.filename)) for x in old.infolist()]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in bodies:
            archive.writestr(name, mutate(name, body))
    return out.getvalue()


def test_public_probes_are_deterministic_and_actual_actions_encoded():
    a = public_probe_inputs()
    assert a.shape == (128, 115)
    assert np.array_equal(a, public_probe_inputs())
    assert np.all(a[:, -28:].reshape(128, 4, 7).sum(axis=2) == 1)


def test_deterministic_synthetic_export_and_no_historical_identity():
    a, b = encode_reconstructed(_models()), encode_reconstructed(_models())
    assert a == b
    with zipfile.ZipFile(io.BytesIO(a)) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["arm"] == ARM
        assert manifest["historical_weight_identity_verified"] is False
        assert manifest["actual_training_input_equivalence_verified"] is False
        assert manifest["authorized_to_run_native_p9"] is False
        assert set(manifest["models"]) == {"17", "29"}
        assert set(z.namelist()) == {"manifest.json"} | {
            item["member"] for model in manifest["models"].values()
            for item in model["tensors"]
        } | {"seed17/probe_logits.npy", "seed29/probe_logits.npy"}


def test_strict_hash_snapshot_parity_freezes_restored_params(tmp_path):
    models = _models()
    source = {seed: [p.clone() for p in model.state_dict().values()]
              for seed, model in models.items()}
    path = tmp_path / "checkpoint.zip"
    written = save_reconstructed(path, models)
    assert written["bc_fits_during_export"] == 0
    fresh = _models()
    result = load_reconstructed(path, expected_sha256=written["sha256"], policies=fresh)
    assert result["authorized_to_run_native_p9"] is False
    assert result["bc_fits_during_load"] == 0
    assert all(not p.requires_grad for actor in fresh.values() for p in actor.parameters())
    for seed in (17, 29):
        for before, after in zip(source[seed], fresh[seed].state_dict().values(), strict=True):
            assert torch.equal(before, after)
    with pytest.raises(FileExistsError):
        save_reconstructed(path, models)



def test_snapshot_export_never_mutates_source_actors_even_on_probe_error():
    """Successful or rejected snapshot probes must leave both live actors intact."""
    def snapshot(actors):
        return {
            seed: {
                "training": actor.training,
                "grad": [p.requires_grad for p in actor.parameters()],
                "state": {name: tensor.clone() for name, tensor in actor.state_dict().items()},
            }
            for seed, actor in actors.items()
        }

    def assert_unchanged(actors, before):
        for seed, actor in actors.items():
            assert actor.training == before[seed]["training"]
            assert [p.requires_grad for p in actor.parameters()] == before[seed]["grad"]
            for name, tensor in actor.state_dict().items():
                assert torch.equal(tensor, before[seed]["state"][name])

    good = _models()
    good_before = snapshot(good)
    assert encode_reconstructed(good)
    assert_unchanged(good, good_before)

    class MissingLogits(DummyActor):
        def get_distribution(self, obs):
            return SimpleNamespace(distribution=SimpleNamespace(logits=None))

    broken = {17: DummyActor(17), 29: MissingLogits(29)}
    broken_before = snapshot(broken)
    with pytest.raises(TypeError, match="categorical logits"):
        encode_reconstructed(broken)
    assert_unchanged(broken, broken_before)


def test_refuse_wrong_checkpoint_bytes_and_missing_seed(tmp_path):
    with pytest.raises(ValueError, match="both registered"):
        encode_reconstructed(_models((17,)))
    path = tmp_path / "a.zip"
    path.write_bytes(encode_reconstructed(_models()))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_reconstructed(path, expected_sha256="0" * 64, policies=_models())
    with pytest.raises(ValueError, match="Both actor seeds"):
        load_reconstructed(path, expected_sha256=_sha(path.read_bytes()),
                           policies=_models((17,)))


def test_reject_manifest_claiming_historical_identity(tmp_path):
    payload = encode_reconstructed(_models())

    def alter(name, raw):
        if name != "manifest.json":
            return raw
        obj = json.loads(raw)
        obj["historical_weight_identity_verified"] = True
        obj["authorized_to_run_native_p9"] = True
        return json.dumps(obj).encode()

    path = tmp_path / "lying.zip"
    path.write_bytes(_rewrite(payload, alter))
    with pytest.raises(ValueError, match="provenance manifest"):
        load_reconstructed(path, expected_sha256=_sha(path.read_bytes()), policies=_models())


def test_reject_extra_and_duplicate_members_even_with_matching_hash(tmp_path):
    payload = encode_reconstructed(_models())
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        entries = [(n, source.read(n)) for n in source.namelist()]
    for filename in ("forbidden.pkl", "seed17/000.npy"):
        path = tmp_path / filename.replace("/", "_")
        with zipfile.ZipFile(path, "w") as z:
            for name, body in entries:
                z.writestr(name, body)
            if filename.endswith("000.npy"):
                with pytest.warns(UserWarning, match="Duplicate name"):
                    z.writestr(filename, b"malicious")
            else:
                z.writestr(filename, b"malicious")
        with pytest.raises(ValueError, match="BC member"):
            load_reconstructed(path, expected_sha256=_sha(path.read_bytes()), policies=_models())


def test_reject_modified_member_even_with_recomputed_whole_hash(tmp_path):
    payload = encode_reconstructed(_models())
    path = tmp_path / "tampered.zip"
    path.write_bytes(_rewrite(payload, lambda name, raw:
                              raw + b"altered" if name == "seed17/000.npy" else raw))
    with pytest.raises(ValueError, match="state member hash mismatch"):
        load_reconstructed(path, expected_sha256=_sha(path.read_bytes()), policies=_models())


def test_bad_actor_state_rejected_before_export():
    actor = DummyActor(17)
    with torch.no_grad():
        actor.actor[0].weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        encode_reconstructed({17: actor, 29: DummyActor(29)})


def test_synthetic_restored_policy_logits_match_actor_forward(tmp_path):
    original = _models()
    path = tmp_path / "bc.zip"
    written = save_reconstructed(path, original)
    reconstructed = _models()
    load_reconstructed(path, expected_sha256=written["sha256"],
                       policies=reconstructed)
    probes = torch.from_numpy(public_probe_inputs())
    for seed in (17, 29):
        with torch.no_grad():
            logits_a = original[seed].get_distribution(probes).distribution.logits
            logits_b = reconstructed[seed].get_distribution(probes).distribution.logits
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(logits_a.argmax(dim=1), logits_b.argmax(dim=1))


def test_reject_false_source_attestation_even_with_new_whole_digest(tmp_path):
    payload = encode_reconstructed(_models())

    def alter(name, raw):
        if name != "manifest.json":
            return raw
        doc = json.loads(raw)
        doc["actual_training_input_equivalence_verified"] = True
        return json.dumps(doc).encode()

    path = tmp_path / "source_lie.zip"
    path.write_bytes(_rewrite(payload, alter))
    with pytest.raises(ValueError, match="provenance manifest"):
        load_reconstructed(path, expected_sha256=_sha(path.read_bytes()), policies=_models())


def test_reject_incompatible_fresh_model_schema(tmp_path):
    path = tmp_path / "reference.zip"
    saved = save_reconstructed(path, _models())
    wrong = _models()
    wrong[29].actor = torch.nn.Sequential(
        torch.nn.Linear(115, 20), torch.nn.ReLU(), torch.nn.Linear(20, 6)
    )
    with pytest.raises(ValueError, match="invalid|match"):
        load_reconstructed(path, expected_sha256=saved["sha256"], policies=wrong)


def test_invalid_actor_distribution_contract_has_type_errors():
    from bagi.experiments.g2_e5_p9_xland_bc_checkpoint import _policy_logits

    with pytest.raises(TypeError, match="distribution API"):
        _policy_logits(torch.nn.Linear(115, 6))

    class MissingLogits(DummyActor):
        def get_distribution(self, obs):
            return SimpleNamespace(distribution=SimpleNamespace(logits=None))

    with pytest.raises(TypeError, match="categorical logits"):
        _policy_logits(MissingLogits(17))


def test_probe_parity_failure_does_not_mutate_either_live_actor(tmp_path):
    """A self-hashed, well-formed but false probe must reject before live writes."""
    payload = encode_reconstructed(_models())
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        bodies = {name: source.read(name) for name in source.namelist()}
    logits = np.load(io.BytesIO(bodies["seed29/probe_logits.npy"]), allow_pickle=False)
    logits[0, 0] += np.float32(0.125)
    buf = io.BytesIO()
    np.save(buf, logits, allow_pickle=False)
    bodies["seed29/probe_logits.npy"] = buf.getvalue()
    manifest = json.loads(bodies["manifest.json"])
    manifest["models"]["29"]["probe_logits_sha256"] = _sha(buf.getvalue())
    bodies["manifest.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    path = tmp_path / "false_probe.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in bodies.items():
            archive.writestr(name, body)
    live = _models()
    with torch.no_grad():
        for actor in live.values():
            actor.actor[0].weight.add_(0.25)
    originals = {seed: {key: value.clone() for key, value in actor.state_dict().items()}
                 for seed, actor in live.items()}
    with pytest.raises(ValueError, match="probe distribution/action mismatch"):
        load_reconstructed(path, expected_sha256=_sha(path.read_bytes()), policies=live)
    for seed, actor in live.items():
        assert actor.training
        assert all(p.requires_grad for p in actor.parameters())
        for key, value in actor.state_dict().items():
            assert torch.equal(value, originals[seed][key])
