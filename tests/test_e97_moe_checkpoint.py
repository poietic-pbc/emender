from ndm.e97_moe_checkpoint import SCHEMA, _replicated_owner


def test_checkpoint_schema_and_replicated_ownership_are_stable_and_complete():
    assert SCHEMA == "emender-e97-moe-sharded-v1"
    names = [f"layers.{layer}.mixer.parameter.{index}"
             for layer in range(11) for index in range(17)]
    first = [_replicated_owner(name) for name in names]
    second = [_replicated_owner(name) for name in names]
    assert first == second
    assert all(0 <= owner < 8 for owner in first)
    assert set(first) == set(range(8))
