import hashlib

import tiktoken

from scripts import build_e97_tulu3_sft as builder


def setup_module():
    builder._WORKER_ENCODING = tiktoken.get_encoding(builder.TOKENIZER)


def test_tulu_record_serialization_and_assistant_mask_are_exact():
    row = {
        "id": "example-1",
        "source": "fixture",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "Spell it."},
            {"role": "assistant", "content": "F-O-U-R"},
        ],
    }
    result = builder.serialize_row(row)
    assert "error" not in result
    encoding = builder._WORKER_ENCODING
    tokens = list(builder.struct.unpack(
        f"<{result['tokens']}I", result["token_bytes"]))
    assert encoding.decode(tokens) == (
        "System:\nBe concise.\n\nUser:\nWhat is two plus two?\n\n"
        "Assistant:\nFour.\n\nUser:\nSpell it.\n\nAssistant:\nF-O-U-R\x1e")
    assert len(result["mask_bytes"]) == len(tokens)
    targeted = b"".join(
        encoding.decode_single_token_bytes(token)
        for token, mask in zip(tokens, result["mask_bytes"]) if mask)
    assert targeted == b"Four.F-O-U-R\x1e"
    assert result["targets"] == sum(result["mask_bytes"])


def test_tulu_record_trims_role_boundary_whitespace_before_masking():
    result = builder.serialize_row({
        "id": "whitespace", "source": "fixture", "messages": [
            {"role": "user", "content": "  question  "},
            {"role": "assistant", "content": "\n answer \n"},
        ]})
    assert "error" not in result
    tokens = list(builder.struct.unpack(
        f"<{result['tokens']}I", result["token_bytes"]))
    assert builder._WORKER_ENCODING.decode(tokens) == (
        "User:\nquestion\n\nAssistant:\nanswer\x1e")
    assert result["boundary_whitespace_trimmed_characters"] == 8


def test_tulu_record_rejects_nonassistant_final_message_and_unknown_role():
    base = {"id": "bad", "source": "fixture"}
    result = builder.serialize_row({
        **base, "messages": [{"role": "user", "content": "unfinished"}]})
    assert result["error"] == "final_message_is_not_assistant"
    result = builder.serialize_row({
        **base, "messages": [
            {"role": "user", "content": "call"},
            {"role": "tool", "content": "result"},
        ]})
    assert result["error"] == "unsupported_role:tool"


def test_tulu_validation_split_is_stable_and_bound_to_source_and_id():
    observed = builder._split("source-a", "record-1")
    assert observed == builder._split("source-a", "record-1")
    digest = hashlib.sha256(
        f"{builder.SCHEMA}\0source-a\0record-1".encode()).digest()
    assert observed == (1 if int.from_bytes(digest[:8], "little") % 100 == 0 else 0)
    values = {builder._split("source-a", f"record-{index}") for index in range(1000)}
    assert values == {0, 1}
