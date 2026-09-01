from scripts.rewrite_e97_sft_system_prompt import encode_pieces


class ByteEncoding:
    def encode_ordinary(self, text):
        return list(text.encode())

    def decode_single_token_bytes(self, token):
        return bytes([token])


def test_encode_pieces_preserves_target_boundaries_and_text():
    pieces = [("System:\nnew\n\nAssistant:\n", False), ("Final: done", True), ("\x1e", True)]
    tokens, masks, text = encode_pieces(ByteEncoding(), pieces)
    assert bytes(tokens).decode() == text
    assert sum(masks) == len("Final: done\x1e")
    assert all(mask == 0 for mask in masks[:len("System:\nnew\n\nAssistant:\n")])
