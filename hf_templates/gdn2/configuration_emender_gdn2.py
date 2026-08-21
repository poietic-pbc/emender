from transformers import PretrainedConfig


class EmenderGDN2Config(PretrainedConfig):
    model_type = "emender_gdn2_mlp"

    def __init__(self, gdn2_args=None, vocab_size=50281, **kwargs):
        super().__init__(
            bos_token_id=kwargs.pop("bos_token_id", 50256),
            eos_token_id=kwargs.pop("eos_token_id", 50256),
            pad_token_id=kwargs.pop("pad_token_id", 50256),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            use_cache=kwargs.pop("use_cache", False),
            is_decoder=True,
            **kwargs,
        )
        self.gdn2_args = dict(gdn2_args or {})
        self.vocab_size = int(vocab_size)
        self.num_hidden_layers = int(self.gdn2_args.get("depth", 0))
