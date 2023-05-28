import importlib

import torch


def cos_sin(
    self,
    seq_len: int,
    device="cuda",
    dtype=torch.float16,
) -> torch.Tensor:
    if seq_len != self.seq_len_cached:
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        if dtype in [torch.float16, torch.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()[None, :, :]
        self.sin_cached = emb.sin()[None, :, :]

        self.cos_cached = self.cos_cached.type(dtype)
        self.sin_cached = self.sin_cached.type(dtype)

    return self.cos_cached, self.sin_cached


def inject_monkey_patch_falcon(model_repo: str):
    def fn():
        model_repo_escape = model_repo.replace("/", "_")
        imported_module = importlib.import_module(
            f"transformers_modules.{model_repo_escape}.modelling_RW"
        )
        imported_module.RotaryEmbedding.cos_sin = cos_sin
    return fn
