"""CPU smoke test for the reference code: a tiny randomly initialised Llama, no downloads.

    python reference/test_reference.py

Checks that full fine-tuning and LoRA both train and save under AdamW and Lion, that only
completion tokens are trained on, that Lion takes sign-sized steps, and that the cheap
LoRA norm formula equals the dense computation.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune as ft  # noqa: E402
import update_norms as un  # noqa: E402


class TinyTokenizer:
    """Character-level stand-in for a Hugging Face tokenizer."""
    eos_token_id, pad_token_id, eos_token = 1, 0, "</s>"

    def __call__(self, text, add_special_tokens=True):
        ids = [2 + (ord(c) % 60) for c in text]
        return {"input_ids": ([3] if add_special_tokens else []) + ids}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


PAIRS = [("Question: 2+2?\nAnswer:", " 4 #### 4"), ("Question: 3+5?\nAnswer:", " 8 #### 8")] * 8


def test_prompt_is_masked():
    t = ft.tokenize_pair(TinyTokenizer(), "Question: hi\nAnswer:", " yes", 64)
    n_prompt = len(TinyTokenizer()("Question: hi\nAnswer:")["input_ids"])
    assert all(x == -100 for x in t.labels[:n_prompt])
    assert all(x != -100 for x in t.labels[n_prompt:])


def test_formats():
    p, c = ft.format_example("metamathqa", {"query": "q", "response": "r"}, "chat")
    assert (p, c) == ("[INST] q [/INST]", " r")
    p, c = ft.format_example("arc_easy", {"question": "Q", "answerKey": "B",
                                          "choices": {"text": ["x", "y"], "label": ["A", "B"]}})
    assert c == " y"


def test_lion_takes_sign_steps():
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([1e-9, -3.0, 5.0, -1e-9])
    ft.DecoupledLionW([p], lr=0.1).step()
    assert torch.allclose(p.detach(), torch.tensor([-0.1, 0.1, -0.1, 0.1]))


def run(method: str, optimizer: str, out: Path) -> list[float]:
    args = ft.parse_args(["--model", "tiny", "--task", "gsm8k", "--method", method, "--rank", "4",
                          "--optimizer", optimizer, "--lr", "1e-3", "--max-steps", "6",
                          "--batch-size", "4", "--micro-batch-size", "2", "--log-every", "100",
                          "--out", str(out)])
    return ft.train(args, pairs=PAIRS, base_model=tiny_model(), tokenizer=TinyTokenizer())


def test_training_and_norms():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for method in ["full", "lora"]:
            for opt in ["adamw", "lion"]:
                losses = run(method, opt, d / f"{method}_{opt}")
                assert len(losses) == 6 and all(l == l for l in losses), (method, opt)
                assert (d / f"{method}_{opt}" / "checkpoint-final" / "model.safetensors").exists()
        # The adapter-only measurement must agree with the dense merged-minus-base measurement.
        from safetensors.torch import load_file
        base_sd = tiny_model().state_dict()
        lora_dir = d / "lora_lion" / "checkpoint-final"
        from_adapter = un.measure_adapter(base_sd, str(lora_dir / "adapter"))
        from_merged = un.measure_full(base_sd, load_file(str(lora_dir / "model.safetensors")))
        assert from_adapter["n_mat"] == from_merged["n_mat"] == 14  # 7 projections x 2 layers
        assert abs(from_adapter["rel_fro"] - from_merged["rel_fro"]) < 1e-3 * from_merged["rel_fro"]
        # LoRA norm shortcut == dense norms of scale * B @ A
        a, b, scale = torch.randn(4, 32), torch.randn(48, 4), 2.0
        fro, sig = un.lora_norms(a, b, scale)
        dense = scale * b @ a
        assert abs(fro - dense.norm().item()) < 1e-3 * fro
        assert abs(sig - torch.linalg.matrix_norm(dense, 2).item()) < 1e-3 * sig
        assert abs(un.spectral_norm(dense) - sig) < 1e-2 * sig


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("passed", name)
