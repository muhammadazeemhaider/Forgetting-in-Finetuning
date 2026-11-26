"""Reference implementation of the weight-displacement measurement (results/UPDATE_NORMS*.csv).

NOTE: re-implementation written after the experiments; see reference/README.md.

For every LoRA-targeted projection (q, k, v, o, gate, up, down) of every block, with
dW = W_finetuned - W_base:
  rel_fro  = ||dW||_F / ||W_base||_F                (pooled over all matrices)
  srank    = ||dW||_F^2 / ||dW||_2^2                (stable rank: effective number of directions)
  rel_sig1 = ||dW||_2 / ||W_base||_2                (mean over matrices)

For LoRA, dW = (alpha / r) * B @ A is never formed: its norms come exactly from the r x r
matrices (B^T B)(A A^T), so the measurement is cheap even at 7B.

Usage:
    python reference/update_norms.py --base meta-llama/Llama-3.2-1B --full checkpoints/full/checkpoint-final
    python reference/update_norms.py --base meta-llama/Llama-3.2-1B --adapter checkpoints/r16/checkpoint-final/adapter
"""
from __future__ import annotations

import argparse
import re

import torch

PROJ = re.compile(r"layers\.(\d+)\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")


def spectral_norm(w: torch.Tensor, iters: int = 30) -> float:
    """Top singular value by power iteration (a lower bound for very flat spectra)."""
    v = torch.randn(w.shape[1], dtype=w.dtype, device=w.device)
    for _ in range(iters):
        v = w.T @ (w @ v)
        v /= v.norm() + 1e-12
    return (w @ v).norm().item()


def lora_norms(a: torch.Tensor, b: torch.Tensor, scale: float) -> tuple[float, float]:
    """Frobenius and spectral norm of scale * B @ A from the small r x r product."""
    m = (b.T @ b) @ (a @ a.T)  # same nonzero eigenvalues as (BA)^T (BA)
    eig = torch.linalg.eigvals(m).real.clamp(min=0) * scale**2
    return eig.sum().sqrt().item(), eig.max().sqrt().item()


def summarize(rows: list[tuple[float, float, float, float]]) -> dict[str, float]:
    """rows: (||dW||_F, ||dW||_2, ||W||_F, ||W||_2) per matrix."""
    fro2 = sum(r[0] ** 2 for r in rows)
    base2 = sum(r[2] ** 2 for r in rows)
    return {
        "n_mat": len(rows),
        "rel_fro": (fro2 / base2) ** 0.5,
        "srank": sum((r[0] / r[1]) ** 2 for r in rows if r[1] > 0) / len(rows),
        "rel_sig1": sum(r[1] / r[3] for r in rows) / len(rows),
    }


def measure_full(base_sd: dict, ft_sd: dict) -> dict[str, float]:
    rows = []
    for k, w0 in base_sd.items():
        if not PROJ.search(k):
            continue
        w0 = w0.float()
        dw = ft_sd[k].float() - w0
        rows.append((dw.norm().item(), spectral_norm(dw), w0.norm().item(), spectral_norm(w0)))
    return summarize(rows)


def measure_adapter(base_sd: dict, adapter_dir: str) -> dict[str, float]:
    import json
    from pathlib import Path

    from safetensors.torch import load_file
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    ad = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    rows = []
    for k, w0 in base_sd.items():
        m = PROJ.search(k)
        if not m:
            continue
        tail = "." + k[m.start():].removesuffix(".weight")  # ".layers.3.self_attn.q_proj"
        a = next((v for n, v in ad.items() if n.endswith(tail + ".lora_A.weight")), None)
        b = next((v for n, v in ad.items() if n.endswith(tail + ".lora_B.weight")), None)
        if a is None or b is None:
            continue
        w0 = w0.float()
        fro, sig = lora_norms(a.float(), b.float(), scale)
        rows.append((fro, sig, w0.norm().item(), spectral_norm(w0)))
    return summarize(rows)


def main() -> None:
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", help="full fine-tuned checkpoint")
    g.add_argument("--adapter", help="PEFT LoRA adapter directory")
    args = ap.parse_args()

    base_sd = AutoModelForCausalLM.from_pretrained(args.base).state_dict()
    if args.full:
        print(measure_full(base_sd, AutoModelForCausalLM.from_pretrained(args.full).state_dict()))
    else:
        print(measure_adapter(base_sd, args.adapter))


if __name__ == "__main__":
    main()
