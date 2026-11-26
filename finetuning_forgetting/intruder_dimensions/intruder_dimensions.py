#!/usr/bin/env python3
"""Spectral intruder-dimension analysis for the LoRA vs full fine-tuning study.

This reproduces the primary detector from Shuttleworth et al. (2025): for
each tuned weight matrix, compare its top-k left singular vectors against all
left singular vectors of the corresponding base matrix.  A tuned vector is an
"intruder" at threshold epsilon when its maximum absolute cosine similarity
to any base singular vector is below epsilon.

The implementation is deliberately matrix-major.  It computes one base SVD,
uses it for Full FT and all LoRA ranks, writes one atomic JSONL record per
matrix/variant, and then releases the tensors.  That keeps memory and disk use
bounded and makes preempted Slurm jobs resumable.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from safetensors import safe_open


MODELS: dict[str, str] = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B-Base",
}
TASKS = ("gsm8k", "triviaqa", "hellaswag", "arc_easy")
VARIANTS = ("full-ft", "lora-r16", "lora-r64", "lora-r256")


def canonical_adapter_key(adapter_key: str) -> tuple[str, str] | None:
    """Map a PEFT LoRA tensor key to (base weight key, A-or-B)."""
    match = re.search(r"\.lora_([AB])(?:\.default)?\.weight$", adapter_key)
    if not match:
        return None
    side = match.group(1)
    stem = adapter_key[: match.start()]
    for prefix in ("base_model.model.", "base_model."):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return f"{stem}.weight", side


def layer_number(name: str) -> int | None:
    for pattern in (r"\.layers\.(\d+)\.", r"\.layer\.(\d+)\.", r"\.h\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def module_type(name: str) -> str:
    return name.rsplit(".", 1)[0].rsplit(".", 1)[-1]


class TensorStore:
    """Lazy, key-addressable reader for one or more safetensors shards."""

    def __init__(self, directory_or_file: Path):
        source = Path(directory_or_file)
        if source.is_file():
            files = [source]
        else:
            index = source / "model.safetensors.index.json"
            if index.exists():
                data = json.loads(index.read_text())
                files = sorted({source / value for value in data["weight_map"].values()})
            else:
                files = sorted(source.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors files found under {source}")
        unreadable = [str(path) for path in files if not os.access(path, os.R_OK)]
        if unreadable:
            raise PermissionError("unreadable checkpoint tensor(s): " + ", ".join(unreadable[:3]))
        self.files = files
        self.key_to_file: dict[str, Path] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in self.key_to_file:
                        raise ValueError(f"duplicate tensor key {key}")
                    self.key_to_file[key] = path

    def keys(self) -> Iterable[str]:
        return self.key_to_file.keys()

    def get(self, key: str) -> torch.Tensor:
        path = self.key_to_file[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    path: Path
    rank: int | None = None
    alpha: float | None = None
    scaling: float | None = None


def resolve_base_snapshot(project_root: Path, hub_id: str) -> Path:
    repo = "models--" + hub_id.replace("/", "--")
    roots = [project_root / ".hf_cache" / "hub" / repo / "snapshots"]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(path for path in root.iterdir() if path.is_dir())
    valid = [
        path
        for path in candidates
        if (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
    ]
    if not valid:
        raise FileNotFoundError(f"cached base weights not found for {hub_id} under {roots[0]}")
    return max(valid, key=lambda path: path.stat().st_mtime)


def discover_full_checkpoint(project_root: Path, train_task: str, model: str) -> Path:
    root = project_root / "full_ft_models" / f"finetune_on_{train_task}"
    candidates = list(root.glob(f"**/{model}/checkpoint-final"))
    if not candidates:
        raise FileNotFoundError(f"no Full FT checkpoint for {train_task}/{model}")
    preferred = [path for path in candidates if "opt" in path.parts]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"ambiguous Full FT checkpoint for {train_task}/{model}: {candidates}")


def discover_lora_checkpoint(
    project_root: Path, train_task: str, model: str, rank: int
) -> Path:
    path = (
        project_root
        / "lora_modelweights"
        / f"finetune_on_{train_task}"
        / model
        / f"lora-r{rank}_seed13"
        / "checkpoint-final"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_adapter_variant(path: Path, name: str) -> Variant:
    config = json.loads((path / "adapter_config.json").read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if config.get("use_rslora", False):
        scaling = alpha / math.sqrt(rank)
    else:
        scaling = alpha / rank
    if config.get("fan_in_fan_out", False):
        raise NotImplementedError("fan_in_fan_out=True requires transposing the reconstructed update")
    return Variant(name=name, kind="lora", path=path, rank=rank, alpha=alpha, scaling=scaling)


def discover_variants(
    project_root: Path, train_task: str, model: str, requested: Sequence[str]
) -> list[Variant]:
    variants: list[Variant] = []
    for name in requested:
        if name == "full-ft":
            variants.append(
                Variant(name=name, kind="full", path=discover_full_checkpoint(project_root, train_task, model))
            )
        elif name.startswith("lora-r"):
            rank = int(name.removeprefix("lora-r"))
            path = discover_lora_checkpoint(project_root, train_task, model, rank)
            variants.append(read_adapter_variant(path, name))
        else:
            raise ValueError(f"unknown variant {name}")
    return variants


def adapter_tensor_map(store: TensorStore) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for key in store.keys():
        parsed = canonical_adapter_key(key)
        if parsed is None:
            continue
        base_key, side = parsed
        mapped.setdefault(base_key, {})[side] = key
    incomplete = {key: value for key, value in mapped.items() if set(value) != {"A", "B"}}
    if incomplete:
        raise ValueError(f"incomplete LoRA A/B pairs: {list(incomplete)[:3]}")
    if not mapped:
        raise ValueError("adapter contains no recognized LoRA A/B tensors")
    return mapped


def reconstruct_lora(
    base_weight: torch.Tensor,
    adapter_store: TensorStore,
    pair: Mapping[str, str],
    scaling: float,
    device: torch.device,
) -> torch.Tensor:
    base = base_weight.to(device=device, dtype=torch.float32)
    a = adapter_store.get(pair["A"]).to(device=device, dtype=torch.float32)
    b = adapter_store.get(pair["B"]).to(device=device, dtype=torch.float32)
    update = torch.matmul(b, a).mul_(scaling)
    if update.shape != base.shape:
        raise ValueError(f"LoRA update shape {tuple(update.shape)} != base {tuple(base.shape)}")
    return base + update


def exact_svd(weight: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = weight.to(device=device, dtype=torch.float32)
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        if device.type != "cuda":
            raise
        # gesvd is slower but more robust than the CUDA default for ill-conditioned matrices.
        return torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")


def vector_diagnostics(
    u_base: torch.Tensor,
    s_base: torch.Tensor,
    vh_base: torch.Tensor,
    u_tuned: torch.Tensor,
    s_tuned: torch.Tensor,
    vh_tuned: torch.Tensor,
    top_k: int,
    threshold: float,
) -> list[dict[str, float | int | bool]]:
    k = min(top_k, s_tuned.numel())
    left_coeff = torch.abs(u_tuned[:, :k].T @ u_base)
    right_coeff = torch.abs(vh_tuned[:k, :] @ vh_base.T)
    left_max, left_idx = left_coeff.max(dim=1)
    right_max, right_idx = right_coeff.max(dim=1)
    left_projection_residual = (1.0 - left_coeff.square().sum(dim=1)).clamp(min=0.0, max=1.0)
    right_projection_residual = (1.0 - right_coeff.square().sum(dim=1)).clamp(min=0.0, max=1.0)
    rows: list[dict[str, float | int | bool]] = []
    for j in range(k):
        sigma = float(s_tuned[j].item())
        next_sigma = float(s_tuned[j + 1].item()) if j + 1 < s_tuned.numel() else 0.0
        rows.append(
            {
                "singular_rank": j + 1,
                "singular_value": sigma,
                "base_same_rank_singular_value": float(s_base[j].item()) if j < s_base.numel() else math.nan,
                "relative_spectral_gap": (sigma - next_sigma) / max(abs(sigma), 1e-30),
                "left_max_abs_cosine": float(left_max[j].item()),
                "left_matched_base_rank": int(left_idx[j].item()) + 1,
                "left_projection_residual": float(left_projection_residual[j].item()),
                "right_max_abs_cosine": float(right_max[j].item()),
                "right_matched_base_rank": int(right_idx[j].item()) + 1,
                "right_projection_residual": float(right_projection_residual[j].item()),
                "is_intruder": bool(left_max[j].item() < threshold),
            }
        )
    return rows


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def completed_matrices(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: ignoring partial JSONL line {line_number} in {path}", file=sys.stderr)
                continue
            done.add(record["matrix_name"])
    return done


def open_variant_stores(variants: Sequence[Variant]) -> tuple[dict[str, TensorStore], dict[str, dict[str, dict[str, str]]]]:
    stores: dict[str, TensorStore] = {}
    adapter_maps: dict[str, dict[str, dict[str, str]]] = {}
    for variant in variants:
        source = variant.path / "adapter_model.safetensors" if variant.kind == "lora" else variant.path
        store = TensorStore(source)
        stores[variant.name] = store
        if variant.kind == "lora":
            adapter_maps[variant.name] = adapter_tensor_map(store)
    return stores, adapter_maps


def analyze(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    if args.model not in MODELS:
        raise ValueError(f"unsupported model {args.model}; choose from {sorted(MODELS)}")
    if args.train_task not in TASKS:
        raise ValueError(f"unsupported task {args.train_task}; choose from {TASKS}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    variants = discover_variants(project_root, args.train_task, args.model, args.variants)
    base_path = resolve_base_snapshot(project_root, MODELS[args.model])
    print(f"base={base_path}")
    for variant in variants:
        print(f"{variant.name}={variant.path}")

    # Fail before expensive work if any weight file is inaccessible.
    base_store = TensorStore(base_path)
    stores, adapter_maps = open_variant_stores(variants)

    lora_maps = list(adapter_maps.values())
    if not lora_maps:
        raise ValueError("at least one LoRA variant is required to define the fair target matrix set")
    matrix_keys = sorted(set.intersection(*(set(mapping) for mapping in lora_maps)))
    missing_base = [key for key in matrix_keys if key not in base_store.key_to_file]
    if missing_base:
        raise KeyError(f"base checkpoint missing mapped LoRA tensors: {missing_base[:3]}")
    for variant in variants:
        if variant.kind == "full":
            missing = [key for key in matrix_keys if key not in stores[variant.name].key_to_file]
            if missing:
                raise KeyError(f"{variant.name} missing tensors: {missing[:3]}")
    if args.matrix_regex:
        pattern = re.compile(args.matrix_regex)
        matrix_keys = [key for key in matrix_keys if pattern.search(key)]
    if args.max_matrices is not None:
        matrix_keys = matrix_keys[: args.max_matrices]
    if not matrix_keys:
        raise ValueError("no matrices selected")

    output_paths = {
        variant.name: args.output_root / args.train_task / args.model / f"{variant.name}.jsonl"
        for variant in variants
    }
    done = {name: completed_matrices(path) if args.resume else set() for name, path in output_paths.items()}
    print(f"selected_matrices={len(matrix_keys)} device={device} top_k={args.top_k} epsilon={args.threshold}")

    for matrix_index, key in enumerate(matrix_keys, 1):
        pending = [variant for variant in variants if key not in done[variant.name]]
        if not pending:
            print(f"[{matrix_index}/{len(matrix_keys)}] skip {key}")
            continue
        started = time.perf_counter()
        base_cpu = base_store.get(key)
        u0, s0, vh0 = exact_svd(base_cpu, device)
        for variant in pending:
            store = stores[variant.name]
            if variant.kind == "full":
                tuned = store.get(key).to(device=device, dtype=torch.float32)
            else:
                tuned = reconstruct_lora(
                    base_cpu,
                    store,
                    adapter_maps[variant.name][key],
                    float(variant.scaling),
                    device,
                )
            update_norm = float(torch.linalg.vector_norm(tuned - base_cpu.to(device=device, dtype=torch.float32)).item())
            base_norm = float(torch.linalg.vector_norm(base_cpu.float()).item())
            ut, st, vht = exact_svd(tuned, device)
            vectors = vector_diagnostics(u0, s0, vh0, ut, st, vht, args.top_k, args.threshold)
            record = {
                "schema_version": 1,
                "paper_primary": {"top_k": args.top_k, "epsilon": args.threshold, "side": "left"},
                "model": args.model,
                "base_model": MODELS[args.model],
                "train_task": args.train_task,
                "variant": variant.name,
                "method": variant.kind,
                "lora_rank": variant.rank,
                "lora_alpha": variant.alpha,
                "lora_scaling": variant.scaling,
                "matrix_name": key,
                "layer": layer_number(key),
                "module_type": module_type(key),
                "matrix_rows": int(base_cpu.shape[0]),
                "matrix_cols": int(base_cpu.shape[1]),
                "base_rank": int(min(base_cpu.shape)),
                "base_frobenius_norm": base_norm,
                "update_frobenius_norm": update_norm,
                "relative_update_norm": update_norm / max(base_norm, 1e-30),
                "vectors": vectors,
            }
            append_jsonl(output_paths[variant.name], record)
            del tuned, ut, st, vht
        del base_cpu, u0, s0, vh0
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        print(f"[{matrix_index}/{len(matrix_keys)}] {key} {elapsed:.2f}s", flush=True)


def preflight(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    for model in MODELS:
        try:
            base = resolve_base_snapshot(root, MODELS[model])
            TensorStore(base)
            base_ok = True
        except Exception as exc:  # report the entire manifest rather than stopping early
            base = Path("")
            base_ok = False
            failures.append(f"base/{model}: {exc}")
        for task in TASKS:
            for name in VARIANTS:
                try:
                    variant = discover_variants(root, task, model, [name])[0]
                    source = variant.path / "adapter_model.safetensors" if variant.kind == "lora" else variant.path
                    TensorStore(source)
                    readable = True
                    error = ""
                except Exception as exc:
                    variant = None
                    readable = False
                    error = str(exc)
                    failures.append(f"{task}/{model}/{name}: {exc}")
                rows.append(
                    {
                        "model": model,
                        "train_task": task,
                        "variant": name,
                        "base_readable": base_ok,
                        "checkpoint_readable": readable,
                        "checkpoint": str(variant.path) if variant else "",
                        "error": error,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}: {len(rows)} conditions, {len(failures)} failures")
    for failure in failures[:20]:
        print("FAIL", failure)
    if failures:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("preflight", help="validate all base and tuned checkpoint paths/readability")
    scan.add_argument("--project-root", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    scan.set_defaults(func=preflight)

    run = subparsers.add_parser("analyze", help="analyze one model/training-task condition group")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--train-task", required=True)
    run.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    run.add_argument("--device", default="cuda")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--matrix-regex")
    run.add_argument("--max-matrices", type=int)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(func=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
