#!/usr/bin/env python3
"""Aggregate intruder JSONL records and join them to accuracy/geometry results."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TASK_DISPLAY = {
    "gsm8k": "GSM8K",
    "triviaqa": "TriviaQA",
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
}
MODEL_DISPLAY = {
    "llama-3.2-1b": "Llama-3.2-1B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "pythia-1.4b": "Pythia-1.4B",
    "llama-3.2-3b": "Llama-3.2-3B",
    "qwen2.5-3b": "Qwen2.5-3B",
    "pythia-2.8b": "Pythia-2.8B",
    "smollm3-3b": "SmolLM3-3B",
}
VARIANT_DISPLAY = {
    "full-ft": "Full FT",
    "lora-r16": "LoRA r16",
    "lora-r64": "LoRA r64",
    "lora-r256": "LoRA r256",
}


def records(root: Path) -> Iterable[dict]:
    for path in sorted(root.glob("*/*/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_accuracy(results_root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for task, task_display in TASK_DISPLAY.items():
        path = results_root / f"train_on_{task}" / "FINAL_TABLE.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for model_key, model_display in MODEL_DISPLAY.items():
            model_rows = [row for row in rows if row["Model"] == model_display]
            if not model_rows:
                continue
            target = next(row for row in model_rows if row["Task"] == task_display)
            for variant, display in VARIANT_DISPLAY.items():
                target_acc = float(target[display])
                base_target = float(target["Base"])
                non_target_changes = [
                    float(row[display]) - float(row["Base"])
                    for row in model_rows
                    if row["Task"] != task_display
                ]
                out[(task, model_key, variant)] = {
                    "base_target_accuracy": base_target,
                    "target_accuracy": target_acc,
                    "target_gain": target_acc - base_target,
                    "mean_non_target_change": sum(non_target_changes) / len(non_target_changes),
                }
    return out


def summarize(group: list[dict], epsilons: list[float], ks: list[int]) -> dict[str, object]:
    first = group[0]
    row: dict[str, object] = {
        "train_task": first["train_task"],
        "model": first["model"],
        "variant": first["variant"],
        "method": first["method"],
        "lora_rank": first.get("lora_rank"),
        "matrix_count": len(group),
        "mean_relative_update_norm": sum(x["relative_update_norm"] for x in group) / len(group),
    }
    for k in ks:
        selected = [vector for matrix in group for vector in matrix["vectors"] if vector["singular_rank"] <= k]
        total_energy = sum(vector["singular_value"] ** 2 for vector in selected)
        for epsilon in epsilons:
            intruders = [vector for vector in selected if vector["left_max_abs_cosine"] < epsilon]
            suffix = f"k{k}_eps{epsilon:g}".replace(".", "p")
            row[f"intruder_count_{suffix}"] = len(intruders)
            row[f"intruder_rate_{suffix}"] = len(intruders) / max(len(selected), 1)
            row[f"intruder_energy_{suffix}"] = (
                sum(vector["singular_value"] ** 2 for vector in intruders) / max(total_energy, 1e-30)
            )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    vector_rows: list[dict[str, object]] = []
    for record in records(args.raw_root):
        key = (record["train_task"], record["model"], record["variant"])
        grouped[key].append(record)
        common = {name: value for name, value in record.items() if name != "vectors"}
        common.pop("paper_primary", None)
        for vector in record["vectors"]:
            vector_rows.append({**common, **vector})

    if not grouped:
        raise SystemExit(f"no JSONL records found under {args.raw_root}")
    accuracy = load_accuracy(args.results_root)
    summary_rows = []
    for key, group in sorted(grouped.items()):
        row = summarize(group, args.epsilons, args.ks)
        row.update(accuracy.get(key, {}))
        summary_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "intruder_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    vectors_path = args.output_dir / "intruder_vectors.csv.gz"
    with gzip.open(vectors_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(vector_rows[0]))
        writer.writeheader()
        writer.writerows(vector_rows)
    print(f"wrote {summary_path} ({len(summary_rows)} conditions)")
    print(f"wrote {vectors_path} ({len(vector_rows)} singular-vector rows)")


if __name__ == "__main__":
    main()
