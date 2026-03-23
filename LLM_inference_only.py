#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from swift.llm import PtEngine, InferRequest, RequestConfig


def run_inference_on_file(prompts_json, processed_csv, output_csv, engine, cfg):
    with open(prompts_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    with open(processed_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {str(row["id"]).strip(): row for row in reader}

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if not rows:
        print(f"No rows in {processed_csv}")
        return

    fieldnames = list(next(iter(rows.values())).keys()) + ["LLM_Prediction"]

    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for prompt in tqdm(prompts, desc=f"{Path(prompts_json).stem}"):

            prompt_id = str(prompt.get("id")).strip()
            messages = prompt["messages"]
            labels = prompt["labels"]

            infer_req = InferRequest(messages=messages)

            try:
                response = engine.infer([infer_req], cfg)[0].choices[0].message.content.strip()
            except Exception as e:
                response = ""

            first_token = response.split()[0].strip(".").upper() if response else ""
            predicted_relation = labels.get(first_token, "no_relation")

            if prompt_id in rows:
                row = rows[prompt_id].copy()
                row["LLM_Prediction"] = predicted_relation
                writer.writerow(row)

    print(f"Saved: {output_csv}")


def process_folder(prompts_dir, processed_dir, output_dir, engine, cfg):
    prompts_dir = Path(prompts_dir)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    for prompt_file in sorted(prompts_dir.glob("*.json")):
        stem = prompt_file.stem

        processed_csv = processed_dir / f"{stem}.csv"
        output_csv = output_dir / f"{stem}.csv"

        if not processed_csv.exists():
            print(f"Missing: {processed_csv}")
            continue

        run_inference_on_file(
            str(prompt_file),
            str(processed_csv),
            str(output_csv),
            engine,
            cfg
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts_dir", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    engine = PtEngine(args.model_path)

    cfg = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    process_folder(
        args.prompts_dir,
        args.processed_dir,
        args.output_dir,
        engine,
        cfg
    )

    print("Done.")


if __name__ == "__main__":
    main()