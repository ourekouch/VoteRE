#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
import string
from pathlib import Path
from tqdm import tqdm
from templates import DATASET_TEMPLATES


def transform_row_to_conversation(row, dataset_name):
    subject = row["subject"]
    subject_type = row["subject_type"]
    object_ = row["object"]
    object_type = row["object_type"]
    sentence = row["sentence"]
    gold_relation = row.get("true_label", "no_relation")

    dataset = DATASET_TEMPLATES[dataset_name]
    templates = dataset["templates"]
    valid_conditions = dataset["valid_conditions_rev"]

    entity_pair = f"{subject_type}:{object_type}"
    valid_rels = valid_conditions.get(entity_pair, []).copy()

    if "no_relation" not in valid_rels:
        valid_rels.append("no_relation")

    options = []
    labels = list(string.ascii_uppercase)

    for rel in valid_rels:
        if rel in templates:
            try:
                text = templates[rel][0].format(subj=subject, obj=object_)
                options.append((labels[len(options)], rel, text))
            except Exception:
                continue

    if not options:
        return None

    instruction = "Determine which option can be inferred from the given sentence.\n"
    instruction += f"Sentence: {sentence}\nOptions:"
    for label, _, text in options:
        instruction += f"\n{label}. {text}"
    instruction += "\nWhich option can be inferred from the given sentence?"

    return {
        "id": row.get("id"),
        "messages": [
            {
                "role": "system",
                "content": "You are an LLM trained to infer relationships between entities from context."
            },
            {
                "role": "user",
                "content": instruction
            }
        ],
        "labels": {label: rel for label, rel, _ in options},
        "gold_relation": gold_relation
    }


def normalize_row_from_processed_csv(row):
    return {
        "id": row["id"],
        "sentence": row["Tokens"],
        "subject": row["Subject_Entity"],
        "subject_type": row["Subject_Type"],
        "object": row["Object_Entity"],
        "object_type": row["Object_Type"],
        "subject_start": row.get("Subject_Start"),
        "subject_end": row.get("Subject_End"),
        "object_start": row.get("Object_Start"),
        "object_end": row.get("Object_End"),
        "true_label": row.get("True_Labels", "no_relation")
    }


def csv_to_prompt_json(csv_path, dataset_name, output_json):
    prompts = []

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {
            "id",
            "Tokens",
            "Subject_Entity",
            "Object_Entity",
            "Subject_Type",
            "Object_Type",
            "True_Labels"
        }

        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns in {csv_path}: {sorted(missing)}")

        for row in tqdm(reader, desc=f"Building prompts from {os.path.basename(csv_path)}"):
            adapted = normalize_row_from_processed_csv(row)
            conv = transform_row_to_conversation(adapted, dataset_name)

            if conv is not None:
                conv["subject_start"] = adapted.get("subject_start")
                conv["subject_end"] = adapted.get("subject_end")
                conv["object_start"] = adapted.get("object_start")
                conv["object_end"] = adapted.get("object_end")
                prompts.append(conv)

    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(prompts, out, indent=2, ensure_ascii=False)

    print(f"Saved prompt JSON to: {output_json}")
    print(f"Total prompts: {len(prompts)}")


def process_folder(input_dir, output_dir, dataset_name):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return

    for csv_file in csv_files:
        output_json = output_dir / f"{csv_file.stem}.json"
        csv_to_prompt_json(
            csv_path=str(csv_file),
            dataset_name=dataset_name,
            output_json=str(output_json)
        )


def main():
    parser = argparse.ArgumentParser(description="Convert processed CSV files to prompt JSON files")
    parser.add_argument("--input_dir", required=True, help="Path to processed CSV folder")
    parser.add_argument("--output_dir", required=True, help="Path to prompt JSON folder")
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["TACRED", "TACREV", "RETACRED"],
        help="Dataset name as defined in templates.py"
    )

    args = parser.parse_args()

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )


if __name__ == "__main__":
    main()