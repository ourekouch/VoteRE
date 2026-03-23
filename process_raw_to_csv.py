import csv
import json
import argparse
from pathlib import Path


def convert_json_to_csv(input_path, output_path):
    """
    Convert TACRED-style JSON to structured CSV with full span information.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "id",
        "Tokens",
        "Subject_Entity",
        "Object_Entity",
        "Subject_Type",
        "Object_Type",
        "Subject_Start",
        "Subject_End",
        "Object_Start",
        "Object_End",
        "True_Labels"
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for entry in data:
            tokens = entry["token"]

            subj_start = entry["subj_start"]
            subj_end = entry["subj_end"]
            obj_start = entry["obj_start"]
            obj_end = entry["obj_end"]

            subject = " ".join(tokens[subj_start:subj_end + 1])
            object_ = " ".join(tokens[obj_start:obj_end + 1])

            writer.writerow([
                entry["id"],
                " ".join(tokens),              # Tokens as full sentence string
                subject,
                object_,
                entry["subj_type"],
                entry["obj_type"],
                subj_start,
                subj_end,
                obj_start,
                obj_end,
                entry["relation"]
            ])

    print(f"✔ Processed: {input_path} -> {output_path}")


def process_folder(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    json_files = list(input_folder.glob("*.json"))

    for json_file in json_files:
        output_file = output_folder / (json_file.stem + ".csv")
        convert_json_to_csv(json_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir)