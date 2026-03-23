# VoteRE

VoteRE is a lightweight voting-based framework for relation extraction that combines predictions from multiple voters and applies a configurable majority rule. The framework is designed for clean experimentation with both LoRA-adapted LLM voters and supervised PLM voters.

---

## Overview

The VoteRE pipeline follows four main stages:

1. Raw data processing  
2. Prompt construction  
3. Voter prediction generation  
4. Voting aggregation  

This design allows controlled analysis of:
- number of voters (n),
- majority threshold (k),
- voter diversity,
- robustness and performance trade-offs.

---

## Repository Structure

```
VoteRE/
├── Data/
│   ├── 0_raw/
│   ├── 1_processed/
│   ├── 2_prompts/
│   ├── 3_Voters_Predictions/
│   └── 4_VoteRE/
├── LLM_Models/
├── Logs/
├── llm_inference_only.py
├── process_raw_to_csv.py
├── processed_to_prompt.py
├── scorer.py
├── templates.py
├── vote_re.py
├── VoteRE.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline

### 1. Process Raw Data

```bash
python process_raw_to_csv.py \
    --input_dir "Data/0_raw/TACRED" \
    --output_dir "Data/1_processed/TACRED"
```

---

### 2. Create Prompts

```bash
python processed_to_prompt.py \
    --input_dir "Data/1_processed/TACRED" \
    --output_dir "Data/2_prompts/TACRED" \
    --dataset_name TACRED
```

---

### 3. Run LLM Inference

```bash
python llm_inference_only.py \
    --prompts_dir Data/2_prompts/TACRED \
    --processed_dir Data/1_processed/TACRED \
    --output_dir Data/3_Voters_Predictions/QWEN/TACRED \
    --model_path LLM_Models/QWEN_TACRED/checkpoint-merged
```

---

### 4. Run VoteRE

```bash
python vote_re.py \
    --dataset TACRED \
    --split test \
    --voters LORA_QWEN LORA_LLAMA LORA_MISTRAL \
    --k 2 \
    --output_csv Data/4_VoteRE/TACRED/test_llm_vote_k2.csv
```

---


## Output Format

CSV output includes:
- id
- sentence
- subject
- object
- gold relation
- predictions per voter
- final voted relation


---

## Contact

For questions or collaboration, contact the repository author.
