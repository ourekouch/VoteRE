# Data Structure

This directory contains the full VoteRE data pipeline.

## Structure

```

Data/
├── 0_raw/
├── 1_processed/
├── 2_prompts/
├── 3_Voters_Predictions/
└── 4_VoteRE/
```

---

## Description

### 0_raw
Original dataset files (JSON format).

### 1_processed
Converted CSV files used for processing.

### 2_prompts
Prompt JSON files used for LLM inference.

### 3_Voters_Predictions
Predictions from each voter (LLMs or PLMs).

### 4_VoteRE
Final outputs after voting aggregation.

---

## Data Flow

0_raw → 1_processed → 2_prompts → 3_Voters_Predictions → 4_VoteRE


---

## Data Flow

0_raw → 1_processed → 2_prompts → 3_Voters_Predictions → 4_VoteRE

## Datasets for 0_raw

The raw datasets used in this project are standard benchmarks for relation extraction.  
They must be downloaded separately and placed in the `Data/0_raw/` directory following the expected structure.

- **TACRED**  
  Available from the original release by Zhang et al. (2017):  
  https://nlp.stanford.edu/projects/tacred/

- **TACREV (TACRED-Revisited)**  
  A corrected version of TACRED with improved label quality:  
  https://github.com/DFKI-NLP/tacrev

- **ReTACRED**  
  A refined version of TACRED with systematic relabeling:  
  https://github.com/gstoica27/Re-TACRED
