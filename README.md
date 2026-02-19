**Project Overview**
This repository contains the source code for the study: Tabular Machine Learning vs. Retrieval-Augmented Generation for High-Frequency Fraud Detection: A Benchmark Study. We investigate the "Modality Mismatch" and "Label Blindness" that occur when applying Generative AI to high-precision numerical transaction data.
Repository Structure
•	**champion_ensemble/:** Implementation of the GBDT Hybrid Ensemble (XGBoost, LightGBM, Random Forest).
•	**challenger_rag/:** Contextual RAG architecture using FinBERT embeddings and Llama-3.1-8B-Instruct.
•	**ablation_study/:** Scripts for the 7-variant diagnostic ablation.
•	data_preprocessing/: Temporal partitioning and velocity feature engineering.
**1. Data and Code Availability**
•	Dataset: The study utilizes the IEEE-CIS Fraud Detection dataset, comprising 590,540 real-world e-commerce transactions. It is publicly available on .
•	**Code:** All model training, evaluation, and latency benchmarking scripts are provided under the MIT license.
•	**Reproducibility:** A global random seed (1337) is enforced to ensure deterministic results across data splits and model initializations.
2. Technical Environment
To replicate the results, particularly the 121,045× latency penalty, the following environment is required:
•	**Hardware:** NVIDIA A100-SXM4-80GB.
•	**CUDA Version:** 11.8.
•	**Quantization:** Llama-3.1-8B is loaded in 4-bit NormalFloat (NF4) using bitsandbytes.
•	Retrieval Backend: FAISS with HNSW ($M=32$, $efConstruction=200$, $efSearch=64$).
3. **Key Hyperparameters**
4. Chain-of-Thought (CoT) Prompt Template
The Challenger utilizes a three-step reasoning chain to analyze transactions:
1.	**Count & Profile:** Identify neighbors sharing card/address/email patterns.
2.	**Mismatch Analysis:** Compare TransactionAmt and velocity against historical averages.
3.	**Calculation:** Assign a fraud probability score between 0.0 and 1.0.

