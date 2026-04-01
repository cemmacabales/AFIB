# AFIB Clinical Decision Support — RAG System

A **Retrieval-Augmented Generation (RAG)** system that enables medical professionals to query the **2024 European Society of Cardiology (ESC) Guidelines on Atrial Fibrillation (AF)** using natural language. The system returns accurate, context-grounded answers with source page citations and real-time evaluation metrics.

---

## Overview

Manually searching through hundreds of pages of dense clinical guidelines is time-consuming and error-prone. This project automates that process by combining biomedical embeddings, vector search, cross-encoder reranking, and state-of-the-art language models into an interactive web application.

**Clinical areas covered:**
- AF diagnosis and classification
- Treatment recommendations (rate and rhythm control)
- Anticoagulation strategies and dosing
- Risk stratification (CHA₂DS₂-VASc, HAS-BLED scores)
- Drug interactions and contraindications
- Patient management algorithms (AF-CARE framework)

---

## Features

- **Three LLM backends** — Phi-3 (3.8B), Llama 3 (8B), and Qwen (4B) for performance comparison
- **Biomedical-optimised embeddings** — MedCPT for domain-specific semantic retrieval
- **Cross-encoder reranking** — improves precision of retrieved context
- **4-bit NF4 quantization** — runs on a single consumer-grade GPU (~10 GB VRAM)
- **Table-aware PDF parsing** — preserves structured data from the ESC guidelines
- **Gradio web interface** — streaming responses with real-time pipeline status
- **Source attribution** — every answer is linked to the originating page(s)
- **Comprehensive evaluation** — ROUGE, BERTScore, semantic similarity, and LLM-as-a-judge metrics

---

## Repository Structure

```
AFIB/
├── FINAL NOTEBOOKS/                              # Production-ready notebooks
│   ├── vF_WebDeployment_Phi3_Formatted.ipynb     # Phi-3 implementation
│   ├── vF_WebDeployment_Llama3_Formatted.ipynb   # Llama 3 implementation
│   └── vF_WebDeployment_Qwen_Formatted.ipynb     # Qwen implementation
├── TEST NOTEBOOKS/                               # Development / evaluation notebooks
│   ├── Phi3_w_Metrics_(1).ipynb
│   ├── Llama3_w_Metrics (1).ipynb
│   └── Qwen_w_Metrics.ipynb
├── CSV METRICS/                                  # Per-query evaluation results (20 queries each)
│   ├── PHI.csv
│   ├── LLAMA.csv
│   └── QWEN.csv
├── test prompts images/                          # Screenshot outputs for each model
│   ├── phi/
│   ├── llama/
│   └── qwen/
├── 2024ESC-compressed.pdf                        # Source document: 2024 ESC AF Guidelines
├── AFIB AMM.pdf                                  # Project methodology & technical report (PDF)
└── AFIB AMM.docx                                 # Project methodology & technical report (Word)
```

---

## Technologies

| Category | Libraries / Tools |
|---|---|
| Language | Python 3 |
| Notebooks | Jupyter Notebook (`.ipynb`) |
| LLM Inference | HuggingFace Transformers, PyTorch |
| Quantization | BitsAndBytes (4-bit NF4) |
| RAG Framework | LangChain, LangChain-Community, LangChain-HuggingFace |
| Embeddings | Sentence-Transformers, MedCPT, BiomedBERT |
| Vector Store | FAISS (CPU) |
| PDF Parsing | pdfplumber |
| Reranking | Cross-Encoder (ms-marco-MiniLM / bge-reranker) |
| Web UI | Gradio |
| Evaluation | ROUGE-score, BERTScore, NLTK, Semantic Similarity |
| Data | Pandas, NumPy |

---

## Setup & Installation

### Requirements

- Python 3.9+
- GPU with ~10 GB VRAM (Google Colab T4 tested) **or** a CPU-only setup (slower)

### Install dependencies

```bash
pip install -q -U "transformers>=4.41.2" accelerate bitsandbytes \
    langchain langchain-community langchain-huggingface \
    faiss-cpu sentence-transformers pdfplumber \
    gradio rouge-score bert-score nltk pandas torch
```

---

## Running the Application

### Option A — Google Colab (recommended)

1. Open one of the production notebooks in `FINAL NOTEBOOKS/` in [Google Colab](https://colab.research.google.com/).
2. Upload `2024ESC-compressed.pdf` to the Colab file system.
3. Set the runtime to **GPU** (T4 or better).
4. Run all cells sequentially.
5. When the Gradio cell executes, a public URL will be printed — open it in your browser.

### Option B — Local machine

```bash
# Clone the repo
git clone https://github.com/cemmacabales/AFIB.git
cd AFIB

# Install Jupyter
pip install jupyter

# Launch the notebook of your choice
jupyter notebook "FINAL NOTEBOOKS/vF_WebDeployment_Phi3_Formatted.ipynb"
```

Run all cells in order. The Gradio interface will open automatically (or print a local URL).

---

## Pipeline Walkthrough

| Step | What happens |
|---|---|
| **1. PDF ingestion** | `pdfplumber` extracts text and tables page-by-page; output saved to `extracted_pdf_data.txt` |
| **2. Chunking & embedding** | Text split with overlap; ~1,400 chunks embedded using MedCPT into a FAISS index |
| **3. Model loading** | LLM loaded with 4-bit NF4 quantization; cross-encoder reranker and similarity model initialised |
| **4. Query processing** | Top-k chunks retrieved from FAISS → reranked → passed as context to the LLM → answer streamed |
| **5. Evaluation** | Response scored with ROUGE, BERTScore, semantic similarity, and LLM-as-a-judge (faithfulness, relevancy, context recall) |
| **6. Web interface** | Gradio app serves the full pipeline; users see the answer, source pages, and live metrics |

---

## Example Queries

```
What are the four essential treatment pillars of the AF-CARE framework?
When is oral anticoagulation recommended based on the CHA₂DS₂-VASc score?
Which drugs are recommended as first-choice for rate control in patients with LVEF > 40%?
Is antiplatelet therapy recommended as an alternative to OAC for stroke prevention in AF?
What is the recommended target heart rate for rate control in AF patients?
```

---

## Model Comparison

Three LLM variants were evaluated on **20 unseen clinical queries**:

| Metric | Phi-3 (3.8B) | Llama 3 (8B) | Qwen (4B) |
|---|---|---|---|
| Semantic Similarity (avg) | 0.628 | **0.665** | 0.616 |
| BERTScore F1 (avg) | 0.83 | — | — |
| Faithfulness (avg, /10) | 9.3 | — | — |
| Answer Relevancy (avg, /10) | 9.8 | — | — |
| Context Recall (avg, /10) | 9.1 | — | — |
| Avg. Latency | ~11.5 s | — | — |

Full per-query results are available in `CSV METRICS/`.

| Model | Reranker |
|---|---|
| Phi-3 | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Llama 3 | `BAAI/bge-reranker-base` |
| Qwen | `BAAI/bge-reranker-base` |

---

## Documentation

- **`AFIB AMM.pdf` / `AFIB AMM.docx`** — 8-page technical report covering the project methodology, architecture diagrams, and results summary.
- Detailed markdown cells within each notebook explain every stage of the pipeline.

---

## License

This project is for academic and research purposes. The 2024 ESC Guidelines PDF is the intellectual property of the European Society of Cardiology.
