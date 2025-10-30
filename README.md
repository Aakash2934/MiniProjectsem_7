
# 🧬 Clinical Trial Semantic Search Engine

## 📖 Overview

Finding relevant clinical trials is a critical but incredibly difficult task. The eligibility criteria for trials are often written in dense, unstructured medical text, making it nearly impossible to search for them based on specific patient attributes or the semantic meaning of a query.

This project solves this problem by building a complete, end-to-end **Natural Language Processing (NLP)** pipeline.  
The system ingests over **3.5 million unstructured clinical trial criteria**, uses a **custom-trained BioBERT model** to understand and extract key information, and deploys this knowledge into a **high-speed semantic search engine**.

This engine allows researchers, doctors, and patients to find relevant clinical trials not by just matching keywords, but by understanding the **intent and meaning** of their search query.

> This project bridges the gap between massive, unstructured text data and a queryable, structured knowledge base.

---

## 🎯 Problem Statement

Clinical trial databases (like ClinicalTrials.gov) are vast, but their search functionality is often limited to keyword matching.

### ❌ Challenges
- **Unstructured Data:** Eligibility criteria are free-text blocks (e.g., "patient must be over 18 and have no history of NSCLC").
- **Keyword Mismatch:** A search for "lung cancer" might miss "non-small cell lung cancer" or "NSCLC".
- **Scalability:** Impossible for humans to manually read and categorize millions of criteria.

### ✔️ Our Solution
- Automatically structure **3.5M+ criteria** using a custom-trained **NER model**.
- Extract **7 key entity types:**  
  `CONDITION`, `DRUG`, `LAB_TEST`, `VALUE`, `OPERATOR`, `PROCEDURE`, and `DEMOGRAPHIC`.
- Build a **semantic index** that allows users to search based on meaning and intent.

---

## 🚀 Key Features

- **📑 Automated Data Pipeline (Phase I)**  
  Ingests and caches the dataset using `spaCy` for efficient segmentation of 3.5M+ text blocks.

- **🧠 Custom AI Model (Phase II)**  
  Fine-tuned **BioBERT (dmis-lab/biobert-base-cased-v1.1)** for 7 medical entity types.

- **⚡ Large-Scale Inference (Phase III)**  
  Multiprocessing optimization reduces processing time from 270+ hours to a few hours.

- **🔗 Rule-Based Structuring (Phase III)**  
  Uses `spaCy` dependency parsing to connect entities (e.g., LAB_TEST → VALUE).

- **🔍 Semantic Search (Phase IV)**  
  Uses **all-MiniLM-L6-v2 Sentence Transformer** for 384-dimensional embeddings.

- **🗄️ High-Speed Vector Database (Phase IV)**  
  Employs **ChromaDB** for fast semantic similarity search.

- **🖥️ Interactive API (Phase V)**  
  A FastAPI-powered REST API exposes the `/search/` endpoint for querying trials.

---

## 🏗️ System Architecture — The 5-Phase Pipeline

### **Phase I: Data Ingestion & Preprocessing**
- **Dataset:** `louisbrulenaudet/clinical-trials`
- **Library:** `datasets`
- **Process:** Clean, filter, and segment text using `spaCy`’s `nlp.pipe()`.
- **Output:**  
  - `train.csv`  
  - `test.csv`  
  (3.5M+ rows of segmented eligibility criteria)

---

### **Phase II: The Core Intelligence Engine (NER Model)**
1. **Sampling:** Select 1,000 unique criteria for manual annotation.  
2. **Annotation:** Create gold-standard `all.jsonl` file with labeled entities.  
3. **Data Prep:** Convert to IOB2-tagged “flashcards” format.  
4. **Training:** Fine-tune BioBERT using Cross-Entropy Loss.  

#### 📘 Key Equations
**Cross-Entropy Loss:**  
\[
Loss = -\sum_i y_i \log(\hat{y}_i)
\]

**F1-Score:**  
\[
F1 = 2 \times \frac{P \times R}{P + R}
\]

#### 📦 Output:
Custom NER model → `models/clinical-ner-model/`

---

### **Phase III: Structuring & Normalization**
- **Inference:** Process all 3.5M+ criteria using 12-core multiprocessing.
- **Relation Extraction:** Connect entities via dependency parsing.
- **Normalization:** Map entity text to **UMLS codes**.

**Output:** `train_structured_knowledge.jsonl`

---

### **Phase IV: Vectorization & Indexing**
- **Model:** `all-MiniLM-L6-v2`
- **Process:** Convert each sentence into a 384-dimensional semantic vector.
- **Similarity Metric:** **Cosine Similarity**
  \[
  Similarity = \frac{V_q \cdot V_d}{||V_q|| \times ||V_d||}
  \]

**Output:** `vector_db/` (ChromaDB persistent index)

---

### **Phase V: Deployment via API**
- **Framework:** FastAPI + Uvicorn
- **Endpoint:** `/search/`
- **Functionality:**
  1. Converts query into vector (`V_q`)
  2. Computes cosine similarity with stored vectors
  3. Returns top 5 relevant trials

**Access:**  
`http://127.0.0.1:8000/docs`

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.11+ |
| **DL Framework** | PyTorch |
| **API Framework** | FastAPI + Uvicorn |
| **NLP Libraries** | Hugging Face Transformers, Datasets, spaCy |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Database** | ChromaDB |
| **Utilities** | pandas, tqdm, seqeval |
| **Visualization** | matplotlib, seaborn |

---

## 🧩 Key Models

| Model Type | Name | Purpose |
|-------------|------|----------|
| **NER Model** | dmis-lab/biobert-base-cased-v1.1 | Extract 7 biomedical entities |
| **Vector Model** | all-MiniLM-L6-v2 | Convert text into semantic embeddings |

---

## 📂 Repository Structure

```

clinical_trials_ner/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample.txt
│   │   ├── all.jsonl
│   │   ├── train_with_entities.jsonl
│   │   ├── test_with_entities.jsonl
│   │   └── train_structured_knowledge.jsonl
│   └── ner_dataset/
│
├── scripts/
│   ├── ingest_data.py
│   ├── segmentation.py
│   ├── preprocess_data.py
│   ├── create_annotation_sample.py
│   ├── prepare_ner_data.py
│   ├── train_ner_model.py
│   ├── test_ner_model.py
│   ├── apply_ner_model.py
│   ├── structure_and_normalize.py
│   ├── visualize_results.py
│   ├── vectorize_and_index.py
│   └── api.py
│
├── models/
│   └── clinical-ner-model/
│
├── vector_db/
│
└── plots/
├── 01_training_loss_curve.png
├── 02_final_evaluation_metrics.png
└── 03_entity_distribution.png

````

---

## ▶️ Usage: Running the Full Pipeline

### 🧰 Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/clinical_trials_ner.git
cd clinical_trials_ner

# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
````

---

### 🏁 Run the Pipeline

#### **Phase I: Data Preprocessing**

```bash
python scripts/01_preprocess_data.py
```

#### **Phase II: Model Training**

```bash
python scripts/02_create_annotation_sample.py
# (Annotate 'sample.txt' manually and save as 'all.jsonl')
python scripts/03_prepare_ner_data.py
python scripts/04_train_ner_model.py
python scripts/05_test_ner_model.py
```

#### **Phase III & IV: Processing and Indexing**

```bash
python scripts/06_apply_ner_model.py
python scripts/07_structure_and_normalize.py
python scripts/08_visualize_results.py
python scripts/09_vectorize_and_index.py
```

#### **Phase V: API Deployment**

```bash
python scripts/10_api.py
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and test queries like:

* “patients with non-small cell lung cancer”
* “adults over 50 with diabetes”

---

## ✅ Future Enhancements

* 🌍 **Multi-lingual Support** for global trial databases.
* 📊 **Interactive Dashboard** (Streamlit/React UI).
* 🏥 **EMR Integration** for automatic patient-trial matching.
* 🧩 **Relation Extraction Model** (beyond rule-based parsing).

---

## 👨‍💻 Contributors

* **Sankalp Sathe**
* **Satish Singh**
* **Aakash Shedge**
* **Pranavi Shukla**

**Mentor:** *Dr. Ekta Sarda*