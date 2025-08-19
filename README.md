# 💊 Rx Intel – Intelligent Drug Report Generation using NLP  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Status](https://img.shields.io/badge/Status-Research%20Project-green)  

---

## 📖 Overview  
Clinical trial drug reports often suffer from **inconsistency, lack of structure, and low standardization**, which reduces their usability in healthcare and research.  

**Rx Intel** is an **AI-driven system** that leverages **Natural Language Processing (NLP)** and **fine-tuned biomedical Large Language Models (LLMs)** to automatically generate **structured, standardized, and protocol-compliant drug reports**.  

This project bridges the gap between **raw clinical trial data** and **readable, structured medical documents**, ultimately improving **efficiency, accuracy, and accessibility** for:  
- 🩺 Medical professionals  
- 🔬 Researchers  
- 📊 Regulatory authorities  

---

## 🎯 Problem Statement  
Traditional drug reports generated from clinical trials are:  
- ❌ Unstructured and inconsistent  
- ❌ Difficult to standardize across medical domains  
- ❌ Hard to integrate into downstream systems like EMRs  

This project provides a **solution** by:  
✔️ Automatically generating structured reports compliant with standards like **ICH-M11** and **SPIRIT checklists**  
✔️ Using **biomedical LLMs fine-tuned with LoRA adapters**  
✔️ Performing **semantic mapping** of conditions, drugs, and outcomes for reliability  

---

## 🚀 Key Features  
- 📑 **Automated Report Generation** – Converts raw trial data into structured protocol reports  
- 🧠 **Biomedical LLMs** – Uses domain-specialized models (BioGPT, BioMedLM, LLaMA-2)  
- ⚡ **Parameter-Efficient Fine-Tuning (PEFT)** – LoRA adapters for resource efficiency  
- ✅ **Compliance-Ready Templates** – Aligns with ICH-M11 & SPIRIT clinical standards  
- 🛡️ **Quality Assurance Controls** – Checks for dosage consistency, unit validation, hallucination detection  
- 📊 **Evaluation Metrics** – ROUGE, BLEU, UniEval, and expert reviews  
- 🌍 **Scalability** – Designed for large-scale processing of clinical trial repositories  

---

## 🏗️ System Architecture  

[1] Data Ingestion & Preprocessing
↓
[2] Ontology & Semantic Mapping (MeSH, DrugBank, RxNorm)
↓
[3] JSON Schema Generation (protocol template)
↓
[4] Prompt–Response Pair Creation
↓
[5] Transformer Model (Frozen) + LoRA Fine-Tuning
↓
[6] Validation & Metrics Evaluation
↓
[7] Post-Processing & Quality Assurance
↓
[8] Deployment via API
↓
[9] Monitoring, Logging & Compliance


---

## 🛠️ Tech Stack  

**Languages & Frameworks**  
- Python 3.10+  
- PyTorch, Hugging Face Transformers  
- FastAPI / Flask for deployment  

**Libraries & Tools**  
- PEFT (LoRA Adapters)  
- spaCy, NLTK, Scikit-learn  
- JSON Schema, Pandas, NumPy  

**Datasets**  
- [ClinicalTrials.gov](https://clinicaltrials.gov/)  
- [DrugBank](https://go.drugbank.com/)  
- [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/)  

---

## 📊 Results & Impact  
- ✅ Increased **report readability & usability**  
- ✅ Ensured compliance with **ICH-M11 / SPIRIT standards**  
- ✅ Enhanced **data reliability** for clinical applications  
- ✅ Scalable framework for **large clinical datasets**  
- ✅ Contributes to **faster drug research & approval cycles**  

---

## 📥 Installation  

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## ▶️ Usage
1️⃣ Preprocess Clinical Trial Data
``` python preprocess.py --input raw_data.json --output clean_data.json```

2️⃣ Train with LoRA Fine-Tuning
```python train.py --data clean_data.json --epochs 5```

3️⃣ Generate Drug Protocol Report
```python generate_report.py --input trial.json --output report.json```


Example Output:
```
{
  "drug_name": "Atorvastatin",
  "condition": "Hypercholesterolemia",
  "dosage": "20mg/day",
  "outcomes": "Reduced LDL cholesterol by 45%",
  "phase": "Phase III",
  "sponsor": "Pfizer"
}
```
📂 Repository Structure
├── data/                 # Sample clinical trial datasets  
├── models/               # Fine-tuned model checkpoints  
├── scripts/              # Training & preprocessing scripts  
├── outputs/              # Generated reports  
├── preprocess.py         # Data cleaning & normalization  
├── train.py              # Fine-tuning script with LoRA  
├── generate_report.py    # Report generation tool  
├── requirements.txt      # Python dependencies  
└── README.md             # Documentation  

## ✅ Future Enhancements  

🌍 Multi-lingual clinical report generation  
📊 Interactive dashboard for visualization  
🔍 Explainable AI for better interpretability  
🏥 Direct integration with Electronic Medical Records (EMR) systems  


## 👨‍💻 Contributors  

Sankalp Sathe 
Satish Singh 
Aakash Shedge 
Pranavi Shukla   

**Mentor:** Dr. Ekta Sarda  
