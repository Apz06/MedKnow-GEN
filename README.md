# 🧬 MedKnow-GEN
### A Generative Agent for Dynamic Cancer Knowledge Graph Construction

> **Rajalakshmi Engineering College** | Dept. of AI & ML | AI19711 – Phase I Project  
> **Team:** V Poonguzhali (221501512) · M Aparna (221501901)  
> **Mentor:** Dr. S Poonkuzhali

---

## 🔍 Overview

MedKnow-GEN is an intelligent biomedical system that converts unstructured cancer
literature into a structured, queryable knowledge graph. It supports semantic search,
link prediction, and natural language question answering over Drug–Gene–Disease
relationships.

---

## 🏗️ Project Structure
```
MedKnow-GEN/
├── module1_ner_linking/       # BioBERT/PubMedBERT NER + FAISS entity linking
├── module2_kg_construction/   # Relation extraction, Neo4j, TransE embeddings
├── module3_rag_qa/            # RAG pipeline, Cypher generation, FLAN-T5 QA
├── data/
│   ├── raw/                   # PubMed abstracts, BC5CDR corpus (not in Git)
│   └── processed/             # Linked entities, triplets (not in Git)
├── notebooks/                 # Jupyter experiments
├── utils/                     # Shared config, tokenizer helpers
├── frontend/                  # Streamlit UI
├── .env.example               # Rename to .env and fill secrets
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| NER & Entity Linking | BioBERT, PubMedBERT, Sentence-BERT, FAISS |
| Relation Extraction | BioLinkBERT, SciBERT, softmax classifier |
| KG Storage | Neo4j, Cypher |
| KG Embeddings | TransE / RotatE / ComplEx (PyKEEN) |
| RAG & QA | LangChain, Llama3 / Mistral, FLAN-T5 |
| Frontend | Streamlit, FastAPI |

---

## 🚀 Setup
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/MedKnow-GEN.git
cd MedKnow-GEN

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your Neo4j credentials and API keys

# 5. Start Neo4j (via Docker)
docker run -p 7474:7474 -p 7687:7687 neo4j:latest
```

---

## 📦 Data Sources

- **BC5CDR corpus** — BioBERT NER fine-tuning (chemicals & diseases)
- **PubMed abstracts** — via Biopython Entrez API
- **NCBI Gene database** — entity linking for genes
- **DrugBank** — entity linking for drugs

---

## 📌 Modules

### Module 1 — Multi-Modal Entity Recognition & Linking
Fine-tuned BioBERT/PubMedBERT on BC5CDR for NER. Two-stage entity linking:
Sentence-BERT + FAISS for candidate generation → cross-encoder re-ranking.

### Module 2 — Dynamic Knowledge Graph Construction
BioLinkBERT extracts (head, relation, tail) triplets. Neo4j stores the graph.
TransE embeddings trained via PyKEEN for link prediction and scoring.

### Module 3 — RAG QA & Summarization
Natural language → Cypher via Llama3/Mistral. Subgraph retrieval from Neo4j.
FLAN-T5 generates grounded answers. Hybrid extractive-abstractive summarization.