"""
MedKnow-GEN — Streamlit Web Interface
Cancer Knowledge Graph QA System
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from module3_rag_qa.qa_pipeline import MedKnowQA


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedKnow-GEN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #f0f4ff, #fafaff);
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        color: #1a1a2e;
    }
    .source-item {
        background: #f8f9ff;
        border: 1px solid #e0e4ff;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #444;
    }
    .cypher-box {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.88rem;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "connected" not in st.session_state:
    st.session_state.connected = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = True

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.markdown("## ⚙️ Configuration")
    st.divider()

    neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    neo4j_user = st.text_input("Neo4j Username", value="neo4j")
    neo4j_pass = st.text_input("Neo4j Password", type="password", value="password")

    st.divider()

    llm_backend = st.radio(
        "LLM Backend",
        ["Ollama (Local — Llama3)", "OpenAI GPT", "Rule-Based (No LLM)"],
        index=0,
    )

    if llm_backend == "OpenAI GPT":
        openai_key = st.text_input("OpenAI API Key", type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    st.divider()
    st.session_state.debug_mode = st.checkbox("Show Debug Output", value=True)

    st.divider()

    if st.button("🔌 Connect to Neo4j"):
        with st.spinner("Connecting..."):
            try:
                os.environ["NEO4J_URI"] = neo4j_uri
                os.environ["NEO4J_USERNAME"] = neo4j_user
                os.environ["NEO4J_PASSWORD"] = neo4j_pass

                st.session_state.qa_system = MedKnowQA(debug=st.session_state.debug_mode)
                st.session_state.connected = True
                st.success("✅ Connected!")

            except Exception as e:
                st.session_state.connected = False
                st.session_state.qa_system = None
                st.error(f"❌ Connection failed: {e}")

    if st.session_state.connected:
        st.success("🟢 Neo4j Connected")
    else:
        st.warning("🔴 Not Connected")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.session_state.question_input = ""
        st.rerun()

    st.divider()
    st.markdown("**About**")
    st.caption(
        "MedKnow-GEN is a generative agent for dynamic "
        "cancer knowledge graph construction and QA.\n\n"
        "Rajalakshmi Engineering College | AI & ML Dept"
    )


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧬 MedKnow-GEN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'A Generative Agent for Dynamic Cancer Knowledge Graph Construction'
    '</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Ask a Question", "🔍 Explore Graph", "📊 About"])


# ── Tab 1: QA ─────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Ask a cancer-related question")

    st.markdown("**💡 Try these:**")
    col1, col2, col3, col4 = st.columns(4)
    suggested = [
        "Which drug treats Lung Cancer?",
        "Which gene is linked to Colorectal Cancer?",
        "What does Cisplatin target?",
        "Which drug targets EGFR?",
    ]

    for col, q in zip([col1, col2, col3, col4], suggested):
        if col.button(q, key=f"sugg_{q}"):
            st.session_state.question_input = q

    st.divider()

    st.text_input(
        "Your question:",
        key="question_input",
        placeholder="e.g. Which drugs treat Lung Cancer?",
    )

    ask_col, _ = st.columns([1, 3])
    ask_clicked = ask_col.button("🔎 Ask MedKnow-GEN")

    if ask_clicked:
        question = st.session_state.question_input.strip()

        if not question:
            st.warning("⚠️ Please enter a question.")
        elif not st.session_state.connected or st.session_state.qa_system is None:
            st.warning("⚠️ Please connect to Neo4j first using the sidebar.")
        else:
            with st.spinner("Searching knowledge graph..."):
                try:
                    result = st.session_state.qa_system.ask(question)

                    if st.session_state.debug_mode:
                        st.write("DEBUG RESULT:")
                        st.write(result)

                    st.session_state.last_result = result
                    st.session_state.chat_history.append(result)

                except Exception as e:
                    error_result = {
                        "question": question,
                        "answer": "The frontend failed while processing the request.",
                        "sources": [],
                        "context": "",
                        "success": False,
                        "cypher": "",
                        "raw_results": [],
                        "error": str(e),
                    }
                    st.session_state.last_result = error_result
                    st.session_state.chat_history.append(error_result)

    if st.session_state.last_result:
        latest = st.session_state.last_result

        st.markdown("#### 🤖 Answer")
        st.markdown(
            f'<div class="answer-box">{latest.get("answer", "No answer returned.")}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("🔧 View Generated Cypher Query"):
            st.code(latest.get("cypher", "No Cypher generated"), language="cypher")

        if latest.get("sources"):
            st.markdown("#### 📚 Supporting Evidence from Knowledge Graph")
            for src in latest.get("sources", []):
                st.markdown(
                    f'<div class="source-item">🔗 {src}</div>',
                    unsafe_allow_html=True,
                )

        if latest.get("error"):
            st.error(latest.get("error"))

        if st.session_state.debug_mode:
            with st.expander("🐞 Debug Details"):
                st.write("Success:", latest.get("success"))
                st.write("Question:", latest.get("question"))
                st.write("Context:", latest.get("context"))
                st.write("Raw Results:", latest.get("raw_results"))

        st.divider()

    if len(st.session_state.chat_history) > 1:
        with st.expander(f"📜 Chat History ({len(st.session_state.chat_history)} questions)"):
            for i, item in enumerate(reversed(st.session_state.chat_history[:-1])):
                st.markdown(f"**Q{len(st.session_state.chat_history)-i-1}:** {item.get('question', '')}")
                st.markdown(f"**A:** {item.get('answer', 'No answer')}")
                st.divider()


# ── Tab 2: Graph Explorer ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Explore the Cancer Knowledge Graph")

    col_a, col_b = st.columns([2, 1])
    entity_input = col_a.text_input(
        "Enter an entity to explore:",
        placeholder="e.g. Lung Cancer, Gefitinib, TP53",
    )
    entity_type = col_b.selectbox("Type", ["Any", "Disease", "Drug", "Gene"])

    if st.button("🔍 Explore Entity"):
        if not st.session_state.connected or st.session_state.qa_system is None:
            st.warning("⚠️ Please connect to Neo4j first.")
        elif entity_input.strip():
            with st.spinner("Fetching relationships..."):
                try:
                    retriever = st.session_state.qa_system.retriever
                    results = retriever.retrieve_subgraph(entity_input)

                    if results:
                        st.markdown(f"#### Relationships for **{entity_input}**")
                        for r in results:
                            head = r.get("head", "")
                            relation = r.get("relation", "")
                            tail = r.get("tail", "")

                            color_map = {
                                "TREATS": "#4f46e5",
                                "TARGETS": "#059669",
                                "ASSOCIATED_WITH": "#d97706",
                                "INHIBITS": "#dc2626",
                                "CAUSES": "#7c3aed",
                            }
                            color = color_map.get(relation, "#6b7280")
                            st.markdown(
                                f"**{head}** "
                                f'<span style="color:{color}; font-weight:600;">─[{relation}]→</span> '
                                f"**{tail}**",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info(f"No relationships found for '{entity_input}'.")

                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown("#### 🔢 Quick Stats")

    if st.session_state.connected and st.session_state.qa_system is not None:
        try:
            retriever = st.session_state.qa_system.retriever

            node_count = retriever.execute_cypher(
                "MATCH (n) RETURN count(n) AS count"
            )
            rel_count = retriever.execute_cypher(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            )
            drug_count = retriever.execute_cypher(
                "MATCH (n:Drug) RETURN count(n) AS count"
            )
            gene_count = retriever.execute_cypher(
                "MATCH (n:Gene) RETURN count(n) AS count"
            )
            dis_count = retriever.execute_cypher(
                "MATCH (n:Disease) RETURN count(n) AS count"
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Nodes", node_count[0]["count"] if node_count else 0)
            c2.metric("Relationships", rel_count[0]["count"] if rel_count else 0)
            c3.metric("Drugs", drug_count[0]["count"] if drug_count else 0)
            c4.metric("Genes", gene_count[0]["count"] if gene_count else 0)
            c5.metric("Diseases", dis_count[0]["count"] if dis_count else 0)

        except Exception as e:
            st.warning(f"Could not fetch stats: {e}")
    else:
        st.info("Connect to Neo4j to see graph statistics.")


# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### About MedKnow-GEN")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Project:** MedKnow-GEN: A Generative Agent for Dynamic Cancer
Knowledge Graph Construction

**Institution:** Rajalakshmi Engineering College
**Department:** Artificial Intelligence & Machine Learning
**Course:** AI19711 – Phase I Project

**Team:**
- V Poonguzhali (221501512)
- M Aparna (221501901)

**Mentor:** Dr. S Poonkuzhali
        """)

    with col2:
        st.markdown("""
**Tech Stack:**

| Layer | Tools |
|---|---|
| NER | BioBERT, PubMedBERT |
| Entity Linking | Sentence-BERT, FAISS |
| Relation Extraction | BioLinkBERT |
| KG Storage | Neo4j |
| KG Embeddings | TransE, RotatE |
| RAG & QA | LangChain, FLAN-T5 |
| LLM | Llama3 / Mistral |
| Frontend | Streamlit |
        """)

    st.divider()
    st.markdown("""
**Modules:**
1. **Module 1** — Multi-Modal Entity Recognition & Linking
2. **Module 2** — Dynamic Knowledge Graph Construction
3. **Module 3** — Retrieval-Augmented Generative QA System
    """)