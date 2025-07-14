import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ───────────────────────────────────────────────────────────────────────
# 0️⃣  Environment & API setup
# ───────────────────────────────────────────────────────────────────────
load_dotenv()                                    # reads .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ───────────────────────────────────────────────────────────────────────
# 1️⃣  Helper – custom CSS to make the UI look modern & responsive
# ───────────────────────────────────────────────────────────────────────
def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            /* ===== base page & typography ===== */
            html, body, [class*="css"]{
                font-family: 'Inter',sans-serif;
                background:#f8fafc;
            }
            /* widen content area on desktop */
            @media (min-width:768px){
                .main .block-container{max-width:1200px;padding:2rem 4rem;}
            }
            /* ===== header ===== */
            .main h2{color:#2563eb;font-weight:700;}
            /* ===== text input ===== */
            .stTextInput>div>div>input{
                border-radius:8px;border:1px solid #cbd5e1;
                padding:0.5rem 0.75rem;font-size:16px;
            }
            /* ===== buttons ===== */
            .stButton>button{
                background:#2563eb;color:#fff;
                padding:0.55rem 1.25rem;border-radius:8px;border:none;
                transition:background .25s;
            }
            .stButton>button:hover{background:#1e40af;}
            /* ===== file uploader ===== */
            .stFileUploadDropzone{
                border:2px dashed #60a5fa !important;
                background:#fff !important;
                border-radius:10px;padding:1.25rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────────
# 2️⃣  Core backend functions (unchanged logic)
# ───────────────────────────────────────────────────────────────────────
def get_pdf_text(pdf_docs) -> str:
    """Extract raw text from a list of uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=1_000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context; if the answer is not
    in the context say "answer is not available in the context" – do NOT hallucinate.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,  # FAISS 1.8+ safety gate
    )
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question},
                     return_only_outputs=True)
    st.write("**Reply:**", response["output_text"])


# ───────────────────────────────────────────────────────────────────────
# 3️⃣  Streamlit front‑end (responsive two‑column layout)
# ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="PDFMindBot", page_icon="💁", layout="wide")
    apply_custom_css()

    st.markdown("## 💁‍♂️ Chat with **PDFMindBot**")

    # Responsive layout: chat area (75 %)  |  control panel (25 %)
    chat_col, control_col = st.columns([3, 1], gap="large")

    # ----- control panel -----
    with control_col:
        st.markdown("### 📂 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("🚀 Submit & Process") and pdf_docs:
            with st.spinner("⏳ Reading & indexing…"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Ready! Ask away on the left ➡️")

    # ----- chat area -----
    with chat_col:
        user_question = st.text_input(
    label="", 
    placeholder="🔍 Type your question here..."
)

        if user_question:
            user_input(user_question)


# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
