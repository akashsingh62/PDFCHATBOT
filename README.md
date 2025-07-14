# Chat with PDF using Gemini 💬📄

A sleek, privacy‑first Streamlit app that lets you *talk* to your PDF documents using Google’s Gemini LLM – no cloud database required.

![Screenshot](assets/screenshot.png) <!-- Replace with your own screenshot -->

---

## ✨ Features

- **Multi‑PDF support** – upload and query multiple PDFs at once.
- **Gemini‑powered answers** – accurate, context‑aware responses.
- **Local FAISS vector store** – ensures data privacy.
- **Streamlit UI** – fast, interactive, and easy-to-use.
- **.env for secrets** – keep your API keys safe.

---

## 🧰 Tech Stack

| Layer        | Tool                                     |
| ------------ | ---------------------------------------- |
| UI           | Streamlit                               |
| LLM          | Gemini 2.0 Flash (Google Generative AI) |
| Embeddings   | GoogleGenerativeAI Embedding-001        |
| Vector Store | FAISS                                   |
| PDF Parsing  | PyPDF2                                  |
| LLM Orchestration | LangChain                         |

---

## 🛠️ Prerequisites

- Python >= 3.10
- Google Generative AI API Key → [Get it here](https://aistudio.google.com/app/apikey)

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/chat-pdf-gemini.git
cd chat-pdf-gemini

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Then open .env and set GOOGLE_API_KEY

# 5. Run the app
streamlit run app.py
