
# ChatPDF with Amazon Bedrock & FAISS

Interact with your PDF files using natural language! This Streamlit-powered application leverages Amazon Bedrock's large language models (Claude v2 and LLaMA3-70B) for intelligent question answering and FAISS for document retrieval.

## 🚀 Features

- 📄 Upload and process PDFs automatically from the `data/` directory.
- 🔍 Chunk and embed text using `amazon.titan-embed-text-v1` from Bedrock.
- 🧠 Retrieve answers using:
  - 🧑‍🏫 **Claude v2** (Anthropic)
  - 🦙 **LLaMA 3 70B Instruct** (Meta)
- 🧠 Vector indexing and similarity search powered by **FAISS**.
- 🌐 Fully interactive **Streamlit** UI.

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Embedding & LLMs:** Amazon Bedrock (Titan Embedding, Claude, LLaMA3)
- **Vector DB:** FAISS
- **PDF Processing:** LangChain `PyPDFDirectoryLoader`
- **Text Splitting:** RecursiveCharacterTextSplitter

---

## 📁 Project Structure

```
├── app.py                 # Main application logic
├── requirements.txt       # Python dependencies
├── data/                  # PDF files directory
├── faiss_index/           # Stores FAISS vector index files
│   ├── index.faiss
│   └── index.pkl
├── venv/                  # (Optional) Virtual environment
```

---

## 📥 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/jayd972/ChatPDF-with-Amazon-Bedrock-FAISS-Ask-Questions-from-PDFs-using-Claude-or-LLaMA2.git
cd ChatPDF-with-Amazon-Bedrock-FAISS-Ask-Questions-from-PDFs-using-Claude-or-LLaMA2
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

3. **Install Requirements**

```bash
pip install -r requirements.txt
```

4. **Configure AWS Bedrock**

Ensure your AWS credentials are configured to access Bedrock services:

```bash
aws configure
```

You’ll need access to the following Bedrock models:
- `amazon.titan-embed-text-v1`
- `anthropic.claude-v2`
- `meta.llama3-70b-instruct-v1:0`

---

## 🧪 Run the App

```bash
streamlit run app.py
```

---

## 📝 Usage

1. Place your PDF files inside the `/data` directory.
2. Launch the app.
3. Click **“Vectors Update”** in the sidebar to create a new FAISS index.
4. Type your question.
5. Click on **“Claude Output”** or **“Llama2 Output”** to get an answer based on your vector store.

---

## 📌 Notes

- Adjust chunk sizes in `RecursiveCharacterTextSplitter` if you're working with large or complex documents.
- The default embedding model is `amazon.titan-embed-text-v1`; you can swap this as needed.

---

## 🤝 Contributing

Feel free to fork the repository and submit pull requests for enhancements or fixes. Contributions are always welcome!
