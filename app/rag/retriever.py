import pandas as pd
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()

def load_vectorstore(csv_path="data/job_roles_dataset.csv"):
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        content = (
            f"Role: {row['Role']}\n"
            f"Skills: {row['Required Skills']}\n"
            f"Description: {row['Description']}\n"
            f"Traits: {row['Traits']}"
        )
        documents.append(Document(page_content=content))

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
