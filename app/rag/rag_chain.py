from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from rag.retriever import load_vectorstore

def create_rag_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain
