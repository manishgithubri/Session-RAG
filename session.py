import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

pdf_path = "sample.pdf"
loader = PyPDFLoader(pdf_path)
documents  = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  
)

os.environ['GEMINI_API_KEY'] = "YOUR_API_KEY"

llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
    )
)


query = "Summarize the main objectives of this document?"
response = qa_chain.invoke(query)
print(query)
print(response)
