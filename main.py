from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

reader = PdfReader("sample.pdf")
text=""

for page in reader.pages:
    text+=page.extract_text() + "\n"

chunk_size =500
chunks=[]
for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i+chunk_size])

print("Total Chunks:", len(chunks))


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings=model.encode(
    chunks,
    show_progress_bar=True

)
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="docs")

collection.add(
    embeddings=embeddings.tolist(),
    documents=chunks,   
    ids=[f"id_{i}" for i in range(len(chunks))]
)

query = "What is the main topic of the document?"
query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),  
    n_results=3
)

print(results["documents"][0])