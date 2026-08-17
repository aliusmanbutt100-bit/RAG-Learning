from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1 — Document padho aur chunks banao
with open("styluxe_policy.txt", "r") as f:
    text = f.read()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_text(text)

# Step 2 — Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3 — Vector DB
vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings)

# Step 4 — AI se jawab lo
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

query = "What is the return policy and do you have discounts?"

results = vectordb.similarity_search(query, k=2)
context = "\n".join([r.page_content for r in results])

print("Context jo AI ko diya:")
print(context)
print("\n" + "="*50)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant for Styluxe Wears. Answer only based on the context provided."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nSawaal: {query}"
        }
    ]
)

print("AI ka jawab:")
print(response.choices[0].message.content)