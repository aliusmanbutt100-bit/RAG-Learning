"""
RAG Stand for Retrival Augmented Generation
Step 1: PDF ko chote chote pieces mein todo (chunks)
        ↓
Step 2: Har piece ko numbers mein convert karo (embeddings)
        ↓  
Step 3: Jab sawaal aaye — sirf relevant piece nikalo
        aur AI ko do
"""
#Step 1 Chunks: learning to split documents in chunks
from langchain_text_splitters import CharacterTextSplitter

# Document padho
with open("styluxe_policy.txt", "r") as f:
    text = f.read()

print("Original text length:", len(text))

# Text ko chunks mein todo
splitter = CharacterTextSplitter(
    chunk_size=200,    # har chunk 200 characters ka
    chunk_overlap=20   # chunks thoda overlap karein
)

chunks = splitter.split_text(text)

print(f"\nTotal chunks banay: {len(chunks)}")
print("\nPehla chunk:")
print(chunks[0])
print("\nDoosra chunk:")
print(chunks[1])

#Step2 Embedding:convert text into numbers bcz computer cant read text and it only can understand numbers
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embedding model load karo
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Pehle chunk ko numbers mein convert karo
vector = embeddings.embed_query(chunks[0])   #ye sirf pahla chunk conert huwa numbers mei not all chunks

print(f"\nChunk numbers mein convert hua!")
print(f"Total numbers: {len(vector)}")
print(f"Pehle 5 numbers: {vector[:5]}")

#step3 Vector Database:it is used to store all the numbers that model made so it can use it later also
from langchain_community.vectorstores import Chroma

# Vector database banao — saare chunks store karo
# Ye line saare chunks ko automatically embed karti hai
vectordb = Chroma.from_texts(
    texts=chunks,          # saare 5 chunks
    embedding=embeddings   # har ek ko numbers mein convert karo
)

print("\nVector DB ban gayi!")
print(f"Total chunks stored: {vectordb._collection.count()}")

# Ab search karo
query = "Do you have any discounts?"
results = vectordb.similarity_search(query, k=2)   #k ki value set karny ka ye rule hai ky jitny total chunks hongy utni oska 10% k set kary gai like abhi 5 chunks thy toh k=1 ya k=2 sahi hai but jaab chunks zayada ho jai like 100 toh iska 10% k=5 ya k=10

print(f"\nSawaal: {query}")
print(f"Relevant chunk mila:")

for i, result in enumerate(results): #we use for loop to get more than one result
    print(f"\nResult {i+1}:")
    print(result.page_content)

