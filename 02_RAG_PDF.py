#In previous it reads the text but now we see how to read pdf file in RAG its so simple 
"""
we just have to change the:
with open("styluxe_policy.txt", "r") as f:
    text = f.read()
into:
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("styluxe_policy.pdf")
documents = loader.load()
text = "\n".join([doc.page_content for doc in documents])
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# PDF load karo
loader = PyPDFLoader("styluxe_policy_pdf.pdf")
documents = loader.load()

# Text nikalo
text = "\n".join([doc.page_content for doc in documents])
print("PDF se text nikala:")
print(text)
print(f"\nTotal characters: {len(text)}")

# Chunks banao
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_text(text)

print(f"\nTotal chunks: {len(chunks)}")
print(f"\nPehla chunk:")
print(chunks[0])
#so we see chunk in output so it works(baki saab same hai bs itna portion change hoga when we need to read pdf file)