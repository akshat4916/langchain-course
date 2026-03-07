import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


if __name__ == "__main__":
    print("Ingesting data...")

    loader = TextLoader("/Users/akshat4916/Documents/GitHub Repos/langchain-course/mediumblog1.txt")
    document = loader.load()

    print("Splitting data into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks of data.")

    print("Ingesting data into Pinecone...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=1536)

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ['INDEX_NAME']
    )
    print("Data ingested successfully!")