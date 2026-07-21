import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="/home/cmpark/projects/langchain-course/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)}")

    # Connect to your local Ollama server acting as OpenAI
    embeddings = OpenAIEmbeddings(
        model="nomic-embed-text",
        openai_api_base="http://localhost:11434/v1", # The local endpoint
        openai_api_key="local-no-key-needed",        # Dummy bypass key
        check_embedding_ctx_length=False,             # Ollama expects string input, not pre-tokenized ints
    )
    #embeddings = HuggingFaceEmbeddings(
    #   model_name="BAAI/bge-small-en-v1.5"
    #)

    print("ingesting")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finish")

