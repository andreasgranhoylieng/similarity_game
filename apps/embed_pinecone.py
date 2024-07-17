from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
from tqdm import tqdm
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os

def load_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=model_name)

def load_words(file_path: str) -> list:
    with open(file_path, "r") as f:
        return f.read().splitlines()

def compute_embeddings(embed_model: HuggingFaceEmbedding, words: list) -> list:
    return [embed_model.get_text_embedding(word) for word in tqdm(words, desc="Computing embeddings")]

def create_dataframe(words: list, embeddings: list) -> pd.DataFrame:
    return pd.DataFrame({'word': words, 'embedding': embeddings})

def setup_pinecone(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        )

def index_words(pc: Pinecone, index_name: str, df: pd.DataFrame, batch_size: int = 100):
    index = pc.Index(index_name)
    batches = [
        [{"id": row['word'], "values": row['embedding']} for _, row in df.iloc[i:i + batch_size].iterrows()]
        for i in range(0, len(df), batch_size)
    ]
    for batch in tqdm(batches, desc="Indexing words (batches)"):
        index.upsert(vectors=batch, namespace="words")

def query_pinecone(index, embed_model: HuggingFaceEmbedding, query_word: str, top_k: int = 3):
    return index.query(
        namespace="words",
        vector=embed_model.get_text_embedding(query_word),
        top_k=top_k,
        include_values=False
    )

def main():
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    embed_model = load_embedding_model("BAAI/bge-small-en-v1.5")
    
    words = load_words("../data/raw/dictionary.txt")
    embeddings = compute_embeddings(embed_model, words)
    
    df = create_dataframe(words, embeddings)
    
    pc = setup_pinecone(pinecone_api_key)
    index_name = "similarity-game"
    
    create_pinecone_index(pc, index_name, len(embeddings[0]))
    index_words(pc, index_name, df)
    
    index = pc.Index(index_name)
    print(index.describe_index_stats())
    
    #query_results1 = query_pinecone(index, embed_model, "take")
    #print(query_results1)
    
    # Optional: Uncomment to delete the index
    #pc.delete_index(index_name)

if __name__ == "__main__":
    main()
