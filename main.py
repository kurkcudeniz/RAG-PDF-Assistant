# main.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

# API key yükle
load_dotenv()

print("RAG Sistemi Başlatılıyor...")

# 1. PDF YÜKLEME
def load_pdf(file_path):
    """PDF dosyasını yükle"""
    print(f"📄 PDF yükleniyor: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ {len(documents)} sayfa yüklendi")
    return documents

# 2. CHUNKING
def chunk_documents(documents):
    """Dökümanları parçalara böl"""
    print("✂️  Chunking yapılıyor...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)} chunk oluşturuldu")
    print(f"📝 Örnek chunk: {chunks[0].page_content[:100]}...")
    
    return chunks

# 3. EMBEDDING OLUŞTURMA
def create_embeddings(chunks):
    """Chunk'ları embedding'e çevir"""
    print("🔢 Embedding'ler oluşturuluyor...")
    
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)
    
    print(f"✅ {len(embeddings)} embedding oluşturuldu")
    print(f"📏 Embedding boyutu: {len(embeddings[0])}")
    
    return embeddings, texts

# 4. QDRANT'A KAYDETME
def index_to_qdrant(embeddings, texts):
    """Embedding'leri Qdrant'a kaydet"""
    print("💾 Qdrant'a kayıt yapılıyor...")
    
    client = QdrantClient(":memory:")
    collection_name = "my_documents"
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )
    
    points = []
    for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={"text": text}
        )
        points.append(point)
    
    client.upsert(collection_name=collection_name, points=points)
    
    print(f"✅ {len(points)} vektör Qdrant'a kaydedildi")
    return client, collection_name

# 5. SORU SORMA
def search_similar(client, collection_name, query, top_k=3):
    """Soruya benzer chunk'ları bul"""
    print(f"🔍 Arama yapılıyor: '{query}'")
        
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")  
    )
    query_embedding = embeddings_model.embed_query(query)

    # --- KODU BURADA GÜNCELLE ---
    query_response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k
    )
    # ---------------------------
    
    # query_response nesnesinin içindeki 'points' listesini alıyoruz
    results = query_response.points 
    
    print(f"✅ {len(results)} sonuç bulundu")
    
    for idx, result in enumerate(results):
        print(f"\n--- Sonuç {idx+1} (Skor: {result.score:.3f}) ---")
        # Yeni yapıda payload'a doğrudan erişim
        print(result.payload["text"][:200] + "...")

    return results

# MAIN
def main():
    documents = load_pdf("sample.pdf")
    chunks = chunk_documents(documents)
    embeddings, texts = create_embeddings(chunks)
    client, collection_name = index_to_qdrant(embeddings, texts)
    
    query = "Bu belge ne hakkında?"
    results = search_similar(client, collection_name, query)
    
    print("\n✅ RAG sistemi başarıyla çalıştı!")

if __name__ == "__main__":
    main()
