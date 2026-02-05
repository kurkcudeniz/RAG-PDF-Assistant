# main.py
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
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

# 6. HIBRIT ARAMA (BM25 + VEKTÖR) - BU BLOĞU main() ÜSTÜNE TAŞI
def hybrid_search(client, collection_name, query, chunks, top_k=3):
    print(f"\n🚀 Hibrit arama süreci başladı: '{query}'")
    
    # --- Adım A: Vektör Araması (Semantic - Anlamsal) ---
    vector_results = search_similar(client, collection_name, query, top_k=top_k)
    vector_texts = [res.payload["text"] for res in vector_results]
    
    # --- Adım B: BM25 Araması (Lexical - Kelime Bazlı) ---
    # Metinleri kelimelerine ayırıyoruz (Tokenization)
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Soruyu kelimelerine ayırıp BM25 algoritmasına göre en iyi sonuçları alıyoruz
    tokenized_query = query.lower().split()
    bm25_top_results = bm25.get_top_n(tokenized_query, [c.page_content for c in chunks], n=top_k)
    
    # --- Adım C: Skor Birleştirme (Score Fusion) ---
    # Mülakat Cevabı: "Hem vektörde hem de BM25'te ortak çıkan sonuçlara öncelik verdim."
    final_results = []
    
    # Önce iki listede de ortak olanları ekleyelim (En kaliteli sonuçlar)
    combined = list(set(vector_texts) & set(bm25_top_results))
    final_results.extend(combined)
    
    # Eksik kalan yerleri vektör sonuçlarıyla tamamlayalım
    for res in vector_texts:
        if res not in final_results:
            final_results.append(res)
            
    print(f"✅ Hibrit arama tamamlandı. {len(final_results[:top_k])} sonuç optimize edildi.")
    return final_results[:top_k]


# 7. RERANKING (HAKEM MODEL)
def rerank_results(query, candidates):
    print(f"⚖️  Reranking uygulanıyor (Cross-Encoder)...")
    
    # Küçük ve hızlı bir reranker modeli yüklüyoruz
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Soruyu ve her bir adayı eşleştiriyoruz
    pairs = [[query, cand] for cand in candidates]
    
    # Skorları hesaplıyoruz
    scores = model.predict(pairs)
    
    # Skorlara göre adayları yeniden sıralıyoruz
    reranked = sorted(list(zip(candidates, scores)), key=lambda x: x[1], reverse=True)
    
    print("✅ Yeniden sıralama tamamlandı.")
    return [item[0] for item in reranked]

        

# MAIN
def main():
    documents = load_pdf("sample.pdf")
    chunks = chunk_documents(documents)
    embeddings, texts = create_embeddings(chunks)
    client, collection_name = index_to_qdrant(embeddings, texts)

    query = "Bu belge ne hakkında?"
    
    # ESKİ SATIRI SİLDİK VEYA YORUMA ALDIK:
    # results = search_similar(client, collection_name, query)
    
    # YENİ HIBRIT MOTORU ÇALIŞTIRIYORUZ:
    results = hybrid_search(client, collection_name, query, chunks)
    
    print("\n✅ RAG sistemi HIBRIT modda başarıyla çalıştı!")
    # ... önceki adımlar aynı ...
    
    # 1. Adayları topla (Hybrid Search)
    candidates = hybrid_search(client, collection_name, query, chunks)
    
    # 2. Adayları akıllıca sırala (Reranking)
    final_results = rerank_results(query, candidates)
    
    print("\n🏆 EN DOĞRU SONUÇLAR (Reranked):")
    for idx, text in enumerate(final_results[:3]):
        print(f"{idx+1}. {text[:150]}...")





if __name__ == "__main__":
    main()


