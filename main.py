import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# 1. HAZIRLIK VE IMPORTLAR
load_dotenv()
app = FastAPI(title="Pro RAG Sistemi", description="Eylül Romanı Analiz API")

# --- FONKSİYONLAR (Alet Çantası) ---

def load_pdf(file_path):
    print(f"📄 PDF yükleniyor: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()

def chunk_documents(documents):
    print("✂️ Chunking yapılıyor...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def create_embeddings(chunks):
    print("🔢 Embedding'ler oluşturuluyor...")
    model = OpenAIEmbeddings(model="text-embedding-ada-002")
    texts = [c.page_content for c in chunks]
    embeddings = model.embed_documents(texts)
    return embeddings, texts

def index_to_qdrant(embeddings, texts):
    print("💾 Qdrant'a kayıt yapılıyor...")
    client = QdrantClient(":memory:")
    col_name = "my_documents"
    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    points = [PointStruct(id=i, vector=e, payload={"text": t}) for i, (e, t) in enumerate(zip(embeddings, texts))]
    client.upsert(collection_name=col_name, points=points)
    return client, col_name

def search_similar(client, col_name, query, top_k=5):
    model = OpenAIEmbeddings(model="text-embedding-ada-002")
    q_emb = model.embed_query(query)
    res = client.query_points(collection_name=col_name, query=q_emb, limit=top_k)
    return res.points

def hybrid_search(client, col_name, query, chunks, top_k=5):
    print(f"🚀 Hibrit arama: '{query}'")
    v_res = search_similar(client, col_name, query, top_k=top_k)
    v_texts = [r.payload["text"] for r in v_res]
    
    token_corpus = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(token_corpus)
    bm25_res = bm25.get_top_n(query.lower().split(), [c.page_content for c in chunks], n=top_k)
    
    final = list(set(v_texts) & set(bm25_res))
    for t in v_texts:
        if t not in final: final.append(t)
    return final[:top_k]

def rerank_results(query, candidates):
    print("⚖️ Reranking (Cross-Encoder)...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = model.predict([[query, c] for c in candidates])
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in reranked]

# --- SİSTEMİ AYAĞA KALDIRMA (Sadece text.pdf ile) ---

print("\n⚙️ Sistem Hazırlanıyor...")
raw_docs = load_pdf("text.pdf") # Sadece hedef dosyan
chunks = chunk_documents(raw_docs)
embs, txts = create_embeddings(chunks)
q_client, collection = index_to_qdrant(embs, txts)

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"mesaj": "Eylül Romanı RAG API Hazır!"}


@app.get("/ask")
def ask_question(query: str):
    """Kullanıcıdan soru alır, arama yapar ve OpenAI ile cevap üretir."""
    # 1. Aşama: Adayları topla (Hybrid Search)
    candidates = hybrid_search(q_client, collection, query, chunks)
    
    # 2. Aşama: Reranking (Hakem ile sırala)
    final_results = rerank_results(query, candidates)
    
    # 3. Aşama: GENERATION (Gerçek RAG burasıdır!)
    # Bulduğumuz en iyi metinleri birleştirip OpenAI'a 'Buradan bakarak cevapla' diyoruz.
    context = "\n".join(final_results[:3])
    
    # OpenAI istemcisini oluşturuyoruz
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Yapay zekaya bağlamı ve soruyu gönderiyoruz
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sen Mehmet Rauf'un Eylül romanı konusunda uzman bir asistansın. Verilen 'Bağlam' içindeki bilgileri kullanarak soruyu cevapla."},
            {"role": "user", "content": f"Bağlam: {context}\n\nSoru: {query}"}
        ]
    )
    
    return {
        "sorgu": query,
        "yapay_zeka_cevabı": response.choices[0].message.content, # Modelin yazdığı gerçek cevap
        "dayandığı_kaynaklar": final_results[:2] # Kanıt olarak sunduğumuz paragraflar
    }

# --- SUNUCUYU ATEŞLE (Sildiğin Kısım Burası) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
