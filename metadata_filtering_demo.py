from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

print("🔍 Metadata Filtering Demo with ChromaDB")

# 1. API Key
api_key = input("Enter OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = api_key

# 2. Sample documents with metadata
documents = [
    Document(
        page_content="Machine learning algorithms can be supervised, unsupervised, or reinforcement learning based.",
        metadata={"chapter": 1, "topic": "ML Basics", "difficulty": "beginner"}
    ),
    Document(
        page_content="Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
        metadata={"chapter": 2, "topic": "Deep Learning", "difficulty": "intermediate"}
    ),
    Document(
        page_content="Natural Language Processing enables computers to understand and generate human language.",
        metadata={"chapter": 3, "topic": "NLP", "difficulty": "intermediate"}
    ),
    Document(
        page_content="Transfer learning allows models trained on one task to be adapted for related tasks.",
        metadata={"chapter": 2, "topic": "Deep Learning", "difficulty": "advanced"}
    ),
    Document(
        page_content="Feature engineering is the process of creating meaningful input variables for ML models.",
        metadata={"chapter": 1, "topic": "ML Basics", "difficulty": "beginner"}
    ),
    Document(
        page_content="Transformer architecture revolutionized NLP with attention mechanisms.",
        metadata={"chapter": 3, "topic": "NLP", "difficulty": "advanced"}
    ),
]

print(f"✅ Created {len(documents)} sample documents")

# 3. Embeddings
print("\n🔢 Creating embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# 4. ChromaDB
print("💾 Storing in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="metadata_demo"
)

print("✅ ChromaDB ready!")

# 5. Test 1: Normal search
print("\n" + "="*60)
print("🔍 TEST 1: Normal Search (No Filter)")
print("="*60)

query1 = "deep learning neural networks"
print(f"Query: {query1}\n")

results1 = vectorstore.similarity_search(query1, k=3)

for i, doc in enumerate(results1, 1):
    print(f"\n📄 Result {i}:")
    print(f"Chapter: {doc.metadata['chapter']}")
    print(f"Topic: {doc.metadata['topic']}")
    print(f"Difficulty: {doc.metadata['difficulty']}")
    print(f"Content: {doc.page_content[:80]}...")

# 6. Test 2: Filter by chapter
print("\n" + "="*60)
print("🔍 TEST 2: Filter by Chapter")
print("="*60)

query2 = "learning models"
print(f"Query: {query2}")
print(f"Filter: chapter = 1 (ML Basics only)\n")

results2 = vectorstore.similarity_search(
    query2,
    k=3,
    filter={"chapter": 1}
)

for i, doc in enumerate(results2, 1):
    print(f"\n📄 Result {i}:")
    print(f"Chapter: {doc.metadata['chapter']}")
    print(f"Topic: {doc.metadata['topic']}")
    print(f"Difficulty: {doc.metadata['difficulty']}")
    print(f"Content: {doc.page_content[:80]}...")

# 7. Test 3: Multiple filters
print("\n" + "="*60)
print("🔍 TEST 3: Multiple Filters (AND)")
print("="*60)

query3 = "machine learning"
print(f"Query: {query3}")
print(f"Filter: topic = 'NLP' AND difficulty = 'advanced'\n")

results3 = vectorstore.similarity_search(
    query3,
    k=2,
    filter={
        "$and": [
            {"topic": {"$eq": "NLP"}},
            {"difficulty": {"$eq": "advanced"}}
        ]
    }
)

for i, doc in enumerate(results3, 1):
    print(f"\n📄 Result {i}:")
    print(f"Chapter: {doc.metadata['chapter']}")
    print(f"Topic: {doc.metadata['topic']}")
    print(f"Difficulty: {doc.metadata['difficulty']}")
    print(f"Content: {doc.page_content}")

print("\n✅ Metadata Filtering Demo Complete!")
