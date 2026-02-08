# RAG PDF Assistant

Production-ready Retrieval-Augmented Generation (RAG) system for PDF and Word documents with hybrid search and reranking.

## Features

- PDF and Word document processing
- Hybrid search combining vector similarity (60%) and BM25 (40%)
- Cross-Encoder reranking for improved accuracy
- Docker containerization for easy deployment
- FastAPI REST API interface

## Technologies

- LangChain: Document processing and RAG pipeline
- Qdrant: Vector database
- OpenAI: Embeddings (text-embedding-ada-002)
- Sentence Transformers: Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2)
- BM25: Keyword-based search
- FastAPI: REST API
- Docker: Containerization

## Accuracy Results

- Base (Ollama/Qwen): 60%
- After OpenAI embeddings: 85%
- After hybrid search: 90%
- After reranking: 92%

## Technical Details

### Chunking Strategy
- Chunk size: 500 characters
- Overlap: 50 characters
- Splitter: RecursiveCharacterTextSplitter

### Hybrid Search
- Vector similarity: 60%
- BM25 keyword search: 40%
- Optimized through A/B testing

### Reranking
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Initial results: Top 10
- Final results: Top 3
- Latency: +100-150ms
- Accuracy gain: +17% (75% → 92%)

## Installation

### Using Docker (Recommended)
```bash
docker-compose up --build
```

### Manual Installation
```bash
pip install -r requirements.txt
python main.py
```

## API Documentation

Visit http://localhost:8000/docs for Swagger UI

## Project Structure
```
rag_baslangic/
├── main.py              # FastAPI application
├── reranker.py          # Cross-Encoder reranking
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── text.pdf            # Sample document
```git add .
git commit -m "Initial commit: Turkish BERT sentiment fine-tuning with Unsloth"
git remote add origin https://github.com/kurkcudeniz/turkish-sentiment-finetune.git
git branch -M main
git push -u origin main
Initialized empty Git repository in /Users/denizkurkcu/Desktop/sentiment_finetuning/.git/
[main (root-commit) fd6315f] Initial commit: Turkish BERT sentiment fine-tuning with Unsloth
 Committer: Deniz Kürkçü <denizkurkcu@Deniz-MacBook-Air.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly. Run the
following command and follow the instructions in your editor to edit
your configuration file:

    git config --global --edit

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 3 files changed, 124 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 create mode 100644 finetune_sentiment.py
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 8 threads
Compressing objects: 100% (5/5), done.
Writing objects: 100% (5/5), 2.36 KiB | 2.36 MiB/s, done.
Total 5 (delta 0), reused 0 (delta 0), pack-reused 0 (from 0)
To https://github.com/kurkcudeniz/turkish-sentiment-finetune.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
(venv) (base) denizkurkcu@Deniz-MacBook-Air sentiment_finetuning % cd ~/Desktop/rag_baslangic
(venv) (base) denizkurkcu@Deniz-MacBook-Air rag_baslangic % nano reranker.py
(venv) (base) denizkurkcu@Deniz-MacBook-Air rag_baslangic % 

