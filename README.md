# RAG-PDF-Assistant

Bu proje, PDF belgeleri üzerinden anlamsal arama yapılmasını sağlayan bir Retrieval-Augmented Generation (RAG) prototipidir.

## Teknik Özellikler
* PDF İşleme: PyPDFLoader entegrasyonu ile doküman okuma.
* Metin Parçalama: RecursiveCharacterTextSplitter kullanılarak optimize edilmiş chunking işlemi.
* Vektör Embedding: OpenAI text-embedding-ada-002 modeli ile semantik vektör üretimi.
* Vektör Veritabanı: Qdrant (In-memory) üzerinde yüksek performanslı benzerlik araması.

## Kullanılan Teknolojiler
* Dil: Python 3.9+
* Kütüphaneler: LangChain, Qdrant-client, OpenAI
* Veritabanı: Qdrant

## Kurulum
1. Depoyu yerel makinenize klonlayın:
   ```bash
   git clone [https://github.com/kurkcudeniz/RAG-PDF-Assistant.git](https://github.com/kurkcudeniz/RAG-PDF-Assistant.git)
