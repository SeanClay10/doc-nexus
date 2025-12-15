import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Document Processor Class
class DocumentProcessor: 
    def __init__(self):
        # Initializes the DocumentProcessor with a text splitter and OpenAI embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    def process_documents(self,documents):
        # Processes a list of documents by splitting them into chunks and creating embeddings.
        splits = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, vector_store
    
    def create_embeddings_batch(self, texts, batch_size=32):
        # Creates embeddings for a list of texts in batches.
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def compute_similarity_matrix(self, embeddings):
        return cosine_similarity(embeddings)