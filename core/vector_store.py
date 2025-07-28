"""
Vector Store - FAISS-based vector storage with Apple Silicon optimization
High-performance document retrieval for M2 Max
"""
import asyncio
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_AVAILABLE = True
except ImportError as e:
    VECTOR_AVAILABLE = False
    print(f"⚠️  Vector libraries not available: {e}")

from config import settings

class VectorStore:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        # Only initialize once
        if VectorStore._initialized:
            return
            
        self.index = None
        self.embedding_model = None
        # Use memory-mapped storage for large document collections
        self.documents = []
        self.document_metadata = {}
        self.initialized = False
        
        # Memory optimization settings
        self.max_documents_in_memory = 1000  # Keep only recent docs in memory
        self.document_storage_path = Path("data/documents_storage.pkl")
        
        # Mark as initialized  
        VectorStore._initialized = True
        
    async def initialize(self):
        """Initialize FAISS index and embedding model"""
        if not VECTOR_AVAILABLE:
            raise Exception("Vector libraries not available. Please install faiss-cpu and sentence-transformers")
        
        print("🔍 Initializing Vector Store...")
        
        # Load embedding model (optimized for speed on Apple Silicon)
        print(f"📥 Loading embedding model: {settings.embedding_model}")
        # Optimize for memory usage and speed
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            cache_folder=Path("models/embeddings"),  # Local cache
            use_auth_token=None
        )
        
        # Enable half precision for memory savings (2x reduction)
        if hasattr(self.embedding_model, '_modules'):
            try:
                import torch
                if torch.backends.mps.is_available():
                    # Use half precision on MPS for memory savings
                    self.embedding_model.half()
                    print("🚀 Enabled half-precision embeddings for memory optimization")
            except Exception as e:
                print(f"⚠️  Half precision not available: {e}")
        
        # Enable Apple Silicon optimizations if available
        if hasattr(self.embedding_model, 'device'):
            try:
                # Try to use MPS (Metal Performance Shaders) if available
                import torch
                if torch.backends.mps.is_available():
                    self.embedding_model = self.embedding_model.to('mps')
                    print("🚀 Using Metal Performance Shaders acceleration")
                elif torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to('cuda')
                    print("🚀 Using CUDA acceleration")
            except Exception as e:
                print(f"⚠️  GPU acceleration not available: {e}")
        
        # Create FAISS index (Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(settings.vector_dim)
        
        # Try to load existing index
        await self._load_existing_index()
        
        self.initialized = True
        print("✅ Vector Store initialized successfully!")
    
    async def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to vector store"""
        try:
            if not chunks:
                return
            
            print(f"📚 Adding {len(chunks)} chunks to vector store...")
            
            # Extract text for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings in batches for better performance
            embeddings = await self._generate_embeddings_batch(texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings.astype('float32'))
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store document metadata with memory optimization
            for i, chunk in enumerate(chunks):
                doc_id = len(self.documents)
                self.documents.append(chunk["text"])
                self.document_metadata[doc_id] = {
                    **chunk,
                    "embedding_index": self.index.ntotal - len(chunks) + i
                }
            
            # Memory management: if we have too many documents, persist to disk
            if len(self.documents) > self.max_documents_in_memory:
                await self._persist_documents_to_disk()
            
            # Save index to disk
            await self._save_index()
            
            print(f"✅ Added {len(chunks)} chunks successfully! Total: {self.index.ntotal}")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise
    
    async def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            k = k or settings.top_k_results
            
            if not self.initialized:
                await self.initialize()
                
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embeddings_batch([query])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(self.documents):  # Valid result
                    result = {
                        "text": self.documents[idx],
                        "score": float(score),
                        "rank": i + 1,
                        "metadata": self.document_metadata.get(idx, {}),
                        "similarity": float(score)  # Cosine similarity score
                    }
                    results.append(result)
            
            print(f"🔍 Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts in batches"""
        try:
            # Process in batches for better memory usage on large document sets
            batch_size = settings.batch_size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Run embedding generation in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None, 
                    lambda: self.embedding_model.encode(
                        batch_texts, 
                        convert_to_numpy=True, 
                        show_progress_bar=False
                    )
                )
                all_embeddings.append(batch_embeddings)
                
                # Small delay to prevent overwhelming the system
                if len(texts) > batch_size:
                    await asyncio.sleep(0.01)
            
            # Concatenate all batches
            if len(all_embeddings) == 1:
                return all_embeddings[0]
            else:
                return np.vstack(all_embeddings)
                
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise
    
    async def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            index_path = settings.vector_store_dir / "faiss_index.bin"
            metadata_path = settings.vector_store_dir / "metadata.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                "documents": self.documents,
                "document_metadata": self.document_metadata,
                "index_size": self.index.ntotal,
                "vector_dim": settings.vector_dim
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            print(f"💾 Saved index with {self.index.ntotal} vectors")
                
        except Exception as e:
            print(f"❌ Error saving index: {e}")
    
    async def _load_existing_index(self):
        """Load existing FAISS index and metadata"""
        try:
            index_path = settings.vector_store_dir / "faiss_index.bin"
            metadata_path = settings.vector_store_dir / "metadata.pkl"
            
            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.documents = metadata.get("documents", [])
                    self.document_metadata = metadata.get("document_metadata", {})
                
                print(f"📚 Loaded existing index with {len(self.documents)} documents ({self.index.ntotal} vectors)")
                
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
            # Create new index
            self.index = faiss.IndexFlatIP(settings.vector_dim)
    
    async def get_size(self) -> Dict[str, int]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": settings.vector_dim,
            "memory_usage_mb": self.index.ntotal * settings.vector_dim * 4 / (1024 * 1024) if self.index else 0
        }
    
    async def remove_by_source(self, source_path: str) -> int:
        """Remove all chunks from a document by its source path"""
        try:
            # Find all chunks from this source
            indices_to_remove = []
            for idx, metadata in self.document_metadata.items():
                if metadata.get("deleted"):
                    continue
                if metadata.get("source") == source_path:
                    indices_to_remove.append(idx)
            
            if not indices_to_remove:
                return 0
            
            # Mark chunks as deleted (same approach as delete_document)
            for idx in indices_to_remove:
                if idx in self.document_metadata:
                    self.document_metadata[idx]["deleted"] = True
            
            await self._save_index()
            print(f"🗑️  Removed {len(indices_to_remove)} chunks from {source_path}")
            return len(indices_to_remove)
            
        except Exception as e:
            print(f"❌ Error removing document by source: {e}")
            return 0

    async def clear_index(self):
        """Clear all documents from vector store"""
        try:
            if self.index:
                self.index.reset()
            self.documents = []
            self.document_metadata = {}
            await self._save_index()
            print("🗑️  Vector store cleared successfully!")
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
    
    async def delete_document(self, filename: str) -> bool:
        """Delete all chunks from a specific document"""
        try:
            # Find all chunks from this document
            indices_to_remove = []
            for idx, metadata in self.document_metadata.items():
                if metadata.get("source") == filename or metadata.get("filename") == filename:
                    indices_to_remove.append(idx)
            
            if not indices_to_remove:
                return False
            
            # FAISS doesn't support individual vector deletion easily
            # So we rebuild the index without the deleted documents
            remaining_embeddings = []
            remaining_docs = []
            remaining_metadata = {}
            
            for idx in range(len(self.documents)):
                if idx not in indices_to_remove:
                    # We'd need to store embeddings to rebuild - for now, just mark as deleted
                    remaining_docs.append(self.documents[idx])
                    remaining_metadata[len(remaining_docs) - 1] = self.document_metadata[idx]
            
            # For simplicity, mark as deleted in metadata rather than rebuilding
            for idx in indices_to_remove:
                if idx in self.document_metadata:
                    self.document_metadata[idx]["deleted"] = True
            
            await self._save_index()
            print(f"🗑️  Marked {len(indices_to_remove)} chunks as deleted from {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False
    
    async def _persist_documents_to_disk(self):
        """Persist older documents to disk to free memory"""
        try:
            # Keep only recent documents in memory
            keep_count = self.max_documents_in_memory // 2
            
            # Save older documents to disk
            older_docs = self.documents[:-keep_count]
            older_metadata = {k: v for k, v in self.document_metadata.items() if k < len(older_docs)}
            
            # Load existing persisted data
            persisted_data = {"documents": [], "metadata": {}}
            if self.document_storage_path.exists():
                with open(self.document_storage_path, 'rb') as f:
                    persisted_data = pickle.load(f)
            
            # Append new data
            persisted_data["documents"].extend(older_docs)
            persisted_data["metadata"].update(older_metadata)
            
            # Save to disk
            self.document_storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.document_storage_path, 'wb') as f:
                pickle.dump(persisted_data, f)
            
            # Keep only recent documents in memory
            self.documents = self.documents[-keep_count:]
            self.document_metadata = {
                k - len(older_docs): v for k, v in self.document_metadata.items() 
                if k >= len(older_docs)
            }
            
            print(f"📦 Persisted {len(older_docs)} documents to disk, keeping {len(self.documents)} in memory")
            
        except Exception as e:
            print(f"❌ Error persisting documents: {e}")
    
    async def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all documents in the vector store"""
        try:
            documents = {}
            
            for metadata in self.document_metadata.values():
                if metadata.get("deleted"):
                    continue
                    
                filename = metadata.get("source") or metadata.get("filename", "Unknown")
                
                if filename not in documents:
                    documents[filename] = {
                        "filename": filename,
                        "chunk_count": 0,
                        "total_chars": 0
                    }
                
                documents[filename]["chunk_count"] += 1
                documents[filename]["total_chars"] += len(metadata.get("text", ""))
            
            return list(documents.values())
            
        except Exception as e:
            print(f"❌ Error getting document list: {e}")
            return [] 