"""
Vector Store - FAISS-based vector storage with Apple Silicon optimization
High-performance document retrieval for M2 Max
Support for multiple document libraries
"""

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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

        self.embedding_model = None
        self.initialized = False

        # Multiple libraries support
        self.libraries = {}  # library_name -> {index, documents, metadata}
        self.active_library = "default"  # Current active library for searches
        self.libraries_config_path = settings.vector_store_dir / "libraries.json"

        # Memory optimization settings
        self.max_documents_in_memory = 1000  # Keep only recent docs in memory

        # Mark as initialized
        VectorStore._initialized = True

    async def initialize(self):
        """Initialize FAISS index and embedding model"""
        if not VECTOR_AVAILABLE:
            raise Exception(
                "Vector libraries not available. Please install faiss-cpu and sentence-transformers"
            )

        print("🔍 Initializing Vector Store with multiple libraries support...")

        # Load embedding model (optimized for speed on Apple Silicon)
        print(f"📥 Loading embedding model: {settings.embedding_model}")
        # Optimize for memory usage and speed
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            cache_folder=Path("models/embeddings"),  # Local cache
            use_auth_token=None,
        )

        # Enable half precision for memory savings (2x reduction)
        if hasattr(self.embedding_model, "_modules"):
            try:
                import torch

                if torch.backends.mps.is_available():
                    # Use half precision on MPS for memory savings
                    self.embedding_model.half()
                    print("🚀 Enabled half-precision embeddings for memory optimization")
            except Exception as e:
                print(f"⚠️  Half precision not available: {e}")

        # Enable Apple Silicon optimizations if available
        if hasattr(self.embedding_model, "device"):
            try:
                # Try to use MPS (Metal Performance Shaders) if available
                import torch

                if torch.backends.mps.is_available():
                    self.embedding_model = self.embedding_model.to("mps")
                    print("🚀 Using Metal Performance Shaders acceleration")
                elif torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to("cuda")

            except Exception as e:
                print(f"⚠️  GPU acceleration not available: {e}")

        # Load existing libraries or create default
        await self._load_libraries_config()

        # Ensure default library exists
        if "default" not in self.libraries:
            await self.create_library("default", "Default document library")

        self.initialized = True
        print(f"✅ Vector Store initialized with {len(self.libraries)} libraries")

    async def create_library(
        self, library_name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Create a new document library"""
        try:
            if library_name in self.libraries:
                return {"status": "error", "message": "Library already exists"}

            # Create new FAISS index
            index = faiss.IndexFlatIP(settings.vector_dim)

            # Initialize library structure
            self.libraries[library_name] = {
                "index": index,
                "documents": [],
                "metadata": {},
                "description": description,
                "created_at": asyncio.get_event_loop().time(),
                "document_count": 0,
            }

            # Save library
            await self._save_library(library_name)
            await self._save_libraries_config()

            print(f"📚 Created new library: {library_name}")
            return {"status": "success", "message": f"Library '{library_name}' created"}

        except Exception as e:
            print(f"❌ Error creating library: {e}")
            return {"status": "error", "message": str(e)}

    async def delete_library(self, library_name: str) -> Dict[str, Any]:
        """Delete a document library"""
        try:
            if library_name == "default":
                return {"status": "error", "message": "Cannot delete default library"}

            if library_name not in self.libraries:
                return {"status": "error", "message": "Library not found"}

            # Remove from memory
            del self.libraries[library_name]

            # Delete files
            library_dir = settings.vector_store_dir / library_name
            if library_dir.exists():
                import shutil

                shutil.rmtree(library_dir)

            # Update config
            await self._save_libraries_config()

            # Switch to default if this was active
            if self.active_library == library_name:
                self.active_library = "default"

            print(f"🗑️ Deleted library: {library_name}")
            return {"status": "success", "message": f"Library '{library_name}' deleted"}

        except Exception as e:
            print(f"❌ Error deleting library: {e}")
            return {"status": "error", "message": str(e)}

    async def list_libraries(self) -> List[Dict[str, Any]]:
        """List all available libraries"""
        libraries = []
        for name, lib in self.libraries.items():
            libraries.append(
                {
                    "name": name,
                    "description": lib.get("description", ""),
                    "document_count": len(lib["documents"]),
                    "vector_count": lib["index"].ntotal if lib["index"] else 0,
                    "active": name == self.active_library,
                    "created_at": lib.get("created_at", 0),
                }
            )
        return libraries

    async def set_active_library(self, library_name: str) -> Dict[str, Any]:
        """Set the active library for searches"""
        if library_name not in self.libraries:
            return {"status": "error", "message": "Library not found"}

        self.active_library = library_name
        await self._save_libraries_config()

        print(f"📖 Set active library: {library_name}")
        return {
            "status": "success",
            "message": f"Active library set to '{library_name}'",
        }

    async def add_documents(
        self, chunks: List[Dict[str, Any]], library_name: str = None
    ):
        """Add document chunks to specified library (or active library)"""
        target_library = library_name or self.active_library

        if target_library not in self.libraries:
            raise Exception(f"Library '{target_library}' not found")

        try:
            if not chunks:
                return

            print(f"📚 Adding {len(chunks)} chunks to library '{target_library}'...")

            # Extract text for embedding
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings in batches for better performance
            embeddings = await self._generate_embeddings_batch(texts)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings.astype("float32"))

            # Get library reference
            library = self.libraries[target_library]

            # Add to FAISS index
            library["index"].add(embeddings.astype("float32"))

            # Store document metadata with memory optimization
            for i, chunk in enumerate(chunks):
                doc_id = len(library["documents"])
                library["documents"].append(chunk["text"])
                library["metadata"][doc_id] = {
                    **chunk,
                    "embedding_index": library["index"].ntotal - len(chunks) + i,
                    "library": target_library,
                }

            # Update document count
            library["document_count"] = len(library["documents"])

            # Save library to disk
            await self._save_library(target_library)

            print(
                f"✅ Added {len(chunks)} chunks to library '{target_library}'! Total: {library['index'].ntotal}"
            )

        except Exception as e:
            print(f"❌ Error adding documents to library '{target_library}': {e}")
            raise

    async def search(
        self, query: str, k: int = None, library_name: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in specified library (or active library)"""
        target_library = library_name or self.active_library

        if target_library not in self.libraries:
            print(f"⚠️ Library '{target_library}' not found, using default")
            target_library = "default"

        library = self.libraries[target_library]

        if not library["index"] or library["index"].ntotal == 0:
            print(f"📭 No documents in library '{target_library}'")
            return []

        try:
            k = k or settings.vector_store_top_k

            # Generate query embedding
            query_embedding = await self._generate_embeddings_batch([query])
            faiss.normalize_L2(query_embedding.astype("float32"))

            # Search in library's index
            similarities, indices = library["index"].search(
                query_embedding.astype("float32"), k
            )

            # Retrieve matching documents
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if (
                    idx < len(library["documents"]) and similarity > 0.1
                ):  # Threshold for relevance
                    doc_metadata = library["metadata"].get(idx, {})
                    result = {
                        "text": library["documents"][idx],
                        "similarity": float(similarity),
                        "rank": i + 1,
                        "library": target_library,
                        **doc_metadata,
                    }
                    results.append(result)

            print(
                f"🔍 Found {len(results)} relevant documents in library '{target_library}'"
            )
            return results

        except Exception as e:
            print(f"❌ Error searching library '{target_library}': {e}")
            return []

    async def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts with performance optimization"""
        try:
            # Use larger batch sizes for better GPU/MPS utilization
            batch_size = settings.batch_size
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Generate embeddings for this batch
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=min(len(batch_texts), batch_size),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,  # We'll normalize manually for cosine similarity
                )

                all_embeddings.append(batch_embeddings)

            # Concatenate all batches
            embeddings = (
                np.vstack(all_embeddings)
                if len(all_embeddings) > 1
                else all_embeddings[0]
            )

            print(
                f"📊 Generated embeddings for {len(texts)} texts (shape: {embeddings.shape})"
            )
            return embeddings

        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise

    async def _save_library(self, library_name: str):
        """Save individual library index and metadata to disk"""
        try:
            library = self.libraries[library_name]
            library_dir = settings.vector_store_dir / library_name
            library_dir.mkdir(parents=True, exist_ok=True)

            index_path = library_dir / "faiss_index.bin"
            metadata_path = library_dir / "metadata.pkl"

            # Save FAISS index
            faiss.write_index(library["index"], str(index_path))

            # Save metadata
            metadata = {
                "documents": library["documents"],
                "metadata": library["metadata"],
                "index_size": library["index"].ntotal,
                "vector_dim": settings.vector_dim,
                "description": library.get("description", ""),
                "created_at": library.get("created_at", 0),
                "document_count": library["document_count"],
            }

            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            print(
                f"💾 Saved library '{library_name}' with {library['index'].ntotal} vectors"
            )

        except Exception as e:
            print(f"❌ Error saving library '{library_name}': {e}")

    async def _load_libraries_config(self):
        """Load libraries configuration and data from disk"""
        try:
            # Load libraries config if it exists
            config = {}
            if self.libraries_config_path.exists():
                with open(self.libraries_config_path, "r") as f:
                    config = json.load(f)
                    self.active_library = config.get("active_library", "default")

            # Load each library
            vector_store_dir = Path(settings.vector_store_dir)
            if vector_store_dir.exists():
                for library_dir in vector_store_dir.iterdir():
                    if library_dir.is_dir():
                        library_name = library_dir.name
                        await self._load_library(library_name)

            print(f"📚 Loaded {len(self.libraries)} libraries")

        except Exception as e:
            print(f"⚠️  Could not load libraries config: {e}")
            # Initialize with empty libraries dict
            self.libraries = {}

    async def _load_library(self, library_name: str):
        """Load individual library from disk"""
        try:
            library_dir = settings.vector_store_dir / library_name
            index_path = library_dir / "faiss_index.bin"
            metadata_path = library_dir / "metadata.pkl"

            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)

                # Initialize library structure
                self.libraries[library_name] = {
                    "index": index,
                    "documents": metadata.get("documents", []),
                    "metadata": metadata.get("metadata", {}),
                    "description": metadata.get("description", ""),
                    "created_at": metadata.get("created_at", 0),
                    "document_count": metadata.get(
                        "document_count", len(metadata.get("documents", []))
                    ),
                }

                print(
                    f"📖 Loaded library '{library_name}' with {len(self.libraries[library_name]['documents'])} documents"
                )

        except Exception as e:
            print(f"⚠️  Could not load library '{library_name}': {e}")

    async def _save_libraries_config(self):
        """Save libraries configuration to disk"""
        try:
            settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "active_library": self.active_library,
                "libraries": {
                    name: {
                        "description": lib.get("description", ""),
                        "created_at": lib.get("created_at", 0),
                        "document_count": lib["document_count"],
                    }
                    for name, lib in self.libraries.items()
                },
            }

            with open(self.libraries_config_path, "w") as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            print(f"❌ Error saving libraries config: {e}")

    async def get_size(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        total_documents = sum(len(lib["documents"]) for lib in self.libraries.values())
        total_vectors = sum(
            lib["index"].ntotal if lib["index"] else 0
            for lib in self.libraries.values()
        )

        return {
            "total_libraries": len(self.libraries),
            "total_documents": total_documents,
            "total_vectors": total_vectors,
            "active_library": self.active_library,
            "dimension": settings.vector_dim,
            "memory_usage_mb": (
                total_vectors * settings.vector_dim * 4 / (1024 * 1024)
                if total_vectors
                else 0
            ),
            "libraries": await self.list_libraries(),
        }

    async def remove_document_from_store(
        self, file_path: str, library_name: str = None
    ):
        """Remove document from specified library"""
        target_library = library_name or self.active_library

        if target_library not in self.libraries:
            return

        try:
            library = self.libraries[target_library]
            documents_to_remove = []

            # Find documents matching the file path
            for doc_id, metadata in library["metadata"].items():
                if metadata.get("source") == file_path:
                    documents_to_remove.append(doc_id)

            if documents_to_remove:
                print(
                    f"🗑️ Removing {len(documents_to_remove)} document chunks from library '{target_library}'"
                )

                # Note: FAISS doesn't support removing vectors, so we rebuild the index
                # This is expensive but necessary for data consistency
                await self._rebuild_library_index(target_library, documents_to_remove)

        except Exception as e:
            print(f"❌ Error removing document from library '{target_library}': {e}")

    async def _rebuild_library_index(
        self, library_name: str, docs_to_remove: List[int]
    ):
        """Rebuild library index without specified documents"""
        try:
            library = self.libraries[library_name]

            # Keep documents not in removal list
            new_documents = []
            new_metadata = {}
            texts_to_embed = []

            for doc_id, doc_text in enumerate(library["documents"]):
                if doc_id not in docs_to_remove:
                    new_id = len(new_documents)
                    new_documents.append(doc_text)
                    new_metadata[new_id] = library["metadata"].get(doc_id, {})
                    texts_to_embed.append(doc_text)

            # Create new index
            new_index = faiss.IndexFlatIP(settings.vector_dim)

            if texts_to_embed:
                # Generate embeddings for remaining documents
                embeddings = await self._generate_embeddings_batch(texts_to_embed)
                faiss.normalize_L2(embeddings.astype("float32"))
                new_index.add(embeddings.astype("float32"))

            # Replace library data
            library["index"] = new_index
            library["documents"] = new_documents
            library["metadata"] = new_metadata
            library["document_count"] = len(new_documents)

            # Save updated library
            await self._save_library(library_name)

            print(
                f"🔄 Rebuilt library '{library_name}' with {len(new_documents)} documents"
            )

        except Exception as e:
            print(f"❌ Error rebuilding library '{library_name}': {e}")


# Global instance
_vector_store_instance = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
