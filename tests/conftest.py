"""
Pytest configuration and fixtures for LLMind tests
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

from config import settings
from core.mlx_manager import MLXManager
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.chat_engine import ChatEngine
from core.audio_manager import AudioManager

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Create a test client for FastAPI app"""
    # Import app only when needed to avoid premature component creation
    from main import app
    with TestClient(app) as client:
        yield client

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_mlx_manager():
    """Create a mock MLX manager for testing"""
    manager = Mock(spec=MLXManager)
    manager.initialize = AsyncMock()
    manager.get_available_models = AsyncMock(return_value=[
        {
            "name": "test-model",
            "status": "available",
            "size": "1GB",
            "type": "test"
        }
    ])
    manager.download_model = AsyncMock()
    manager.switch_model = AsyncMock()
    manager.generate_response = AsyncMock(return_value="Test response")
    manager.stream_response = AsyncMock()
    manager.get_tokens_per_second = Mock(return_value=50.0)
    manager.get_memory_usage = Mock(return_value={
        "used_gb": 1.5,
        "total_gb": 16.0,
        "percentage": 9.4
    })
    manager.get_model_info = Mock(return_value={
        "name": "test-model",
        "status": "loaded"
    })
    manager.current_model = Mock()
    manager.current_model_name = "test-model"
    return manager

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    store = Mock(spec=VectorStore)
    store.initialize = AsyncMock()
    store.add_documents = AsyncMock()
    store.search = AsyncMock(return_value=[
        {
            "text": "Test document content",
            "score": 0.95,
            "metadata": {"filename": "test.txt", "chunk_index": 0}
        }
    ])
    store.get_size = AsyncMock(return_value={
        "total_documents": 5,
        "total_vectors": 50,
        "dimension": 384
    })
    store.initialized = True
    return store

@pytest.fixture
def mock_document_processor():
    """Create a mock document processor for testing"""
    processor = Mock(spec=DocumentProcessor)
    processor.process_file = AsyncMock(return_value={
        "filename": "test.txt",
        "file_hash": "test_hash",
        "text_length": 100,
        "chunks": [
            {
                "id": "test_chunk_0",
                "text": "Test content",
                "source": "test.txt",
                "chunk_index": 0,
                "metadata": {"filename": "test.txt"}
            }
        ],
        "chunk_count": 1
    })
    processor.get_document_count = AsyncMock(return_value=1)
    processor.get_document_list = AsyncMock(return_value=[
        {
            "filename": "test.txt",
            "file_hash": "test_hash",
            "text_length": 100,
            "chunk_count": 1
        }
    ])
    return processor

@pytest.fixture
def mock_audio_manager():
    """Create a mock audio manager for testing"""
    manager = Mock(spec=AudioManager)
    manager.initialize = AsyncMock()
    manager.speech_to_text = AsyncMock(return_value="Test transcription")
    manager.text_to_speech = AsyncMock(return_value="base64_audio_data")
    manager.get_audio_settings = Mock(return_value={
        "current_voice": "Samantha",
        "tts_rate": 200,
        "whisper_available": True
    })
    manager.get_voices = AsyncMock(return_value=[
        {"name": "Samantha", "language": "en_US"},
        {"name": "Alex", "language": "en_US"}
    ])
    manager.whisper_model = Mock()
    return manager

@pytest.fixture
def mock_chat_engine():
    """Create a mock chat engine for testing"""
    engine = Mock(spec=ChatEngine)
    engine.stream_response = AsyncMock()
    engine.generate_response = AsyncMock(return_value={
        "response": "Test chat response",
        "sources": [],
        "metadata": {"response_time": 0.5}
    })
    engine.get_avg_response_time = Mock(return_value=0.3)
    engine.conversation_history = []
    return engine

@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing"""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is a sample text file for testing document processing.")
    return file_path

@pytest.fixture
def sample_pdf_content():
    """Return sample PDF content as bytes"""
    # This is a minimal PDF content for testing
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000110 00000 n
0000000205 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
295
%%EOF"""

@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir, monkeypatch):
    """Set up test environment with temporary directories"""
    # Override settings for testing
    test_settings = {
        "models_dir": temp_dir / "models",
        "documents_dir": temp_dir / "documents",
        "vector_store_dir": temp_dir / "vector_store",
        "static_dir": temp_dir / "static",
        "templates_dir": temp_dir / "templates"
    }
    
    for key, value in test_settings.items():
        monkeypatch.setattr(settings, key, value)
        value.mkdir(exist_ok=True)

class AsyncIterator:
    """Helper class for testing async generators"""
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item 