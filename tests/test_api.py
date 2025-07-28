"""
API Tests for LLMind FastAPI endpoints
Tests all REST API endpoints and WebSocket functionality
"""
import pytest
import json
import io
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health and system endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "app_name" in data
        assert "version" in data

    def test_system_info(self, client):
        """Test system info endpoint"""
        response = client.get("/api/system/info")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "system" in data
        assert "settings" in data


class TestModelEndpoints:
    """Test MLX model management endpoints"""
    
    @patch('main.mlx_manager')
    def test_get_models(self, mock_manager, client):
        """Test get available models endpoint"""
        mock_manager.get_available_models = AsyncMock(return_value=[
            {"name": "test-model", "status": "available", "size": "1GB"}
        ])
        mock_manager.get_model_info = lambda: {"name": "test-model", "status": "loaded"}
        
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "models" in data
        assert "current_model" in data

    @patch('main.mlx_manager')
    def test_download_model(self, mock_manager, client):
        """Test model download endpoint"""
        mock_manager.download_model = AsyncMock()
        
        response = client.post("/api/models/download", data={"model_name": "test-model"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_manager.download_model.assert_called_once_with("test-model")

    @patch('main.mlx_manager')
    def test_switch_model(self, mock_manager, client):
        """Test model switch endpoint"""
        mock_manager.switch_model = AsyncMock()
        
        response = client.post("/api/models/switch", data={"model_name": "test-model"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_manager.switch_model.assert_called_once_with("test-model")


class TestDocumentEndpoints:
    """Test document management endpoints"""
    
    @patch('main.document_processor')
    @patch('main.vector_store')
    def test_upload_documents(self, mock_vector_store, mock_doc_processor, client):
        """Test document upload endpoint"""
        # Mock document processor
        mock_doc_processor.process_file = AsyncMock(return_value={
            "filename": "test.txt",
            "chunk_count": 1,
            "text_length": 100,
            "chunks": [{"text": "test content"}]
        })
        
        # Mock vector store
        mock_vector_store.add_documents = AsyncMock()
        
        # Create test file
        test_file = io.BytesIO(b"This is test content for document upload.")
        
        response = client.post(
            "/api/documents/upload",
            files={"files": ("test.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data

    @patch('main.document_processor')
    @patch('main.vector_store')
    def test_get_documents(self, mock_vector_store, mock_doc_processor, client):
        """Test get documents endpoint"""
        mock_doc_processor.get_document_list = AsyncMock(return_value=[
            {"filename": "test.txt", "chunk_count": 1}
        ])
        mock_vector_store.get_document_list = AsyncMock(return_value=[
            {"filename": "test.txt", "chunk_count": 1}
        ])
        
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "processed_documents" in data

    @patch('main.document_processor')
    @patch('main.vector_store')
    def test_delete_document(self, mock_vector_store, mock_doc_processor, client):
        """Test document deletion endpoint"""
        mock_doc_processor.get_document_by_hash = AsyncMock(return_value={
            "filename": "test.txt"
        })
        mock_doc_processor.delete_document = AsyncMock(return_value=True)
        mock_vector_store.delete_document = AsyncMock(return_value=True)
        
        response = client.delete("/api/documents/test_hash")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestChatEndpoints:
    """Test chat functionality endpoints"""
    
    @patch('main.chat_engine')
    def test_chat_endpoint(self, mock_chat_engine, client):
        """Test chat response endpoint"""
        mock_chat_engine.generate_response = AsyncMock(return_value={
            "response": "Test response",
            "sources": [],
            "metadata": {"response_time": 0.5}
        })
        
        response = client.post("/api/chat", data={"message": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "response" in data

    @patch('main.chat_engine')
    def test_get_chat_history(self, mock_chat_engine, client):
        """Test get chat history endpoint"""
        mock_chat_engine.get_conversation_history = lambda: [
            {"type": "user", "message": "Hello"}
        ]
        mock_chat_engine.get_stats = lambda: {"total_conversations": 1}
        
        response = client.get("/api/chat/history")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "history" in data
        assert "stats" in data

    @patch('main.chat_engine')
    def test_clear_chat_history(self, mock_chat_engine, client):
        """Test clear chat history endpoint"""
        mock_chat_engine.clear_conversation_history = lambda: None
        
        response = client.delete("/api/chat/history")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestAudioEndpoints:
    """Test audio/voice functionality endpoints"""
    
    @patch('main.audio_manager')
    def test_get_voices(self, mock_audio_manager, client):
        """Test get available voices endpoint"""
        mock_audio_manager.get_voices = AsyncMock(return_value=[
            {"name": "Samantha", "language": "en_US"}
        ])
        mock_audio_manager.get_audio_settings = lambda: {
            "current_voice": "Samantha",
            "whisper_available": True
        }
        
        response = client.get("/api/audio/voices")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "voices" in data

    @patch('main.audio_manager')
    def test_text_to_speech(self, mock_audio_manager, client):
        """Test text-to-speech endpoint"""
        mock_audio_manager.text_to_speech = AsyncMock(return_value="base64_audio_data")
        mock_audio_manager.current_voice = "Samantha"
        
        response = client.post("/api/audio/tts", data={"text": "Hello world"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "audio" in data

    @patch('main.audio_manager')
    def test_speech_to_text(self, mock_audio_manager, client):
        """Test speech-to-text endpoint"""
        mock_audio_manager.speech_to_text = AsyncMock(return_value="Hello world")
        
        response = client.post("/api/audio/stt", data={"audio": "base64_audio_data"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "text" in data


class TestPerformanceEndpoints:
    """Test performance monitoring endpoints"""
    
    @patch('main.mlx_manager')
    @patch('main.chat_engine')
    @patch('main.vector_store')
    @patch('main.document_processor')
    @patch('main.audio_manager')
    def test_performance_metrics(self, mock_audio, mock_doc, mock_vector, mock_chat, mock_mlx, client):
        """Test performance metrics endpoint"""
        # Setup mocks
        mock_mlx.get_tokens_per_second = lambda: 50.0
        mock_mlx.get_avg_response_time = lambda: 0.3
        mock_mlx.get_memory_usage = lambda: {"used_gb": 1.5}
        mock_mlx.get_gpu_utilization = lambda: 70.0
        
        mock_chat.get_avg_response_time = lambda: 0.5
        mock_chat.conversation_history = []
        
        mock_vector.get_size = AsyncMock(return_value={
            "total_documents": 5,
            "total_vectors": 50
        })
        
        mock_doc.get_document_count = AsyncMock(return_value=5)
        mock_audio.get_system_audio_info = AsyncMock(return_value={"system": "Darwin"})
        
        response = client.get("/api/performance")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "model_performance" in data
        assert "vector_store" in data

    @patch('main.mlx_manager')
    def test_update_settings(self, mock_mlx_manager, client):
        """Test settings update endpoint"""
        mock_mlx_manager.update_temperature = lambda x: None
        mock_mlx_manager.update_max_tokens = lambda x: None
        
        settings_data = {
            "temperature": 0.8,
            "max_tokens": 1000,
            "chunk_size": 256,
            "top_k_results": 3
        }
        
        response = client.post("/api/settings/update", json=settings_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestWebSocketConnection:
    """Test WebSocket functionality"""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws") as websocket:
            # Test connection is established
            assert websocket is not None

    @patch('main.chat_engine')
    def test_websocket_chat(self, mock_chat_engine, client):
        """Test WebSocket chat functionality"""
        from tests.conftest import AsyncIterator
        
        # Mock streaming response
        mock_chat_engine.stream_response = AsyncMock(return_value=AsyncIterator([
            "Hello", " there", "!"
        ]))
        
        with client.websocket_connect("/ws") as websocket:
            # Send chat message
            websocket.send_text(json.dumps({
                "type": "chat",
                "message": "Hello",
                "context": []
            }))
            
            # Should receive chat_start message
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "chat_start"

    @patch('main.audio_manager')
    def test_websocket_voice(self, mock_audio_manager, client):
        """Test WebSocket voice functionality"""
        mock_audio_manager.speech_to_text = AsyncMock(return_value="Hello world")
        
        with client.websocket_connect("/ws") as websocket:
            # Send voice message
            websocket.send_text(json.dumps({
                "type": "voice",
                "audio": "base64_audio_data"
            }))
            
            # Should receive transcription
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "voice_transcription"
            assert message["text"] == "Hello world"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self, client):
        """Test handling of invalid endpoints"""
        response = client.get("/api/invalid")
        assert response.status_code == 404

    def test_malformed_request(self, client):
        """Test handling of malformed requests"""
        response = client.post("/api/chat")  # Missing required data
        assert response.status_code == 422

    @patch('main.mlx_manager')
    def test_model_error_handling(self, mock_manager, client):
        """Test model operation error handling"""
        mock_manager.download_model = AsyncMock(side_effect=Exception("Download failed"))
        
        response = client.post("/api/models/download", data={"model_name": "invalid-model"})
        assert response.status_code == 500

    def test_large_file_upload(self, client):
        """Test handling of large file uploads"""
        # Create a large file (simulated)
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB
        large_file = io.BytesIO(large_content)
        
        response = client.post(
            "/api/documents/upload",
            files={"files": ("large.txt", large_file, "text/plain")}
        )
        
        # Should handle gracefully (either process or reject cleanly)
        assert response.status_code in [200, 413, 422, 500] 