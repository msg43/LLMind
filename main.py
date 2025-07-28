"""
LLMind: High-performance local chatbot with MLX and FastAPI
Enhanced with DSPy-based Hybrid Reasoning System
Optimized for Apple Silicon (M2 Max) with GUI control and voice capabilities
"""
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import json
import asyncio
from pathlib import Path
import logging

from config import settings
from core.mlx_manager import MLXManager
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.chat_engine import ChatEngine
from core.audio_manager import AudioManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize core components
mlx_manager = MLXManager()
document_processor = DocumentProcessor()
vector_store = VectorStore()
chat_engine = ChatEngine(mlx_manager, vector_store)
audio_manager = AudioManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}...")
    
    # Start initialization in background tasks - don't block the application startup
    async def initialize_components():
        """Initialize components in background"""
        try:
            logger.info("🔧 Initializing core components in background...")
            
            # Initialize core components
            await mlx_manager.initialize()
            await vector_store.initialize()
            await audio_manager.initialize()
            
            # Initialize enhanced chat engine with hybrid reasoning
            await chat_engine.initialize()
            
            logger.info(f"✅ {settings.app_name} fully initialized!")
            logger.info(f"🎤 Voice features: {'enabled' if settings.enable_voice else 'disabled'}")
            logger.info(f"🧠 Hybrid reasoning: {'enabled' if settings.enable_dspy else 'disabled'}")
            logger.info(f"🎯 Reasoning stack: {settings.default_reasoning_stack}")
            
        except Exception as e:
            logger.error(f"❌ Component initialization error: {e}")
            # Don't crash the app, just log the error
    
    # Start background initialization
    initialization_task = asyncio.create_task(initialize_components())
    
    logger.info(f"✅ {settings.app_name} web interface started!")
    logger.info(f"📊 Dashboard available at: http://{settings.host}:{settings.port}")
    logger.info(f"⏳ Core components initializing in background...")
    
    yield
    
    # Shutdown cleanup
    logger.info("🛑 Shutting down components...")
    # Cancel initialization if still running
    if not initialization_task.done():
        initialization_task.cancel()
        try:
            await initialization_task
        except asyncio.CancelledError:
            logger.info("🛑 Background initialization cancelled")

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="High-performance local chatbot with MLX, document processing, voice capabilities, and advanced hybrid reasoning",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Static files and templates
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
templates = Jinja2Templates(directory=settings.templates_dir)

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    try:
        # Get current system status - handle components that may still be initializing
        try:
            models = await mlx_manager.get_available_models()
            model_info = mlx_manager.get_model_info()
            performance = {
                "tokens_per_second": mlx_manager.get_tokens_per_second(),
                "memory_usage": mlx_manager.get_memory_usage(),
                "avg_response_time": chat_engine.get_avg_response_time()
            }
        except Exception as e:
            logger.warning(f"MLX Manager not ready: {e}")
            models = []
            model_info = {"name": "Initializing...", "status": "loading"}
            performance = {
                "tokens_per_second": 0,
                "memory_usage": {"used_gb": 0, "total_gb": 0, "percentage": 0},
                "avg_response_time": 0
            }
        
        try:
            vector_stats = await vector_store.get_size()
            doc_count = await document_processor.get_document_count()
        except Exception as e:
            logger.warning(f"Vector Store not ready: {e}")
            vector_stats = {"total_vectors": 0, "memory_usage_mb": 0}
            doc_count = 0
        
        try:
            audio_settings = audio_manager.get_audio_settings()
        except Exception as e:
            logger.warning(f"Audio Manager not ready: {e}")
            audio_settings = {
                "current_voice": "Initializing...",
                "tts_rate": 200,
                "whisper_available": False,
                "voices_count": 0
            }
        
        # Get hybrid reasoning stats
        try:
            reasoning_stacks = chat_engine.get_available_reasoning_stacks()
            current_stack = chat_engine.get_current_reasoning_stack()
            reasoning_stats = chat_engine.get_reasoning_performance_stats()
        except Exception as e:
            logger.warning(f"Chat Engine not ready: {e}")
            reasoning_stacks = {
                "auto": {
                    "name": "Auto-Select",
                    "description": "Initializing reasoning system...",
                    "strategy_order": []
                }
            }
            current_stack = "auto"
            reasoning_stats = {
                "overall_stats": {
                    "total_queries": 0,
                    "average_processing_time": 0.0,
                    "success_rate": 0.0
                }
            }
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "app_name": settings.app_name,
            "version": settings.app_version,
            "models": models,
            "current_model": model_info,
            "performance": performance,
            "vector_stats": vector_stats,
            "document_count": doc_count,
            "audio_settings": audio_settings,
            "reasoning_stacks": reasoning_stacks,
            "current_stack": current_stack,
            "reasoning_stats": reasoning_stats,
            "settings": {
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "chunk_size": settings.chunk_size,
                "top_k_results": settings.top_k_results,
                "enable_dspy": settings.enable_dspy,
                "default_reasoning_stack": settings.default_reasoning_stack
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time communication"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "chat":
                # Check if chat engine is ready
                if not hasattr(chat_engine, 'hybrid_manager') or not chat_engine.hybrid_manager:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Chat engine is still initializing. Please wait a moment and try again."
                    }))
                    continue
                
                # Stream chat response with hybrid reasoning
                await websocket.send_text(json.dumps({
                    "type": "chat_start",
                    "message": "Processing your question with hybrid reasoning..."
                }))
                
                try:
                    conversation_context = message_data.get("context", [])
                    metadata = message_data.get("metadata", {})
                    
                    # Handle story requests with higher token limits
                    max_tokens_override = None
                    if metadata.get("isStoryRequest", False):
                        max_tokens_override = metadata.get("preferredTokens", 1500)
                        logger.info(f"📝 Story request detected - using {max_tokens_override} tokens")
                    
                    async for chunk in chat_engine.stream_response(
                        message_data["message"], 
                        conversation_context,
                        max_tokens_override=max_tokens_override
                    ):
                        await websocket.send_text(json.dumps({
                            "type": "chat_chunk",
                            "content": chunk
                        }))
                    
                    await websocket.send_text(json.dumps({
                        "type": "chat_end"
                    }))
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Chat error: {str(e)}"
                    }))
            
            elif message_data["type"] == "voice":
                # Check if audio manager is ready
                if not audio_manager.whisper_model:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Audio processing is still initializing. Please wait a moment and try again."
                    }))
                    continue
                
                try:
                    # Process voice input
                    audio_data = message_data["audio"]
                    text = await audio_manager.speech_to_text(audio_data)
                    await websocket.send_text(json.dumps({
                        "type": "voice_transcription", 
                        "text": text
                    }))
                except Exception as e:
                    logger.error(f"Voice processing error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Voice processing error: {str(e)}"
                    }))
            
            elif message_data["type"] == "tts":
                # Check if audio manager is ready
                if not audio_manager.available_voices:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Text-to-speech is still initializing. Please wait a moment and try again."
                    }))
                    continue
                
                try:
                    # Generate text-to-speech
                    text = message_data["text"]
                    voice = message_data.get("voice", None)
                    audio_data = await audio_manager.text_to_speech(text, voice)
                    await websocket.send_text(json.dumps({
                        "type": "tts_audio",
                        "audio": audio_data
                    }))
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Text-to-speech error: {str(e)}"
                    }))
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# === CHAT ENDPOINTS ===

@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    context: str = Form(None)
):
    """Generate chat response with hybrid reasoning"""
    try:
        conversation_context = json.loads(context) if context else None
        result = await chat_engine.generate_response(message, conversation_context)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history")
async def get_chat_history():
    """Get conversation history"""
    try:
        history = chat_engine.get_conversation_history()
        stats = chat_engine.get_stats()
        return {
            "status": "success",
            "history": history,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"❌ Chat history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/history")
async def clear_chat_history():
    """Clear conversation history"""
    try:
        chat_engine.clear_conversation_history()
        return {"status": "success", "message": "Chat history cleared"}
    except Exception as e:
        logger.error(f"❌ Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === CHAT HISTORY MANAGEMENT ENDPOINTS ===

@app.get("/api/chats")
async def get_chat_list():
    """Get list of all saved chats"""
    try:
        chats = await chat_engine.get_chat_list()
        return {
            "status": "success",
            "chats": chats,
            "total_count": len(chats)
        }
    except Exception as e:
        logger.error(f"❌ Get chat list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get a specific chat by ID"""
    try:
        chat_data = await chat_engine.load_chat(chat_id)
        return {
            "status": "success",
            "chat": chat_data
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        logger.error(f"❌ Get chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a specific chat"""
    try:
        success = await chat_engine.delete_chat(chat_id)
        if success:
            return {"status": "success", "message": "Chat deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        logger.error(f"❌ Delete chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{chat_id}/export")
async def export_chat_markdown(chat_id: str):
    """Export a chat to markdown format"""
    try:
        markdown_content = await chat_engine.export_chat_to_markdown(chat_id)
        
        # Get chat data for filename
        chat_data = await chat_engine.load_chat(chat_id)
        safe_title = "".join(c for c in chat_data["title"] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title[:50]}.md"
        
        from fastapi.responses import Response
        return Response(
            content=markdown_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        logger.error(f"❌ Export chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats/new")
async def start_new_chat(chat_data: Dict[str, Any] = None):
    """Start a new chat session"""
    try:
        title = chat_data.get("title") if chat_data else None
        chat_id = chat_engine.start_new_chat(title)
        return {
            "status": "success",
            "chat_id": chat_id,
            "message": "New chat started"
        }
    except Exception as e:
        logger.error(f"❌ Start new chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === HYBRID REASONING ENDPOINTS ===

@app.get("/api/reasoning/stacks")
async def get_reasoning_stacks():
    """Get all available hybrid reasoning stacks"""
    try:
        stacks = chat_engine.get_available_reasoning_stacks()
        current = chat_engine.get_current_reasoning_stack()
        return {
            "status": "success",
            "stacks": stacks,
            "current_stack": current
        }
    except Exception as e:
        logger.error(f"❌ Get reasoning stacks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reasoning/stack")
async def switch_reasoning_stack(stack_data: Dict[str, str]):
    """Switch to a different hybrid reasoning stack"""
    try:
        stack_name = stack_data.get("stack_name")
        if not stack_name:
            raise HTTPException(status_code=400, detail="stack_name is required")
        
        success = await chat_engine.switch_reasoning_stack(stack_name)
        if success:
            return {"status": "success", "message": f"Switched to {stack_name} stack"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown stack: {stack_name}")
    except Exception as e:
        logger.error(f"❌ Switch reasoning stack error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reasoning/performance")
async def get_reasoning_performance():
    """Get comprehensive reasoning performance statistics"""
    try:
        stats = chat_engine.get_reasoning_performance_stats()
        return {"status": "success", "performance": stats}
    except Exception as e:
        logger.error(f"❌ Get reasoning performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reasoning/optimize")
async def optimize_reasoning(optimization_data: Dict[str, Any]):
    """Optimize reasoning system with example queries"""
    try:
        examples = optimization_data.get("examples", [])
        if not examples:
            raise HTTPException(status_code=400, detail="examples are required")
        
        success = await chat_engine.optimize_reasoning_for_examples(examples)
        if success:
            return {"status": "success", "message": "Reasoning system optimized successfully"}
        else:
            return {"status": "warning", "message": "Optimization completed with warnings"}
    except Exception as e:
        logger.error(f"❌ Optimize reasoning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reasoning/custom-stack")
async def create_custom_reasoning_stack(stack_data: Dict[str, Any]):
    """Create a custom hybrid reasoning stack"""
    try:
        name = stack_data.get("name")
        config = stack_data.get("config")
        
        if not name or not config:
            raise HTTPException(status_code=400, detail="name and config are required")
        
        success = await chat_engine.create_custom_reasoning_stack(name, config)
        if success:
            return {"status": "success", "message": f"Custom stack '{name}' created successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create custom stack")
    except Exception as e:
        logger.error(f"❌ Create custom reasoning stack error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reasoning/config")
async def update_reasoning_config(config_data: Dict[str, Any]):
    """Update hybrid reasoning configuration"""
    try:
        success = await chat_engine.update_reasoning_configuration(config_data)
        if success:
            return {"status": "success", "message": "Reasoning configuration updated"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update configuration")
    except Exception as e:
        logger.error(f"❌ Update reasoning config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === EXISTING ENDPOINTS CONTINUE... ===
# Document management endpoints
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        result = await document_processor.process_file(file)
        
        # Add to vector store
        await vector_store.add_documents([{
            "text": result["text"],
            "metadata": result["metadata"]
        }])
        
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"❌ Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents():
    """Get list of processed documents"""
    try:
        docs = await document_processor.get_document_list()
        count = await document_processor.get_document_count()
        vector_stats = await vector_store.get_size()
        
        return {
            "status": "success",
            "documents": docs,
            "total_count": count,
            "vector_stats": vector_stats
        }
    except Exception as e:
        logger.error(f"❌ Get documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = await document_processor.delete_document(document_id)
        if success:
            # Remove from vector store
            await vector_store.remove_document(document_id)
            return {"status": "success", "message": "Document deleted"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"❌ Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/add-monitored-folder")
async def add_monitored_folder(folder_data: Dict[str, str]):
    """Add a folder to continuous monitoring"""
    try:
        folder_path = folder_data.get("folder_path", "").strip()
        if not folder_path:
            raise HTTPException(status_code=400, detail="Folder path is required")
        
        result = await document_processor.add_monitored_folder(folder_path)
        
        if result["status"] == "success":
            logger.info(f"✅ Added monitored folder: {folder_path}")
        else:
            logger.warning(f"⚠️  Failed to add monitored folder: {result['message']}")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Add monitored folder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/remove-monitored-folder")
async def remove_monitored_folder(folder_data: Dict[str, str]):
    """Remove a folder from monitoring"""
    try:
        folder_path = folder_data.get("folder_path", "").strip()
        if not folder_path:
            raise HTTPException(status_code=400, detail="Folder path is required")
        
        result = await document_processor.remove_monitored_folder(folder_path)
        
        if result["status"] == "success":
            logger.info(f"✅ Removed monitored folder: {folder_path}")
        else:
            logger.warning(f"⚠️  Failed to remove monitored folder: {result['message']}")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Remove monitored folder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/monitoring-status")
async def get_monitoring_status():
    """Get current folder monitoring status"""
    try:
        result = await document_processor.get_monitoring_status()
        return result
        
    except Exception as e:
        logger.error(f"❌ Get monitoring status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/initialize-watcher")
async def initialize_document_watcher():
    """Initialize the document file watcher"""
    try:
        await document_processor.initialize_watcher()
        return {
            "status": "success",
            "message": "Document watcher initialized successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Initialize watcher error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === MODEL MANAGEMENT ENDPOINTS ===
@app.get("/api/models")
async def get_models():
    """Get available models"""
    try:
        models = await mlx_manager.get_available_models()
        current = mlx_manager.get_model_info()
        return {
            "status": "success",
            "models": models,
            "current_model": current
        }
    except Exception as e:
        logger.error(f"❌ Get models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/switch")
async def switch_model(model_name: str = Form(...)):
    """Switch to a different model"""
    try:
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        
        await mlx_manager.switch_model(model_name)
        return {"status": "success", "message": f"Switched to {model_name}"}
    except Exception as e:
        logger.error(f"❌ Switch model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/download")
async def download_model(model_name: str = Form(...)):
    """Download a new model"""
    try:
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        
        await mlx_manager.download_model(model_name)
        return {"status": "success", "message": f"Model {model_name} downloaded successfully"}
    except Exception as e:
        logger.error(f"❌ Download model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/test-hf-token")
async def test_hf_token(request: Request):
    """Test if a Hugging Face token is valid"""
    try:
        data = await request.json()
        token = data.get("token", "").strip()
        
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        
        # Test token by trying to access user info
        from huggingface_hub import whoami
        try:
            user_info = whoami(token=token)
            return {
                "status": "success", 
                "message": f"Token valid for user: {user_info.get('name', 'Unknown')}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Invalid token: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"❌ Test HF token error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === SYSTEM STATUS ENDPOINTS ===
@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        model_info = mlx_manager.get_model_info()
        performance = {
            "tokens_per_second": mlx_manager.get_tokens_per_second(),
            "memory_usage": mlx_manager.get_memory_usage(),
            "avg_response_time": chat_engine.get_avg_response_time()
        }
        
        vector_stats = await vector_store.get_size()
        doc_count = await document_processor.get_document_count()
        chat_stats = chat_engine.get_stats()
        
        return {
            "status": "success",
            "system": {
                "app_name": settings.app_name,
                "version": settings.app_version,
                "model": model_info,
                "performance": performance,
                "documents": {"count": doc_count, "vector_stats": vector_stats},
                "chat": chat_stats,
                "reasoning": chat_engine.get_reasoning_performance_stats()
            }
        }
    except Exception as e:
        logger.error(f"❌ System status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === PROMPT CONFIGURATION ENDPOINTS ===
@app.get("/api/prompts/config")
async def get_prompt_config():
    """Get current prompt configuration"""
    try:
        return {
            "status": "success",
            "config": {
                "prompt_format": settings.prompt_format,
                "system_prompt": settings.system_prompt,
                "custom_template": settings.custom_template
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompts/config")
async def update_prompt_config(config_data: Dict[str, Any]):
    """Update prompt configuration"""
    try:
        if "prompt_format" in config_data:
            settings.prompt_format = config_data["prompt_format"]
        if "system_prompt" in config_data:
            settings.system_prompt = config_data["system_prompt"]
        if "custom_template" in config_data:
            settings.custom_template = config_data["custom_template"]
        
        return {"status": "success", "message": "Prompt configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompts/test")
async def test_prompt_config(test_data: Dict[str, Any]):
    """Test prompt configuration with a sample query"""
    try:
        user_message = test_data.get("user_message", "Hello, how are you?")
        temp_format = test_data.get("prompt_format", settings.prompt_format)
        temp_system = test_data.get("system_prompt", settings.system_prompt)
        temp_template = test_data.get("custom_template", settings.custom_template)
        
        # Temporarily update settings
        old_format = settings.prompt_format
        old_system = settings.system_prompt
        old_template = settings.custom_template
        
        settings.prompt_format = temp_format
        settings.system_prompt = temp_system
        settings.custom_template = temp_template
        
        try:
            # Generate test response
            result = await chat_engine.generate_response(user_message)
            
            return {
                "status": "success",
                "test_response": result["response"],
                "prompt_used": result["metadata"].get("prompt_length", 0)
            }
        finally:
            # Restore original settings
            settings.prompt_format = old_format
            settings.system_prompt = old_system
            settings.custom_template = old_template
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts/presets")
async def get_prompt_presets():
    """Get predefined prompt presets"""
    try:
        presets = {
            "qa": {
                "name": "Q&A Format",
                "description": "Simple question and answer format",
                "system_prompt": "You are a helpful AI assistant. Give direct, concise answers.",
                "template": "Q: {user_message}\nA:"
            },
            "assistant": {
                "name": "Assistant Format", 
                "description": "Conversational assistant format",
                "system_prompt": "You are a helpful, friendly AI assistant. Be concise and direct in your responses.",
                "template": "{system_prompt}\n\nHuman: {user_message}\nAssistant:"
            },
            "instruction": {
                "name": "Instruction Format",
                "description": "Llama instruction format with tags",
                "system_prompt": "You are a helpful AI assistant. Provide clear, direct answers without unnecessary elaboration.",
                "template": "<s>[INST] {system_prompt} {user_message} [/INST]"
            },
            "conversation": {
                "name": "Conversation Format",
                "description": "Multi-turn conversation format",
                "system_prompt": "You are a helpful AI assistant. Keep responses focused and concise.",
                "template": "### System: {system_prompt}\n### User: {user_message}\n### Assistant:"
            }
        }
        
        return {"status": "success", "presets": presets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === VOICE/AUDIO ENDPOINTS ===

@app.get("/api/audio/voices")
async def get_voices():
    """Get available TTS voices"""
    try:
        voices = await audio_manager.get_voices()
        settings_info = audio_manager.get_audio_settings()
        return {
            "status": "success",
            "voices": voices,
            "current_settings": settings_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audio/tts")
async def text_to_speech(
    text: str = Form(...), 
    voice: str = Form(None)
):
    """Convert text to speech"""
    try:
        audio_data = await audio_manager.text_to_speech(text, voice)
        return {
            "status": "success",
            "audio": audio_data,
            "voice": voice or audio_manager.current_voice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audio/stt")
async def speech_to_text(audio: str = Form(...)):
    """Convert speech to text"""
    try:
        text = await audio_manager.speech_to_text(audio)
        return {
            "status": "success",
            "text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audio/settings")
async def update_audio_settings(settings_data: Dict[str, Any]):
    """Update audio settings"""
    try:
        await audio_manager.update_settings(settings_data)
        new_settings = audio_manager.get_audio_settings()
        return {
            "status": "success",
            "settings": new_settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/optimize")
async def optimize_voice_settings(preference: Dict[str, str]):
    """Get optimized voice settings based on user preference"""
    try:
        voice_preference = preference.get("preference", "female")
        recommendations = await audio_manager.optimize_voice_settings(voice_preference)
        
        return {
            "status": "success",
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"❌ Voice optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-hf-token")
async def test_huggingface_token(token_data: Dict[str, str]):
    """Test if a Hugging Face token is valid"""
    try:
        import requests
        token = token_data.get("token", "").strip()
        
        if not token:
            return {
                "status": "error",
                "message": "No token provided"
            }
        
        # Test token by making a simple API call
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            return {
                "status": "success",
                "message": f"Valid token for user: {user_info.get('name', 'Unknown')}"
            }
        else:
            return {
                "status": "error",
                "message": "Invalid token or authentication failed"
            }
            
    except Exception as e:
        logger.error(f"❌ HF token test error: {e}")
        return {
            "status": "error", 
            "message": f"Error testing token: {str(e)}"
        }

# === PERFORMANCE AND SYSTEM ENDPOINTS ===

@app.get("/api/performance")
async def get_performance_metrics():
    """Get real-time performance metrics"""
    try:
        return {
            "status": "success",
            "model_performance": {
                "tokens_per_second": mlx_manager.get_tokens_per_second(),
                "avg_response_time": mlx_manager.get_avg_response_time(),
                "memory_usage": mlx_manager.get_memory_usage(),
                "gpu_utilization": mlx_manager.get_gpu_utilization()
            },
            "chat_performance": {
                "avg_response_time": chat_engine.get_avg_response_time(),
                "total_conversations": len(chat_engine.conversation_history) // 2
            },
            "vector_store": await vector_store.get_size(),
            "document_count": await document_processor.get_document_count(),
            "audio_info": audio_manager.get_audio_settings()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings")
async def get_settings():
    """Get current application settings"""
    try:
        return {
            "status": "success",
            "settings": {
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "chunk_size": settings.chunk_size,
                "top_k_results": settings.top_k_results,
                "enable_dspy": settings.enable_dspy,
                "default_reasoning_stack": settings.default_reasoning_stack,
                "tts_voice": settings.tts_voice,
                "tts_rate": settings.tts_rate
            }
        }
    except Exception as e:
        logger.error(f"❌ Get settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings/update")
async def update_settings(settings_data: Dict[str, Any]):
    """Update application settings"""
    try:
        # Update model parameters
        if "temperature" in settings_data:
            mlx_manager.update_temperature(float(settings_data["temperature"]))
        if "max_tokens" in settings_data:
            mlx_manager.update_max_tokens(int(settings_data["max_tokens"]))
        if "chunk_size" in settings_data:
            settings.chunk_size = max(100, min(2000, int(settings_data["chunk_size"])))
        if "top_k_results" in settings_data:
            settings.top_k_results = max(1, min(20, int(settings_data["top_k_results"])))
        
        # Handle HF token
        if "hf_token" in settings_data:
            import os
            hf_token = str(settings_data["hf_token"]).strip()
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                # Also save to .env file for persistence
                env_file = Path(".env")
                env_lines = []
                if env_file.exists():
                    env_lines = env_file.read_text().splitlines()
                
                # Update or add HF_TOKEN
                token_found = False
                for i, line in enumerate(env_lines):
                    if line.startswith('HF_TOKEN='):
                        env_lines[i] = f'HF_TOKEN={hf_token}'
                        token_found = True
                        break
                
                if not token_found:
                    env_lines.append(f'HF_TOKEN={hf_token}')
                
                env_file.write_text('\n'.join(env_lines) + '\n')
            else:
                # Remove token if empty
                if 'HF_TOKEN' in os.environ:
                    del os.environ['HF_TOKEN']
        
        return {
            "status": "success", 
            "message": "Settings updated",
            "current_settings": {
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "chunk_size": settings.chunk_size,
                "top_k_results": settings.top_k_results
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === MLX PROFILING ENDPOINTS ===

@app.post("/api/profiling/run")
async def run_profiling_analysis(
    model_name: Optional[str] = None,
    quick_mode: bool = True,
    test_prompts: Optional[List[str]] = None
):
    """Run MLX profiling analysis"""
    try:
        # Import profiling components
        import sys
        sys.path.append('profiling')
        from profiling.mlx_profiler import MLXProfiler
        from profiling.mlx_optimization_guide import MLXOptimizationGuide
        
        # Initialize profiler
        profiler = MLXProfiler()
        await profiler.initialize()
        
        # Use default test prompts if none provided
        if not test_prompts:
            test_prompts = [
                "Hello, how are you today?",
                "Explain machine learning in simple terms.",
                "Write a brief analysis of renewable energy."
            ]
        
        # Switch model if specified
        if model_name and model_name != profiler.mlx_manager.current_model_name:
            await profiler.mlx_manager.switch_model(model_name)
        
        # Run profiling for each test prompt
        results = {}
        for i, prompt in enumerate(test_prompts):
            runs = 2 if quick_mode else 3
            prompt_results = profiler.profile_generation(prompt, runs=runs)
            results[f"test_{i+1}"] = {
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "results": [
                    {
                        "tokens_per_second": r.tokens_per_second,
                        "total_time": r.total_time,
                        "memory_usage": r.memory_usage,
                        "gpu_utilization": r.gpu_utilization,
                        "cpu_usage": r.cpu_usage
                    } for r in prompt_results
                ]
            }
        
        # Test optimization configurations
        optimizer = MLXOptimizationGuide()
        config_results = profiler.test_optimization_flags(test_prompts[1])
        
        # Analyze bottlenecks
        analysis = profiler.analyze_bottlenecks(profiler.results)
        
        # Get hardware recommendations
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        hardware_profile = "m2_max" if memory_gb >= 64 else "m2" if memory_gb >= 32 else "m1"
        recommendations = optimizer.get_model_recommendations(hardware_profile, memory_gb)
        
        # Generate report
        profiler.generate_report("static/mlx_performance_report.json")
        profiler.plot_performance("static/mlx_performance_plots.png")
        
        return {
            "status": "success",
            "profiling_results": results,
            "optimization_configs": {
                name: {
                    "tokens_per_second": config.tokens_per_second,
                    "total_time": config.total_time,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                } for name, config in config_results.items()
            },
            "analysis": analysis,
            "recommendations": recommendations,
            "report_url": "/static/mlx_performance_report.json",
            "plots_url": "/static/mlx_performance_plots.png",
            "system_info": {
                "memory_gb": memory_gb,
                "hardware_profile": hardware_profile,
                "current_model": profiler.mlx_manager.current_model_name
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")

@app.get("/api/profiling/status")
async def get_profiling_status():
    """Get current profiling status and basic metrics"""
    try:
        return {
            "status": "success",
            "current_metrics": {
                "tokens_per_second": mlx_manager.get_tokens_per_second(),
                "memory_usage": mlx_manager.get_memory_usage(),
                "gpu_utilization": mlx_manager.get_gpu_utilization(),
                "avg_response_time": mlx_manager.get_avg_response_time()
            },
            "model_info": mlx_manager.get_model_info(),
            "profiling_available": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/profiling/compare-models")
async def compare_models(models: List[str]):
    """Compare performance across multiple models"""
    try:
        # Import profiling components
        import sys
        sys.path.append('profiling')
        from profiling.mlx_profiler import MLXProfiler
        
        # Initialize profiler
        profiler = MLXProfiler()
        await profiler.initialize()
        
        # Test prompt for comparison
        test_prompt = "Explain the benefits of artificial intelligence in healthcare."
        comparison_results = {}
        
        for model in models:
            try:
                await profiler.mlx_manager.switch_model(model)
                results = profiler.profile_generation(test_prompt, runs=2)
                
                if results:
                    avg_tps = sum(r.tokens_per_second for r in results) / len(results)
                    avg_memory = sum(r.memory_usage["used_gb"] for r in results) / len(results)
                    
                    comparison_results[model] = {
                        "tokens_per_second": avg_tps,
                        "memory_usage": avg_memory,
                        "status": "success"
                    }
                else:
                    comparison_results[model] = {"status": "no_results"}
                    
            except Exception as e:
                comparison_results[model] = {"status": "error", "error": str(e)}
        
        return {
            "status": "success",
            "comparison_results": comparison_results,
            "test_prompt": test_prompt
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profiling/optimization-guide")
async def get_optimization_guide():
    """Get MLX optimization guide and recommendations"""
    try:
        import sys
        sys.path.append('profiling')
        from profiling.mlx_optimization_guide import MLXOptimizationGuide
        import psutil
        
        optimizer = MLXOptimizationGuide()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        hardware_profile = "m2_max" if memory_gb >= 64 else "m2" if memory_gb >= 32 else "m1"
        
        # Get optimized config for current hardware
        config = optimizer.get_optimized_generation_config(hardware_profile, "chat", "high")
        recommendations = optimizer.get_model_recommendations(hardware_profile, memory_gb)
        
        return {
            "status": "success",
            "optimization_techniques": optimizer.optimization_techniques,
            "mlx_flags": optimizer.mlx_flags,
            "hardware_config": config,
            "model_recommendations": recommendations,
            "hardware_profile": hardware_profile,
            "system_memory": memory_gb
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_status = "loaded" if mlx_manager.current_model else "not loaded"
        vector_status = "initialized" if vector_store.initialized else "not initialized"
        
        return {
            "status": "healthy",
            "app_name": settings.app_name,
            "version": settings.app_version,
            "model_status": model_status,
            "vector_store_status": vector_status,
            "audio_available": audio_manager.whisper_model is not None,
            "total_documents": await document_processor.get_document_count()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# === UTILITY ENDPOINTS ===

@app.get("/api/system/info")
async def get_system_info():
    """Get system information"""
    try:
        import platform
        import sys
        
        return {
            "status": "success",
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "app_version": settings.app_version,
                "mlx_available": mlx_manager.current_model is not None,
                "audio_available": audio_manager.whisper_model is not None
            },
            "settings": {
                "host": settings.host,
                "port": settings.port,
                "debug": settings.debug,
                "default_model": settings.default_model,
                "embedding_model": settings.embedding_model
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Broadcast updates to all connected WebSocket clients
async def broadcast_update(message: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients"""
    if active_connections:
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    ) 