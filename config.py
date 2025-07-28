"""
Configuration settings for LLMind chatbot
Optimized for Apple Silicon (M2 Max with 128GB RAM)
Enhanced with Hybrid Reasoning System
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    app_name: str = "LLMind"
    app_version: str = "1.1.0"  # Updated for hybrid reasoning
    debug: bool = True
    
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    
    # Paths
    base_dir: Path = Path(__file__).parent.absolute()
    models_dir: Path = base_dir / "models"
    documents_dir: Path = base_dir / "documents"
    vector_store_dir: Path = base_dir / "vector_store"
    static_dir: Path = base_dir / "static"
    templates_dir: Path = base_dir / "templates"
    
    # MLX Model Settings (Apple Silicon Optimized)
    default_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    max_tokens: int = 512  # Base setting - strategies can override
    temperature: float = 0.8  # Higher to reduce repetition
    context_length: int = 32768
    
    # Vector Database (FAISS with Metal acceleration)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_results: int = 5
    batch_size: int = 32  # Embedding batch size for performance
    
    # Audio Settings (Local processing) - Professional voices only
    tts_voice: str = "Victoria"  # Professional female voice (default)
    tts_rate: int = 165  # Professional speaking pace
    stt_language: str = "en"
    audio_sample_rate: int = 44100  # Higher quality audio (was 16000)
    whisper_model_size: str = "base"  # Better accuracy than tiny (was tiny)
    
    # Prompt Configuration (Legacy - now managed by hybrid reasoning)
    prompt_format: str = "qa"  # qa, assistant, instruction, conversation, custom
    system_prompt: str = "You are a helpful AI assistant."
    custom_template: str = "Q: {user_message}\nA:"
    
    # === HYBRID REASONING SYSTEM ===
    
    # DSPy Integration
    enable_dspy: bool = True
    dspy_optimization_enabled: bool = True
    dspy_cache_size: int = 1000
    dspy_bootstrap_demos: int = 4
    dspy_labeled_demos: int = 8
    
    # Hybrid Stack Configuration
    default_reasoning_stack: str = "auto"  # auto, speed_optimized, quality_optimized, analytical, contextual, conversational
    enable_strategy_performance_tracking: bool = True
    strategy_confidence_threshold: float = 0.7
    enable_automatic_stack_switching: bool = False  # Experimental feature
    
    # Strategy-Specific Settings
    
    # Fast Path Strategy
    fast_path_max_words: int = 15
    fast_path_confidence_threshold: float = 0.8
    fast_path_max_tokens: int = 150
    fast_path_bypass_vector_search: bool = True
    
    # ReAct Strategy  
    react_max_steps: int = 5
    react_confidence_threshold: float = 0.7
    react_enable_reflection: bool = True
    react_max_tokens_per_step: int = 200
    
    # Chain of Thought Strategy
    cot_max_steps: int = 6
    cot_confidence_threshold: float = 0.6
    cot_enable_step_validation: bool = True
    cot_max_tokens_per_step: int = 150
    
    # Query Decomposition Strategy
    decomp_max_subquestions: int = 5
    decomp_confidence_threshold: float = 0.6
    decomp_enable_prioritization: bool = True
    decomp_max_tokens_per_subquestion: int = 200
    
    # Contextual Reasoning Strategy
    context_min_length: int = 100
    context_max_chunks: int = 5
    context_confidence_threshold: float = 0.7
    context_relevance_threshold: float = 0.6
    context_enable_synthesis: bool = True
    
    # Performance Optimization
    enable_query_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_reasoning: int = 3
    reasoning_timeout_seconds: int = 30
    
    # Performance (Apple Silicon M2 Max Optimized)
    use_metal: bool = True
    batch_size: int = 16
    cache_size: int = 1000
    max_memory_gb: int = 64  # Adjust based on your 128GB RAM
    
    # Voice Features
    enable_voice: bool = True
    voice_activation: bool = True
    auto_play_responses: bool = True
    
    # Experimental Features
    enable_reasoning_explanation: bool = False  # Show reasoning steps to user
    enable_confidence_display: bool = False     # Show confidence scores
    enable_strategy_suggestions: bool = False   # Suggest better strategies
    enable_learning_mode: bool = False          # Learn from user feedback
    
    model_config = {
        "protected_namespaces": (),
        "env_file": ".env", 
        "case_sensitive": False
    }

# Create directories on import
settings = Settings()
for directory in [
    settings.models_dir,
    settings.documents_dir, 
    settings.vector_store_dir,
    settings.static_dir,
    settings.templates_dir
]:
    directory.mkdir(exist_ok=True)

print(f"🏗️  LLMind Enhanced directories created at: {settings.base_dir}") 
print(f"🧠 Hybrid Reasoning: {'enabled' if settings.enable_dspy else 'disabled'}")
print(f"🎯 Default Stack: {settings.default_reasoning_stack}") 