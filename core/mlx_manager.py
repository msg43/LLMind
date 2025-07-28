"""
MLX Model Manager - High-performance local LLM inference with Apple Silicon optimization
Optimized for M2 Max with 128GB RAM
"""
import asyncio
import time
from typing import List, Dict, Optional, AsyncGenerator
import psutil
import subprocess
from pathlib import Path

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  MLX not available. Please install: pip install mlx-lm")

from config import settings

class MLXManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLXManager, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        # Only initialize once
        if MLXManager._initialized:
            return
            
        self.current_model = None
        self.tokenizer = None
        self.current_model_name = None
        self.model_cache = {}
        self.performance_stats = {
            "tokens_per_second": 0,
            "response_times": [],
            "total_tokens": 0
        }
        
        # Mark as initialized
        MLXManager._initialized = True
        
    async def initialize(self):
        """Initialize MLX manager (lazy model loading for faster startup)"""
        if not MLX_AVAILABLE:
            raise Exception("MLX not available. Please install mlx-lm package.")
            
        print("🧠 Initializing MLX Manager (lazy loading enabled)...")
        
        # Just verify model availability, don't load yet
        available_models = await self.get_available_models()
        if settings.default_model not in [m["name"] for m in available_models]:
            print(f"📥 Default model {settings.default_model} will be downloaded on first use")
        else:
            print(f"✅ Default model {settings.default_model} available for lazy loading")
        
        print("✅ MLX Manager initialized successfully (models will load on demand)!")
    
    async def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available MLX models with metadata"""
        try:
            # Check local MLX models directory
            mlx_cache = Path.home() / ".cache" / "huggingface" / "hub"
            local_models = []
            
            if mlx_cache.exists():
                for model_dir in mlx_cache.iterdir():
                    if model_dir.is_dir() and "mlx-community" in model_dir.name:
                        model_name = model_dir.name.replace("models--", "").replace("--", "/")
                        local_models.append({
                            "name": model_name,
                            "status": "downloaded",
                            "size": "Unknown",
                            "type": "local"
                        })
            
            # Add popular models that can be downloaded (no authentication required)
            popular_models = [
                {
                    "name": "mlx-community/stablelm-2-zephyr-1_6b-4bit",
                    "status": "available",
                    "size": "~1GB",
                    "type": "fast",
                    "description": "Fast and efficient 1.6B model"
                },
                {
                    "name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", 
                    "status": "available",
                    "size": "~0.3GB",
                    "type": "ultra-fast",
                    "description": "Ultra-lightweight model for basic tasks"
                },
                {
                    "name": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                    "status": "available", 
                    "size": "~1GB",
                    "type": "balanced",
                    "description": "Good balance of speed and quality"
                },
                {
                    "name": "mlx-community/Qwen2.5-3B-Instruct-4bit",
                    "status": "available",
                    "size": "~2GB", 
                    "type": "powerful",
                    "description": "Enhanced performance and capabilities"
                },
                {
                    "name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "status": "available",
                    "size": "~4GB",
                    "type": "advanced",
                    "description": "High-performance model with excellent quality"
                }
            ]
            
            # Merge local and available models
            local_names = [m["name"] for m in local_models]
            for popular in popular_models:
                if popular["name"] not in local_names:
                    local_models.append(popular)
                else:
                    # Update status for downloaded models
                    for local in local_models:
                        if local["name"] == popular["name"]:
                            local.update(popular)
                            local["status"] = "downloaded"
                            break
            
            return sorted(local_models, key=lambda x: x["name"])
            
        except Exception as e:
            print(f"Error getting available models: {e}")
            return [{
                "name": "mlx-community/Llama-3.1-8B-Instruct-4bit",
                "status": "available",
                "size": "~5GB",
                "type": "fallback"
            }]
    
    async def download_model(self, model_name: str):
        """Download MLX model"""
        try:
            print(f"📥 Downloading model: {model_name}")
            
            # Ensure models directory exists
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Use python -m mlx_lm.convert to download and convert the model
            process = await asyncio.create_subprocess_exec(
                "python", "-m", "mlx_lm.convert", 
                "--hf-path", model_name,
                "--mlx-path", f"models/{model_name.replace('/', '_')}",
                "--quantize",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ Model {model_name} downloaded and converted successfully!")
                print(f"📥 Output: {stdout.decode()}")
            else:
                error_msg = stderr.decode() or stdout.decode()
                print(f"❌ Download failed: {error_msg}")
                # Try alternative approach with mlx_lm.load which auto-downloads
                print(f"🔄 Trying alternative download method...")
                await self._download_with_load(model_name)
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                print(f"❌ Authentication Error: Model {model_name} requires Hugging Face authentication")
                print(f"💡 Solution: Set up your HF token in the Settings tab, or choose a different model")
                print(f"📋 Available models that don't require authentication:")
                for model in ["mlx-community/Qwen2.5-1.5B-Instruct-4bit", 
                             "mlx-community/stablelm-2-zephyr-1_6b-4bit"]:
                    print(f"   • {model}")
                raise Exception(f"Authentication required for {model_name}. Please set up HF token or use a different model.")
            else:
                print(f"❌ Error downloading model {model_name}: {e}")
                # Try fallback method
                print(f"🔄 Trying fallback download method...")
                await self._download_with_load(model_name)
    
    async def _download_with_load(self, model_name: str):
        """Fallback download method using mlx_lm.load"""
        try:
            print(f"📥 Fallback: Using mlx_lm.load to download {model_name}")
            
            # Run in a subprocess to avoid blocking
            process = await asyncio.create_subprocess_exec(
                "python", "-c", 
                f"""
import mlx_lm
import os

print('Starting download...')
try:
    # Load model which will auto-download if not present
    model, tokenizer = mlx_lm.load('{model_name}')
    print('Model loaded successfully!')
    
    # Clear from memory immediately to save RAM
    del model, tokenizer
    print('Model cleared from memory - download complete!')
    
except Exception as e:
    print(f'Download error: {{e}}')
    raise
""",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ Model {model_name} downloaded successfully via fallback method!")
                print(f"📥 Output: {stdout.decode()}")
            else:
                error_msg = stderr.decode() or stdout.decode()
                print(f"❌ Fallback download also failed: {error_msg}")
                raise Exception(f"Download failed: {error_msg}")
                
        except Exception as e:
            print(f"❌ Error in fallback download for {model_name}: {e}")
            raise
    
    async def switch_model(self, model_name: str):
        """Switch to a different MLX model"""
        try:
            print(f"🔄 Switching to model: {model_name}")
            
            # Check if model is already loaded
            if model_name in self.model_cache:
                self.current_model, self.tokenizer = self.model_cache[model_name]
                self.current_model_name = model_name
                print(f"✅ Switched to cached model: {model_name}")
                return
            
            # Load new model
            start_time = time.time()
            model, tokenizer = load(model_name)
            load_time = time.time() - start_time
            
            # Cache the model (keep only 2 models in cache to save memory)
            if len(self.model_cache) >= 2:
                # Remove oldest model
                oldest_model = next(iter(self.model_cache))
                del self.model_cache[oldest_model]
                print(f"🗑️  Removed cached model: {oldest_model}")
            
            self.model_cache[model_name] = (model, tokenizer)
            self.current_model = model
            self.tokenizer = tokenizer
            self.current_model_name = model_name
            
            print(f"✅ Model {model_name} loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error switching to model {model_name}: {e}")
            raise
    
    def generate_response_sync(self, prompt: str, max_tokens: int = None) -> str:
        """Synchronous generation optimized for DSPy integration - MAXIMUM PERFORMANCE"""
        if not self.current_model or not self.tokenizer:
            # Lazy load default model on first use
            print(f"🔄 Loading default model on first use: {settings.default_model}")
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.switch_model(settings.default_model))
            finally:
                loop.close()
        
        max_tokens = max_tokens or settings.max_tokens
        
        try:
            start_time = time.time()
            
            # CRITICAL: Direct MLX generation on main thread for maximum Metal acceleration
            response = generate(
                model=self.current_model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Minimal post-processing for speed
            response = self._clean_response(response)
            
            # Update performance stats
            estimated_tokens = len(response.split()) * 1.3
            tokens_per_second = estimated_tokens / response_time if response_time > 0 else 0
            
            self.performance_stats["response_times"].append(response_time)
            self.performance_stats["tokens_per_second"] = tokens_per_second
            self.performance_stats["total_tokens"] += estimated_tokens
            
            # Keep only last 10 response times
            if len(self.performance_stats["response_times"]) > 10:
                self.performance_stats["response_times"] = self.performance_stats["response_times"][-10:]
            
            return response
            
        except Exception as e:
            print(f"❌ Error in MLX generation: {e}")
            return "I apologize, but I encountered an error generating the response."

    async def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using MLX with FULL Metal GPU acceleration"""
        if not self.current_model or not self.tokenizer:
            raise Exception("No model loaded. Please switch to a model first.")
        
        max_tokens = max_tokens or settings.max_tokens
        
        # Intelligent max_tokens adjustment based on prompt complexity
        if max_tokens > 200:
            # For simple Q&A format, use shorter responses
            if prompt.startswith("Q:") and len(prompt.split()) < 20:
                max_tokens = min(max_tokens, 150)  # Cap simple questions
        
        try:
            start_time = time.time()
            
            # CRITICAL: MLX MUST run on main thread for Metal acceleration
            # Using thread executors forces CPU-only mode and kills performance!
            response = generate(
                model=self.current_model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,  # Use user's configured max_tokens
                verbose=False
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Post-process response to remove repetitive patterns
            response = self._clean_response(response)  # RE-ENABLED
            
            # Calculate tokens per second (approximate)
            estimated_tokens = len(response.split()) * 1.3  # Rough estimate
            tokens_per_second = estimated_tokens / response_time if response_time > 0 else 0
            
            # Update performance stats
            self.performance_stats["response_times"].append(response_time)
            self.performance_stats["tokens_per_second"] = tokens_per_second
            self.performance_stats["total_tokens"] += estimated_tokens
            
            # Keep only last 10 response times
            if len(self.performance_stats["response_times"]) > 10:
                self.performance_stats["response_times"] = self.performance_stats["response_times"][-10:]
            
            return response
            
        except Exception as e:
            print(f"❌ Error in MLX generation: {e}")
            return "I apologize, but I encountered an error generating the response."

    async def stream_response(self, prompt: str, max_tokens: int = None) -> AsyncGenerator[str, None]:
        """Stream response generation with Metal acceleration"""
        # Note: MLX doesn't have native streaming yet, so we simulate it
        try:
            # Use sync method directly for Metal GPU performance and to avoid async overhead
            response = self.generate_response_sync(prompt, max_tokens)
            
            # Split response into chunks and yield them
            words = response.split()
            chunk_size = 4  # Slightly larger chunks for better performance
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield chunk
                await asyncio.sleep(0.02)  # Faster streaming delay
                
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def get_tokens_per_second(self) -> float:
        """Get current tokens per second performance"""
        return self.performance_stats.get("tokens_per_second", 0)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "used_gb": memory_info.rss / (1024**3),
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "percentage": process.memory_percent()
        }
    
    def get_gpu_utilization(self) -> float:
        """Get GPU utilization (Metal on Apple Silicon)"""
        try:
            # On Apple Silicon, we can check GPU usage via powermetrics
            # For now, return estimated usage based on model activity
            if self.current_model:
                return min(70.0, self.performance_stats.get("tokens_per_second", 0) * 2)
            return 0.0
        except Exception:
            return 0.0
    
    def update_temperature(self, temperature: float):
        """Update model temperature"""
        settings.temperature = max(0.0, min(1.0, temperature))
        print(f"🌡️  Temperature updated to: {settings.temperature}")
    
    def update_max_tokens(self, max_tokens: int):
        """Update max tokens"""
        settings.max_tokens = max(1, min(8192, max_tokens))
        print(f"📝 Max tokens updated to: {settings.max_tokens}")
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        times = self.performance_stats.get("response_times", [])
        return sum(times) / len(times) if times else 0.0
    
    def get_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        return {
            "name": self.current_model_name or "None",
            "status": "loaded" if self.current_model else "not loaded",
            "cached_models": len(self.model_cache),
            "total_tokens": self.performance_stats.get("total_tokens", 0)
        }
    
    def _clean_response(self, response: str) -> str:
        """Clean response by removing repetitive patterns and unwanted content"""
        import re
        
        # Remove repetitive URLs and patterns
        response = re.sub(r'Read more: https?://[^\s]*', '', response)
        response = re.sub(r'#\w+\s*', '', response)  # Remove hashtags
        response = re.sub(r'\s*View all conversations\s*', '', response)
        
        # Stop at conversation continuation patterns
        stop_patterns = [
            r'\nUser:',
            r'\nHuman:',
            r'\nAssistant:',
            r'\n\[INST\]',
            r'\nQuestion:',
            r'\nAnswer:',
            r'\nQ:',
            r'\nA:',
            r' Q:',  # Q&A continuation with space
            r'Q: '   # Q&A continuation at start
        ]
        
        for pattern in stop_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                response = response[:match.start()].strip()
                break
        
        # Detect numbered repetition patterns like "1. was that the joke 2. was that the joke"
        numbered_pattern = re.search(r'(\d+\.\s+[^0-9]*?)\s*\d+\.\s+.*?\1', response, re.IGNORECASE | re.DOTALL)
        if numbered_pattern:
            # Keep only up to the first repetition
            response = response[:numbered_pattern.start() + len(numbered_pattern.group(1))].strip()
        
        # Detect and stop rambling/repetition
        sentences = response.split('. ')
        if len(sentences) > 2:
            # Check for repetitive content
            clean_sentences = []
            seen_content = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check for repetitive phrases
                sentence_key = ' '.join(sentence.lower().split()[:8])  # First 8 words
                
                if sentence_key not in seen_content:
                    clean_sentences.append(sentence)
                    seen_content.add(sentence_key)
                else:
                    # Found repetition, stop here
                    break
            
            response = '. '.join(clean_sentences)
            if response and not response.endswith('.'):
                response += '.'
        
        # Remove incomplete sentences at the end
        if response:
            # If response ends with an incomplete thought (common patterns)
            incomplete_patterns = [
                r'\s+If you are.*$',
                r'\s+However, if.*$', 
                r'\s+But if.*$',
                r'\s+The.*\s+(is|are|was|were)\s*$',
                r'\s+This.*\s+(is|are|was|were)\s*$',
                r'\s+\d+\.\s*$',  # Trailing numbered list items
                r'\s+was that.*$'  # Trailing "was that" phrases
            ]
            
            for pattern in incomplete_patterns:
                response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        # Final cleanup - ensure response ends properly
        response = response.strip()
        if response and not response.endswith(('.', '!', '?', '"')):
            # Find the last complete sentence
            last_period = response.rfind('.')
            last_exclamation = response.rfind('!')
            last_question = response.rfind('?')
            
            last_punct = max(last_period, last_exclamation, last_question)
            if last_punct > 0:
                response = response[:last_punct + 1].strip()
        
        return response if response else "I'd be happy to help with that!" 