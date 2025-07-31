"""
MLX Model Manager - High-performance local LLM inference with Apple Silicon optimization
Optimized for M2 Max with 128GB RAM
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import psutil

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import generate, load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  MLX not available. Please install: pip install mlx-lm")

try:
    from huggingface_hub import HfApi, list_models

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  Hugging Face Hub not available for model discovery")

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
        self.discovered_models = []  # Cache discovered models
        self.performance_stats = {
            "tokens_per_second": 0,
            "response_times": [],
            "total_tokens": 0,
        }

        # Mark as initialized
        MLXManager._initialized = True

    async def initialize(self):
        """Initialize MLX manager with automatic model discovery"""
        if not MLX_AVAILABLE:
            raise Exception("MLX not available. Please install mlx-lm package.")

        print("🧠 Initializing MLX Manager...")

        # Discover available models at startup
        try:
            print("🔍 Discovering MLX models from Hugging Face Hub...")
            self.discovered_models = await self._discover_mlx_models()
            print(f"✅ Discovered {len(self.discovered_models)} MLX models")
        except Exception as e:
            print(f"❌ Model discovery failed: {e}")
            print("📋 Using fallback model list")
            self.discovered_models = self._get_fallback_models()

        # Verify default model availability
        available_models = await self.get_available_models()
        if settings.default_model not in [m["name"] for m in available_models]:
            print(
                f"📥 Default model {settings.default_model} will be downloaded on first use"
            )
        else:
            print(
                f"✅ Default model {settings.default_model} available for lazy loading"
            )

        # Load the default model now so the system reports 'loaded' status
        try:
            print(f"🚀 Loading default model {settings.default_model} on startup...")
            await self.switch_model(settings.default_model)
            print("✅ Default model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load default model on startup: {e}")

    async def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available MLX models (uses cached discovery from startup)"""
        try:
            # Check local MLX models directory
            mlx_cache = Path.home() / ".cache" / "huggingface" / "hub"
            local_models = []

            if mlx_cache.exists():
                for model_dir in mlx_cache.iterdir():
                    if model_dir.is_dir() and "mlx-community" in model_dir.name:
                        model_name = model_dir.name.replace("models--", "").replace(
                            "--", "/"
                        )
                        local_models.append(
                            {
                                "name": model_name,
                                "status": "downloaded",
                                "size": "Unknown",
                                "type": "local",
                                "downloads": 0,
                            }
                        )

            # Use discovered models from startup
            discovered_models = self.discovered_models.copy()

            # Merge local and discovered models
            local_names = [m["name"] for m in local_models]
            for model in discovered_models:
                if model["name"] not in local_names:
                    local_models.append(model)
                else:
                    # Update status for downloaded models
                    for local in local_models:
                        if local["name"] == model["name"]:
                            local.update(model)
                            local["status"] = "downloaded"
                            break

            # Sort by downloads (descending)
            return sorted(
                local_models, key=lambda x: x.get("downloads", 0), reverse=True
            )

        except Exception as e:
            print(f"Error getting available models: {e}")
            return self._get_fallback_models()

    async def _discover_mlx_models(self) -> List[Dict[str, str]]:
        """Discover MLX models from Hugging Face Hub"""
        if not HF_HUB_AVAILABLE:
            print("⚠️  Hugging Face Hub not available, using fallback list")
            return self._get_fallback_models()

        try:
            api = HfApi()

            # Search for MLX community models
            models = list_models(
                author="mlx-community",
                task="text-generation",
                library="mlx",
                sort="downloads",
                direction=-1,
                limit=100,  # Get top 100 most downloaded
            )

            discovered = []
            for model in models:
                model_info = self._analyze_model(model)
                if model_info:
                    discovered.append(model_info)

            print(f"✅ Discovered {len(discovered)} MLX models")
            return discovered

        except Exception as e:
            print(f"❌ Error discovering models: {e}")
            return self._get_fallback_models()

    def _analyze_model(self, model) -> Optional[Dict[str, str]]:
        """Analyze a model and extract metadata"""
        try:
            name = model.modelId

            # Determine model size and type from name
            size_gb, model_type = self._classify_model(name)

            # Get model description
            description = self._generate_description(name, model_type)

            return {
                "name": name,
                "status": "available",
                "size": f"~{size_gb}GB",
                "type": model_type,
                "description": description,
                "downloads": getattr(model, "downloads", 0),
                "created": (
                    getattr(model, "createdAt", "").split("T")[0]
                    if hasattr(model, "createdAt")
                    else ""
                ),
            }

        except Exception as e:
            print(f"⚠️ Error analyzing model {model.modelId}: {e}")
            return None

    def _classify_model(self, name: str) -> tuple[str, str]:
        """Classify model size and type based on name"""
        name_lower = name.lower()

        # Size estimation for quantized models
        if "0.5b" in name_lower or "500m" in name_lower:
            return "0.3", "ultra-fast"
        elif "1b" in name_lower or "1.5b" in name_lower:
            return "1", "fast"
        elif "3b" in name_lower:
            return "2", "balanced"
        elif "7b" in name_lower:
            if "4bit" in name_lower or "4-bit" in name_lower:
                return "4", "advanced"
            else:
                return "14", "advanced"
        elif "14b" in name_lower:
            if "4bit" in name_lower or "4-bit" in name_lower:
                return "8", "professional"
            else:
                return "28", "professional"
        elif "32b" in name_lower:
            if "4bit" in name_lower or "4-bit" in name_lower:
                return "18", "expert"
            else:
                return "64", "expert"
        elif "70b" in name_lower or "72b" in name_lower:
            if "4bit" in name_lower or "4-bit" in name_lower:
                return "40", "flagship"
            else:
                return "140", "flagship"
        elif "8x22b" in name_lower:
            if "4bit" in name_lower or "4-bit" in name_lower:
                return "45", "specialist"
            else:
                return "176", "specialist"
        else:
            return "5", "standard"

    def _generate_description(self, name: str, model_type: str) -> str:
        """Generate appropriate description based on model name and type"""
        name_lower = name.lower()

        descriptions = {
            "ultra-fast": "Ultra-lightweight model for basic tasks",
            "fast": "Fast and efficient for everyday use",
            "balanced": "Good balance of speed and quality",
            "advanced": "High-performance model with excellent quality",
            "professional": "Professional-grade model with superior reasoning",
            "expert": "Expert-level model for complex tasks and research",
            "flagship": "Flagship model with exceptional performance",
            "specialist": "Mixture-of-experts model optimized for specialized tasks",
        }

        # Add specific model family info
        if "qwen" in name_lower:
            family = "Qwen family - "
        elif "llama" in name_lower:
            family = "Llama family - "
        elif "mistral" in name_lower or "mixtral" in name_lower:
            family = "Mistral family - "
        elif "gemma" in name_lower:
            family = "Gemma family - "
        else:
            family = ""

        return family + descriptions.get(model_type, "High-quality language model")

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        """Fallback curated model list if discovery fails"""
        return [
            {
                "name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "status": "available",
                "size": "~0.3GB",
                "type": "ultra-fast",
                "description": "Qwen family - Ultra-lightweight model for basic tasks",
                "downloads": 5000,
            },
            {
                "name": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "status": "available",
                "size": "~1GB",
                "type": "balanced",
                "description": "Qwen family - Good balance of speed and quality",
                "downloads": 8000,
            },
            {
                "name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                "status": "available",
                "size": "~4GB",
                "type": "advanced",
                "description": "Qwen family - High-performance model with excellent quality",
                "downloads": 15000,
            },
            {
                "name": "mlx-community/Qwen2.5-14B-Instruct-4bit",
                "status": "available",
                "size": "~8GB",
                "type": "professional",
                "description": "Qwen family - Professional-grade model with superior reasoning",
                "downloads": 12000,
            },
            {
                "name": "mlx-community/Qwen2.5-32B-Instruct-4bit",
                "status": "available",
                "size": "~18GB",
                "type": "expert",
                "description": "Qwen family - Expert-level model for complex tasks and research",
                "downloads": 10000,
            },
            {
                "name": "mlx-community/Qwen2.5-72B-Instruct-4bit",
                "status": "available",
                "size": "~40GB",
                "type": "flagship",
                "description": "Qwen family - Flagship model with exceptional performance",
                "downloads": 8000,
            },
        ]

    async def download_model(self, model_name: str):
        """Download MLX model"""
        try:
            print(f"📥 Downloading model: {model_name}")

            # Ensure models directory exists
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Use python -m mlx_lm.convert to download and convert the model
            process = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "mlx_lm.convert",
                "--hf-path",
                model_name,
                "--mlx-path",
                f"models/{model_name.replace('/', '_')}",
                "--quantize",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
                print(
                    f"❌ Authentication Error: Model {model_name} requires Hugging Face authentication"
                )
                print(
                    f"💡 Solution: Set up your HF token in the Settings tab, or choose a different model"
                )
                print(f"📋 Available models that don't require authentication:")
                for model in [
                    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                    "mlx-community/stablelm-2-zephyr-1_6b-4bit",
                ]:
                    print(f"   • {model}")
                raise Exception(
                    f"Authentication required for {model_name}. Please set up HF token or use a different model."
                )
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
                "python",
                "-c",
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
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print(
                    f"✅ Model {model_name} downloaded successfully via fallback method!"
                )
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
            # Don't try to load model in sync context - this causes event loop conflicts
            print(
                f"❌ Model not loaded. Please ensure model is loaded before calling sync generation."
            )
            raise Exception(
                "Model not loaded. Use async generate_response() method or ensure model is pre-loaded."
            )

        max_tokens = max_tokens or settings.max_tokens

        try:
            start_time = time.time()

            # CRITICAL: Direct MLX generation on main thread for maximum Metal acceleration
            response = generate(
                model=self.current_model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Minimal post-processing for speed
            response = self._clean_response(response)

            # Update performance stats
            estimated_tokens = len(response.split()) * 1.3
            tokens_per_second = (
                estimated_tokens / response_time if response_time > 0 else 0
            )

            self.performance_stats["response_times"].append(response_time)
            self.performance_stats["tokens_per_second"] = tokens_per_second
            self.performance_stats["total_tokens"] += estimated_tokens

            # Keep only last 10 response times
            if len(self.performance_stats["response_times"]) > 10:
                self.performance_stats["response_times"] = self.performance_stats[
                    "response_times"
                ][-10:]

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
                verbose=False,
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Post-process response to remove repetitive patterns
            response = self._clean_response(response)  # RE-ENABLED

            # Calculate tokens per second (approximate)
            estimated_tokens = len(response.split()) * 1.3  # Rough estimate
            tokens_per_second = (
                estimated_tokens / response_time if response_time > 0 else 0
            )

            # Update performance stats
            self.performance_stats["response_times"].append(response_time)
            self.performance_stats["tokens_per_second"] = tokens_per_second
            self.performance_stats["total_tokens"] += estimated_tokens

            # Keep only last 10 response times
            if len(self.performance_stats["response_times"]) > 10:
                self.performance_stats["response_times"] = self.performance_stats[
                    "response_times"
                ][-10:]

            return response

        except Exception as e:
            print(f"❌ Error in MLX generation: {e}")
            return "I apologize, but I encountered an error generating the response."

    async def stream_response(
        self, prompt: str, max_tokens: int = None
    ) -> AsyncGenerator[str, None]:
        """Stream response generation with Metal acceleration"""
        # Note: MLX doesn't have native streaming yet, so we simulate it
        try:
            # Use sync method directly for Metal GPU performance and to avoid async overhead
            response = self.generate_response_sync(prompt, max_tokens)

            # Split response into chunks and yield them
            words = response.split()
            chunk_size = 4  # Slightly larger chunks for better performance

            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
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
            "percentage": process.memory_percent(),
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
            "total_tokens": self.performance_stats.get("total_tokens", 0),
        }

    def _clean_response(self, response: str) -> str:
        """Clean response by removing repetitive patterns and unwanted content"""
        import re

        # Remove repetitive URLs and patterns
        response = re.sub(r"Read more: https?://[^\s]*", "", response)
        response = re.sub(r"#\w+\s*", "", response)  # Remove hashtags
        response = re.sub(r"\s*View all conversations\s*", "", response)

        # Stop at conversation continuation patterns
        stop_patterns = [
            r"\nUser:",
            r"\nHuman:",
            r"\nAssistant:",
            r"\n\[INST\]",
            r"\nQuestion:",
            r"\nAnswer:",
            r"\nQ:",
            r"\nA:",
            r" Q:",  # Q&A continuation with space
            r"Q: ",  # Q&A continuation at start
        ]

        for pattern in stop_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                response = response[: match.start()].strip()
                break

        # Detect numbered repetition patterns like "1. was that the joke 2. was that the joke"
        numbered_pattern = re.search(
            r"(\d+\.\s+[^0-9]*?)\s*\d+\.\s+.*?\1", response, re.IGNORECASE | re.DOTALL
        )
        if numbered_pattern:
            # Keep only up to the first repetition
            response = response[
                : numbered_pattern.start() + len(numbered_pattern.group(1))
            ].strip()

        # Detect and stop rambling/repetition
        sentences = response.split(". ")
        if len(sentences) > 2:
            # Check for repetitive content
            clean_sentences = []
            seen_content = set()

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check for repetitive phrases
                sentence_key = " ".join(sentence.lower().split()[:8])  # First 8 words

                if sentence_key not in seen_content:
                    clean_sentences.append(sentence)
                    seen_content.add(sentence_key)
                else:
                    # Found repetition, stop here
                    break

            response = ". ".join(clean_sentences)
            if response and not response.endswith("."):
                response += "."

        # Remove incomplete sentences at the end
        if response:
            # If response ends with an incomplete thought (common patterns)
            incomplete_patterns = [
                r"\s+If you are.*$",
                r"\s+However, if.*$",
                r"\s+But if.*$",
                r"\s+The.*\s+(is|are|was|were)\s*$",
                r"\s+This.*\s+(is|are|was|were)\s*$",
                r"\s+\d+\.\s*$",  # Trailing numbered list items
                r"\s+was that.*$",  # Trailing "was that" phrases
            ]

            for pattern in incomplete_patterns:
                response = re.sub(pattern, "", response, flags=re.IGNORECASE)

        # Final cleanup - ensure response ends properly
        response = response.strip()
        if response and not response.endswith((".", "!", "?", '"')):
            # Find the last complete sentence
            last_period = response.rfind(".")
            last_exclamation = response.rfind("!")
            last_question = response.rfind("?")

            last_punct = max(last_period, last_exclamation, last_question)
            if last_punct > 0:
                response = response[: last_punct + 1].strip()

        return response if response else "I'd be happy to help with that!"
