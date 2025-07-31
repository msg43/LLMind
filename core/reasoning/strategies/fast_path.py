"""
Fast Path Strategy - Ultra-high speed processing for simple queries
Optimized for maximum performance with minimal overhead
"""

import time
from typing import Any, Dict, List

from config import settings

from .base_strategy import BaseReasoningStrategy


class FastPathStrategy(BaseReasoningStrategy):
    """Ultra-fast processing for simple, direct queries"""

    def __init__(self):
        super().__init__(
            name="FastPath",
            description="Ultra-fast processing for simple queries with minimal overhead",
        )

    async def should_handle(
        self, query: str, context: str = "", analysis: Dict[str, Any] = None
    ) -> float:
        """Aggressively detect simple queries for maximum performance"""
        # Use confidence_threshold from analysis if provided
        confidence_threshold = (
            analysis.get("confidence_threshold", 0.7) if analysis else 0.7
        )

        query_lower = query.lower().strip()
        query_words = query_lower.split()

        # VERY HIGH confidence for obvious simple cases
        if len(query_lower) < 10:  # Very short queries
            return 0.95

        # Greetings and social - MAX confidence
        greeting_patterns = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what's up",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
        ]
        if any(pattern in query_lower for pattern in greeting_patterns):
            return 0.98

        # Simple math - MAX confidence
        math_patterns = [
            "what is",
            "calculate",
            "plus",
            "minus",
            "times",
            "divided",
            "multiply",
            "+",
            "-",
            "*",
            "/",
            "=",
        ]
        if (
            any(pattern in query_lower for pattern in math_patterns)
            and len(query_words) < 15
        ):
            return 0.98

        # Short factual questions - HIGH confidence
        if (
            query_lower.startswith(("what", "who", "when", "where", "how"))
            and len(query_words) < 10
        ):
            return 0.90

        # Simple definitions - HIGH confidence
        definition_patterns = ["what is", "define", "meaning of", "explain"]
        if (
            any(pattern in query_lower for pattern in definition_patterns)
            and len(query_words) < 8
        ):
            return 0.90

        # Yes/no questions - HIGH confidence
        if (
            query_lower.startswith(
                (
                    "is",
                    "are",
                    "can",
                    "could",
                    "will",
                    "would",
                    "should",
                    "do",
                    "does",
                    "did",
                )
            )
            and len(query_words) < 12
        ):
            return 0.85

        # Single word or very short queries - HIGH confidence
        if len(query_words) <= 3:
            return 0.85

        # Avoid complex patterns entirely
        complex_indicators = [
            "analyze",
            "compare",
            "contrast",
            "discuss",
            "explain in detail",
            "comprehensive",
            "thorough",
            "elaborate",
            "pros and cons",
        ]
        if any(indicator in query_lower for indicator in complex_indicators):
            return 0.1

        # Multi-sentence queries - avoid
        if query.count(".") > 1 or query.count("?") > 1:
            return 0.1

        # Default for other short, simple queries
        if len(query_words) < 15:
            return 0.70

        return 0.2  # Low confidence for anything else

    async def process_query(
        self,
        query: str,
        context: str = "",
        analysis: Dict[str, Any] = None,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """Ultra-fast processing with minimal overhead"""
        start_time = time.time()

        # PROPER Llama-3.1-Instruct formatting with system prompts
        query_lower = query.lower().strip()

        # Base system prompt for AI identity
        base_system = "You are a helpful AI assistant. Be concise and direct."

        # For greetings, use friendly but brief system prompt
        if any(
            greeting in query_lower
            for greeting in ["hello", "hi", "hey", "how are you"]
        ):
            system_prompt = (
                "You are a friendly AI assistant. Respond warmly but briefly."
            )
            if conversation_context:
                # Include conversation context for more natural greetings
                optimized_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{conversation_context}\nHuman: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                optimized_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            max_tokens = 50  # Very short responses

        # For simple math, use calculation-focused prompt
        elif any(
            math in query_lower for math in ["what is", "calculate", "+", "-", "*", "/"]
        ):
            system_prompt = "You are a helpful AI assistant that provides accurate calculations. Give direct numerical answers."
            optimized_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            max_tokens = 30  # Very short for math

        # For jokes and creative content
        elif any(word in query_lower for word in ["joke", "funny", "laugh", "humor"]):
            system_prompt = "You are a helpful AI assistant that can tell appropriate jokes and be humorous. Keep responses family-friendly."
            if conversation_context:
                # Include conversation context for contextual humor
                optimized_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{conversation_context}\nHuman: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                optimized_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            max_tokens = 150

        # For simple questions that may benefit from conversation context
        elif conversation_context and any(
            word in query_lower
            for word in [
                "yes",
                "no",
                "sure",
                "ok",
                "another",
                "more",
                "that",
                "it",
                "this",
            ]
        ):
            # These are likely responses to previous messages, so include conversation context
            optimized_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{base_system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{conversation_context}\nHuman: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            max_tokens = min(
                settings.max_tokens // 3, 200
            )  # Slightly longer for context-dependent responses

        # For simple questions, clear AI identity
        else:
            optimized_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{base_system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            max_tokens = min(
                settings.max_tokens // 4, 150
            )  # Quarter of normal max tokens

        processing_time = time.time() - start_time

        return {
            "handled": True,
            "optimized_prompt": optimized_prompt,
            "max_tokens": max_tokens,
            "strategy": "FastPath",
            "confidence": 0.9,
            "reasoning": "Simple query detected - using ultra-fast path with proper Llama-3.1 formatting",
            "processing_time": processing_time,
            "context_used": False,  # Never use document context for speed
            "conversation_context_used": bool(conversation_context),
            "token_efficiency": "maximum",  # Indicate this is optimized for speed
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration for FastPath strategy"""
        return {
            "name": "FastPath",
            "description": "Ultra-fast processing for simple queries",
            "max_word_threshold": 15,
            "confidence_thresholds": {
                "greetings": 0.98,
                "math": 0.98,
                "simple_questions": 0.90,
                "definitions": 0.90,
                "yes_no": 0.85,
            },
            "max_tokens": {"greetings": 50, "math": 30, "default": 150},
            "bypass_vector_search": True,
            "minimal_prompt_engineering": True,
        }

    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update strategy configuration"""
        try:
            # FastPath is optimized for speed and doesn't support dynamic configuration
            # to avoid performance overhead. Return True to indicate "success" but don't change anything.
            print(
                "ℹ️  FastPath configuration is optimized and cannot be changed for performance reasons"
            )
            return True
        except Exception as e:
            print(f"❌ Error updating FastPath config: {e}")
            return False
