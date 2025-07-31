"""
Chat Engine - Retrieval Augmented Generation with Hybrid Reasoning
High-performance document-based conversations with DSPy-optimized reasoning
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from config import settings

from .reasoning.hybrid_manager import HybridStackManager


class ChatEngine:
    _instance = None
    _initialized = False

    def __new__(cls, mlx_manager, vector_store):
        if cls._instance is None:
            cls._instance = super(ChatEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, mlx_manager, vector_store):
        # Only initialize once
        if ChatEngine._initialized:
            return

        self.mlx_manager = mlx_manager
        self.vector_store = vector_store
        self.conversation_history = []
        self.response_times = []
        self.conversation_id = 0

        # Initialize hybrid reasoning system
        self.hybrid_manager = HybridStackManager(mlx_manager)

        # Chat history storage
        self.chat_history_dir = Path("data/chat_history")
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
        self.current_chat_id = None
        self.current_chat_title = None
        self.current_chat_messages = []

        # Mark as initialized
        ChatEngine._initialized = True

    async def initialize(self):
        """Initialize the chat engine and hybrid reasoning system"""
        print("💬 Initializing Enhanced Chat Engine with Hybrid Reasoning...")
        await self.hybrid_manager.initialize()
        print("✅ Enhanced Chat Engine initialized!")

    def start_new_chat(self, title: str = None) -> str:
        """Start a new chat session"""
        # Save current chat if it exists
        if self.current_chat_id and self.current_chat_messages:
            self._save_current_chat()

        # Create new chat
        self.current_chat_id = str(uuid.uuid4())
        self.current_chat_title = (
            title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        self.current_chat_messages = []

        return self.current_chat_id

    def _save_current_chat(self):
        """Save the current chat to disk"""
        if not self.current_chat_id or not self.current_chat_messages:
            return

        chat_data = {
            "id": self.current_chat_id,
            "title": self.current_chat_title,
            "created_at": (
                self.current_chat_messages[0]["timestamp"]
                if self.current_chat_messages
                else datetime.now().isoformat()
            ),
            "updated_at": datetime.now().isoformat(),
            "messages": self.current_chat_messages,
            "message_count": len(self.current_chat_messages),
            "metadata": {
                "model_used": self.mlx_manager.current_model_name,
                "total_tokens": sum(
                    len(msg.get("message", "").split())
                    for msg in self.current_chat_messages
                ),
                "reasoning_stack": (
                    self.hybrid_manager.get_current_stack()
                    if hasattr(self.hybrid_manager, "get_current_stack")
                    else "auto"
                ),
            },
        }

        chat_file = self.chat_history_dir / f"{self.current_chat_id}.json"
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved chat: {self.current_chat_title}")

    async def get_chat_list(self) -> List[Dict[str, Any]]:
        """Get list of all saved chats"""
        chats = []

        for chat_file in self.chat_history_dir.glob("*.json"):
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)

                # Create summary
                chat_summary = {
                    "id": chat_data["id"],
                    "title": chat_data["title"],
                    "created_at": chat_data["created_at"],
                    "updated_at": chat_data["updated_at"],
                    "message_count": chat_data["message_count"],
                    "preview": self._get_chat_preview(chat_data["messages"]),
                    "metadata": chat_data.get("metadata", {}),
                }

                chats.append(chat_summary)

            except Exception as e:
                print(f"❌ Error loading chat {chat_file}: {e}")
                continue

        # Sort by updated_at (most recent first)
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats

    def _get_chat_preview(self, messages: List[Dict]) -> str:
        """Get a preview of the chat (first user message)"""
        for msg in messages:
            if msg.get("type") == "user" and msg.get("message"):
                preview = msg["message"][:100]
                return preview + "..." if len(msg["message"]) > 100 else preview
        return "No messages"

    async def load_chat(self, chat_id: str) -> Dict[str, Any]:
        """Load a specific chat by ID"""
        chat_file = self.chat_history_dir / f"{chat_id}.json"

        if not chat_file.exists():
            raise FileNotFoundError(f"Chat {chat_id} not found")

        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                chat_data = json.load(f)

            # Set as current chat
            self.current_chat_id = chat_data["id"]
            self.current_chat_title = chat_data["title"]
            self.current_chat_messages = chat_data["messages"]

            return chat_data

        except Exception as e:
            raise Exception(f"Error loading chat: {e}")

    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a specific chat"""
        chat_file = self.chat_history_dir / f"{chat_id}.json"

        if not chat_file.exists():
            return False

        try:
            chat_file.unlink()

            # If this was the current chat, clear it
            if self.current_chat_id == chat_id:
                self.current_chat_id = None
                self.current_chat_title = None
                self.current_chat_messages = []

            print(f"🗑️ Deleted chat: {chat_id}")
            return True

        except Exception as e:
            print(f"❌ Error deleting chat {chat_id}: {e}")
            return False

    async def export_chat_to_markdown(self, chat_id: str) -> str:
        """Export a chat to markdown format with metadata"""
        chat_file = self.chat_history_dir / f"{chat_id}.json"

        if not chat_file.exists():
            raise FileNotFoundError(f"Chat {chat_id} not found")

        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                chat_data = json.load(f)

            # Create markdown content
            markdown_content = self._generate_markdown_content(chat_data)

            return markdown_content

        except Exception as e:
            raise Exception(f"Error exporting chat: {e}")

    def _generate_markdown_content(self, chat_data: Dict[str, Any]) -> str:
        """Generate markdown content from chat data"""
        md_lines = []

        # Header with metadata
        md_lines.append(f"# {chat_data['title']}")
        md_lines.append("")
        md_lines.append("## Chat Metadata")
        md_lines.append("")
        md_lines.append(f"- **Chat ID:** {chat_data['id']}")
        md_lines.append(f"- **Created:** {chat_data['created_at']}")
        md_lines.append(f"- **Last Updated:** {chat_data['updated_at']}")
        md_lines.append(f"- **Total Messages:** {chat_data['message_count']}")

        metadata = chat_data.get("metadata", {})
        if metadata:
            md_lines.append(
                f"- **Model Used:** {metadata.get('model_used', 'Unknown')}"
            )
            md_lines.append(
                f"- **Total Tokens:** {metadata.get('total_tokens', 'Unknown')}"
            )
            md_lines.append(
                f"- **Reasoning Stack:** {metadata.get('reasoning_stack', 'Unknown')}"
            )

        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## Conversation")
        md_lines.append("")

        # Messages
        for msg in chat_data["messages"]:
            if msg.get("type") == "user":
                md_lines.append(f"### 👤 User")
                md_lines.append("")
                md_lines.append(msg.get("message", ""))
                md_lines.append("")
                md_lines.append(f"*Sent at: {msg.get('timestamp', 'Unknown')}*")
                md_lines.append("")

            elif msg.get("type") == "assistant":
                md_lines.append(f"### 🤖 Assistant")
                md_lines.append("")
                md_lines.append(msg.get("message", ""))
                md_lines.append("")

                # Add response metadata if available
                if msg.get("response_time"):
                    md_lines.append(f"*Response time: {msg['response_time']:.2f}s*")

                if msg.get("sources"):
                    md_lines.append(f"*Sources: {', '.join(msg['sources'])}*")

                if msg.get("reasoning"):
                    reasoning = msg["reasoning"]
                    md_lines.append(
                        f"*Reasoning: {reasoning.get('strategy', 'Unknown')} (confidence: {reasoning.get('confidence', 0):.2f})*"
                    )

                md_lines.append("")

        # Footer
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(
            f"*Exported from LLMind on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        )

        return "\n".join(md_lines)

    async def stream_response(
        self,
        user_message: str,
        conversation_context: List[Dict] = None,
        max_tokens_override: int = None,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with hybrid reasoning optimization"""
        try:
            start_time = time.time()
            print(f"💬 Processing message: {user_message[:100]}...")

            # Start new chat if none exists
            if not self.current_chat_id:
                self.start_new_chat()

            # Step 1: Get context from vector search (if needed by reasoning strategy)
            relevant_docs = []
            context = ""

            if not self._is_definitely_simple_query(user_message):
                relevant_docs = await self.vector_store.search(
                    user_message, k=settings.top_k_results
                )
                context = self._build_context(relevant_docs)

            # Step 2: Ensure we have proper conversation context
            enhanced_conversation_context = self._ensure_conversation_context(
                conversation_context
            )

            # Step 3: Process with hybrid reasoning
            hybrid_result = await self.hybrid_manager.process_query(
                user_message, context, enhanced_conversation_context
            )
            optimized_prompt = hybrid_result.get("optimized_prompt", user_message)
            max_tokens = max_tokens_override or hybrid_result.get(
                "max_tokens", settings.max_tokens
            )

            # Log token override for stories/long content
            if max_tokens_override:
                print(
                    f"📝 Using token override: {max_tokens_override} tokens for enhanced response"
                )

            reasoning_info = {
                "strategy": hybrid_result.get("strategy", "unknown"),
                "confidence": hybrid_result.get("confidence", 0.0),
                "processing_time": hybrid_result.get("total_processing_time", 0.0),
                "bypass_reasoning": hybrid_result.get("bypass_reasoning", False),
            }

            print(
                f"🧠 Strategy: {reasoning_info['strategy']} | Confidence: {reasoning_info['confidence']:.3f}"
            )
            print(f"🔍 Found {len(relevant_docs)} relevant documents")
            print(
                f"📝 Generated optimized prompt with {len(context)} characters of context"
            )

            # Step 4: Stream response using optimized MLX
            full_response = ""
            async for chunk in self.mlx_manager.stream_response(
                optimized_prompt, max_tokens
            ):
                full_response += chunk
                yield chunk

            # Step 5: Track performance
            end_time = time.time()
            response_time = end_time - start_time
            self.response_times.append(response_time)

            # Keep only last 20 response times
            if len(self.response_times) > 20:
                self.response_times = self.response_times[-20:]

            # Step 6: Update conversation history and current chat
            self._update_conversation_history(
                user_message,
                full_response,
                relevant_docs,
                response_time,
                start_time,
                reasoning_info,
                hybrid_result,
            )

            # Add to current chat messages
            self._add_to_current_chat(
                user_message,
                full_response,
                reasoning_info,
                relevant_docs,
                start_time,
                response_time,
            )

            # Auto-save current chat
            self._save_current_chat()

            print(
                f"✅ Response generated in {response_time:.2f}s using {reasoning_info['strategy']}"
            )

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def _add_to_current_chat(
        self,
        user_message: str,
        assistant_response: str,
        reasoning_info: Dict,
        relevant_docs: List[Dict],
        start_time: float,
        response_time: float,
    ):
        """Add messages to current chat session"""
        # Add user message
        user_msg = {
            "id": len(self.current_chat_messages),
            "type": "user",
            "message": user_message,
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "relevant_docs": len(relevant_docs),
            "context_chars": len(self._build_context(relevant_docs)),
        }
        self.current_chat_messages.append(user_msg)

        # Add assistant response
        assistant_msg = {
            "id": len(self.current_chat_messages),
            "type": "assistant",
            "message": assistant_response,
            "timestamp": datetime.fromtimestamp(start_time + response_time).isoformat(),
            "response_time": response_time,
            "sources": [
                doc.get("metadata", {}).get("filename", "Unknown")
                for doc in relevant_docs[:3]
            ],
            "reasoning": reasoning_info,
        }
        self.current_chat_messages.append(assistant_msg)

        # Update chat title if this is the first exchange
        if len(self.current_chat_messages) == 2 and self.current_chat_title.startswith(
            "Chat "
        ):
            # Use first few words of user message as title
            title_words = user_message.split()[:6]
            self.current_chat_title = " ".join(title_words)
            if len(user_message.split()) > 6:
                self.current_chat_title += "..."

    async def generate_response(
        self, user_message: str, conversation_context: List[Dict] = None
    ) -> Dict[str, Any]:
        """Generate complete response with hybrid reasoning and metadata"""
        try:
            start_time = time.time()

            # Step 1: Get context from vector search (if needed)
            relevant_docs = []
            context = ""

            if not self._is_definitely_simple_query(user_message):
                relevant_docs = await self.vector_store.search(
                    user_message, k=settings.top_k_results
                )
                context = self._build_context(relevant_docs)

            # Step 2: Ensure we have proper conversation context
            enhanced_conversation_context = self._ensure_conversation_context(
                conversation_context
            )

            # Step 3: Process with hybrid reasoning system
            hybrid_result = await self.hybrid_manager.process_query(
                user_message, context, enhanced_conversation_context
            )

            # Step 3: Extract optimized prompt and parameters
            optimized_prompt = hybrid_result.get("optimized_prompt", user_message)
            max_tokens = hybrid_result.get("max_tokens", settings.max_tokens)

            # Step 4: Generate response using MLX
            response = await self.mlx_manager.generate_response(
                optimized_prompt, max_tokens
            )

            # Step 5: Track performance
            end_time = time.time()
            response_time = end_time - start_time
            self.response_times.append(response_time)

            # Step 6: Prepare enhanced response with hybrid reasoning metadata
            result = {
                "response": response,
                "sources": [
                    {
                        "filename": doc.get("metadata", {}).get("filename", "Unknown"),
                        "similarity": doc.get("similarity", 0),
                        "chunk_index": doc.get("metadata", {}).get("chunk_index", 0),
                        "text_preview": (
                            doc.get("text", "")[:200] + "..."
                            if len(doc.get("text", "")) > 200
                            else doc.get("text", "")
                        ),
                    }
                    for doc in relevant_docs
                ],
                "metadata": {
                    "response_time": response_time,
                    "relevant_documents": len(relevant_docs),
                    "context_length": len(context),
                    "prompt_length": len(optimized_prompt),
                    "model": self.mlx_manager.current_model_name,
                    "timestamp": datetime.fromtimestamp(end_time).isoformat(),
                    # Enhanced metadata from hybrid reasoning
                    "reasoning": {
                        "strategy": hybrid_result.get("selected_strategy", "unknown"),
                        "confidence": hybrid_result.get("strategy_confidence", 0.5),
                        "stack": hybrid_result.get("current_stack", "auto"),
                        "reasoning_steps": hybrid_result.get("reasoning_steps", []),
                        "optimizations": hybrid_result.get("optimizations", []),
                        "dspy_analysis": hybrid_result.get("dspy_analysis", {}),
                        "processing_time": hybrid_result.get("processing_time", 0.0),
                    },
                },
            }

            # Step 7: Update conversation history
            self._update_conversation_history(
                user_message,
                response,
                relevant_docs,
                response_time,
                start_time,
                result["metadata"]["reasoning"],
                hybrid_result,
            )

            return result

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "reasoning": {
                        "strategy": "error_fallback",
                        "confidence": 0.0,
                        "stack": "auto",
                    },
                },
            }

    async def stream_response_direct(
        self, user_message: str, max_tokens: int = None
    ) -> AsyncGenerator[str, None]:
        """Direct MLX response without any reasoning overhead - for performance testing"""
        try:
            start_time = time.time()
            print(f"🚀 Direct MLX generation: {user_message[:50]}...")

            # Direct MLX generation with minimal overhead
            max_tokens = max_tokens or settings.max_tokens

            # Minimal prompt formatting
            simple_prompt = f"User: {user_message}\nAssistant: "

            # Stream response directly from MLX
            full_response = ""
            chunk_count = 0

            async for chunk in self.mlx_manager.stream_response(
                simple_prompt, max_tokens
            ):
                full_response += chunk
                chunk_count += 1
                yield chunk

            # Track performance
            end_time = time.time()
            response_time = end_time - start_time
            tokens_per_second = chunk_count / response_time if response_time > 0 else 0

            print(
                f"✅ Direct MLX: {response_time:.2f}s, {chunk_count} chunks, {tokens_per_second:.1f} chunks/s"
            )

        except Exception as e:
            print(f"❌ Error in direct MLX generation: {e}")
            yield f"Error: {str(e)}"

    def _is_definitely_simple_query(self, user_message: str) -> bool:
        """Quick check to identify obviously simple queries that don't need vector search"""
        # This is a fast preliminary check - the hybrid manager will do more sophisticated analysis
        query_lower = user_message.lower().strip()
        words = user_message.split()

        # Very short greetings and simple responses
        if len(words) <= 3 and any(
            word in query_lower
            for word in ["hi", "hello", "thanks", "thank you", "bye"]
        ):
            return True

        # Simple math with only numbers and operators
        if len(words) <= 5 and any(
            op in query_lower for op in ["+", "-", "*", "/", "="]
        ):
            return True

        return False

    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Build context string from relevant documents"""
        if not relevant_docs:
            return ""

        context_parts = []
        total_chars = 0
        max_context_chars = 3000  # Reasonable limit for normal conversations

        for i, doc in enumerate(relevant_docs):
            # Include source information
            source = doc.get("metadata", {}).get("filename", "Unknown")
            text = doc["text"]
            score = doc.get("score", 0)
            chunk_idx = doc.get("metadata", {}).get("chunk_index", 0)

            # Format context with metadata
            context_part = f"[Source {i+1}: {source} (chunk {chunk_idx}, relevance: {score:.3f})]\n{text}\n"

            # Check if adding this would exceed our limit
            if total_chars + len(context_part) > max_context_chars:
                context_parts.append(
                    "[Additional relevant sources available but truncated for brevity...]"
                )
                break

            context_parts.append(context_part)
            total_chars += len(context_part)

        return "\n---\n".join(context_parts)

    def _update_conversation_history(
        self,
        user_message: str,
        response: str,
        relevant_docs: List[Dict],
        response_time: float,
        start_time: float,
        reasoning_info: Dict[str, Any],
        hybrid_result: Dict[str, Any],
    ):
        """Update conversation history with enhanced reasoning metadata"""
        self.conversation_history.append(
            {
                "id": len(self.conversation_history),
                "type": "user",
                "message": user_message,
                "timestamp": datetime.fromtimestamp(start_time).isoformat(),
                "relevant_docs": len(relevant_docs),
                "context_chars": len(self._build_context(relevant_docs)),
            }
        )

        self.conversation_history.append(
            {
                "id": len(self.conversation_history),
                "type": "assistant",
                "message": response,
                "timestamp": datetime.fromtimestamp(
                    start_time + response_time
                ).isoformat(),
                "response_time": response_time,
                "sources": [
                    doc.get("metadata", {}).get("filename", "Unknown")
                    for doc in relevant_docs[:3]
                ],
                # Enhanced reasoning metadata
                "reasoning": reasoning_info,
                "hybrid_analysis": {
                    "strategy_result": hybrid_result.get("strategy_result", {}),
                    "dspy_available": hybrid_result.get("dspy_analysis", {}).get(
                        "fallback", True
                    )
                    == False,
                },
            }
        )

        # Keep only last 50 messages
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def _format_conversation_context_for_prompt(
        self, conversation_context: List[Dict] = None
    ) -> str:
        """Format conversation context for use in prompts - more detailed than hybrid manager version"""
        if not conversation_context or len(conversation_context) == 0:
            # Try to use current chat messages if no context provided
            if self.current_chat_messages:
                conversation_context = self.current_chat_messages[
                    -6:
                ]  # Last 6 messages
            else:
                return ""

        # Build conversation history string with proper formatting
        conversation_lines = []
        for msg in conversation_context[-6:]:  # Last 6 messages (3 exchanges)
            if isinstance(msg, dict) and "type" in msg and "message" in msg:
                if msg["type"] == "user":
                    conversation_lines.append(f"Human: {msg['message']}")
                elif msg["type"] == "assistant":
                    # Limit assistant responses to avoid overwhelming context
                    response = msg["message"]
                    if len(response) > 300:
                        response = response[:300] + "..."
                    conversation_lines.append(f"Assistant: {response}")

        if conversation_lines:
            return "\n".join(conversation_lines) + "\n"

        return ""

    def _ensure_conversation_context(
        self, conversation_context: List[Dict] = None
    ) -> List[Dict]:
        """Ensure we have valid conversation context, using current chat if needed"""
        if conversation_context and len(conversation_context) > 0:
            return conversation_context

        # Fallback to current chat messages if available
        if self.current_chat_messages and len(self.current_chat_messages) > 0:
            return self.current_chat_messages[-6:]  # Last 6 messages

        # Ultimate fallback to conversation history
        if self.conversation_history and len(self.conversation_history) > 0:
            return self.conversation_history[-6:]  # Last 6 messages

        return []

    # Enhanced management methods for hybrid reasoning

    async def switch_reasoning_stack(self, stack_name: str) -> bool:
        """Switch to a different hybrid reasoning stack"""
        return await self.hybrid_manager.switch_stack(stack_name)

    def get_available_reasoning_stacks(self) -> Dict[str, Dict[str, Any]]:
        """Get all available hybrid reasoning stacks"""
        return self.hybrid_manager.get_available_stacks()

    def get_current_reasoning_stack(self) -> str:
        """Get current hybrid reasoning stack"""
        return self.hybrid_manager.get_current_stack()

    def get_reasoning_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive reasoning performance statistics"""
        return self.hybrid_manager.get_performance_stats()

    async def optimize_reasoning_for_examples(
        self, examples: List[Dict[str, Any]]
    ) -> bool:
        """Optimize reasoning system based on example queries"""
        return await self.hybrid_manager.optimize_stack_for_examples(examples)

    async def create_custom_reasoning_stack(
        self, name: str, config: Dict[str, Any]
    ) -> bool:
        """Create a custom hybrid reasoning stack"""
        return await self.hybrid_manager.create_custom_stack(name, config)

    async def update_reasoning_configuration(self, config: Dict[str, Any]) -> bool:
        """Update hybrid reasoning configuration"""
        return await self.hybrid_manager.update_configuration(config)

    # Existing methods remain the same

    def get_avg_response_time(self) -> float:
        """Get average response time"""
        return (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0.0
        )

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced chat engine statistics"""
        basic_stats = {
            "total_conversations": len(self.conversation_history) // 2,
            "avg_response_time": self.get_avg_response_time(),
            "total_response_times": len(self.response_times),
            "current_model": self.mlx_manager.current_model_name,
        }

        # Add reasoning performance stats
        reasoning_stats = self.get_reasoning_performance_stats()

        return {**basic_stats, "reasoning_performance": reasoning_stats}
