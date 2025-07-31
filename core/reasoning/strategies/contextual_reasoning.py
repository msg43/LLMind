"""
Contextual Reasoning Strategy - Leveraging document context for informed responses
Optimizes context usage and ensures context-aware reasoning
"""

from typing import Any, Dict, List

from .base_strategy import BaseReasoningStrategy


class ContextualReasoningStrategy(BaseReasoningStrategy):
    """Strategy for context-aware reasoning with document integration"""

    def __init__(self):
        super().__init__(
            name="ContextualReasoning",
            description="Context-aware reasoning that leverages document knowledge effectively",
        )

        # Configuration
        self.config = {
            "min_context_length": 100,
            "max_context_chunks": 5,
            "confidence_threshold": 0.7,
            "context_relevance_threshold": 0.6,
            "enable_context_synthesis": True,
            "max_tokens": 500,
        }

        # Patterns that suggest context-dependent queries
        self.context_indicators = [
            "according to",
            "based on",
            "in the document",
            "the text says",
            "from what I read",
            "the article mentions",
            "the document states",
            "what does it say about",
            "how does the document explain",
            "summarize",
            "extract",
            "from the source",
            "referenced",
        ]

        # Question types that benefit from context
        self.context_question_types = [
            "summarization",
            "extraction",
            "analysis",
            "synthesis",
            "comparison",
            "evaluation",
            "interpretation",
        ]

    async def should_handle(
        self, query: str, context: str = "", analysis: Dict[str, Any] = None
    ) -> float:
        """Determine if query needs contextual reasoning"""

        # Context must be available and substantial
        if not context or len(context) < self.config["min_context_length"]:
            return 0.1

        query_lower = query.lower()

        # Check for explicit context references
        context_ref_score = sum(
            0.3 for indicator in self.context_indicators if indicator in query_lower
        )

        # Check if query asks about document content
        doc_query_patterns = [
            "what does",
            "how does",
            "summarize",
            "explain from",
            "based on",
        ]
        doc_query_score = sum(
            0.2 for pattern in doc_query_patterns if pattern in query_lower
        )

        # Check analysis for context dependency
        analysis_score = 0.0
        if analysis:
            reasoning_type = analysis.get("reasoning_type", "direct")
            if reasoning_type == "contextual":
                analysis_score = 0.8

            intent = analysis.get("intent", "factual")
            if intent in ["analytical", "factual"]:
                analysis_score += 0.2

        # Check for document-specific terms in query
        doc_terms = ["document", "text", "article", "source", "material", "content"]
        doc_term_score = sum(0.15 for term in doc_terms if term in query_lower)

        # Check query complexity vs context availability
        words = query.split()
        complexity_context_match = 0.0
        if len(words) > 10 and len(context) > 500:  # Complex query with rich context
            complexity_context_match = 0.3

        # Combined confidence score
        confidence = (
            context_ref_score * 0.3
            + doc_query_score * 0.25
            + analysis_score * 0.2
            + doc_term_score * 0.15
            + complexity_context_match * 0.1
        )

        return min(confidence, 1.0)

    async def process_query(
        self,
        query: str,
        context: str = "",
        analysis: Dict[str, Any] = None,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """Process query with context-aware reasoning"""

        # Analyze context relevance and structure
        context_analysis = await self._analyze_context(query, context)

        # Determine optimal context usage strategy
        context_strategy = self._determine_context_strategy(query, context_analysis)

        result = {
            "context_analysis": context_analysis,
            "context_strategy": context_strategy,
            "processing_type": "contextual_reasoning",
            "context_chunks_used": len(context_analysis.get("relevant_chunks", [])),
            "conversation_context_used": bool(conversation_context),
            "reasoning_steps": [
                "context_analysis",
                "relevance_assessment",
                "context_integration",
                "conversation_context_integration",
                "reasoning_with_context",
                "response_synthesis",
            ],
            "optimizations": [
                "context_prioritization",
                "relevance_filtering",
                "context_synthesis",
                "conversation_memory",
                "evidence_citation",
            ],
        }

        # Create context-aware prompt with conversation context
        result["optimized_prompt"] = self._create_contextual_prompt(
            query, context, context_analysis, context_strategy, conversation_context
        )

        # Set token limits based on context complexity
        context_complexity = context_analysis.get("complexity_score", 0.5)
        if context_complexity > 0.7:
            result["max_tokens"] = 600
        else:
            result["max_tokens"] = self.config["max_tokens"]

        return result

    async def _analyze_context(self, query: str, context: str) -> Dict[str, Any]:
        """Analyze context for relevance and structure"""

        # Split context into logical chunks
        chunks = self._split_context_into_chunks(context)

        # Assess relevance of each chunk to the query
        relevant_chunks = []
        query_terms = set(query.lower().split())

        for i, chunk in enumerate(chunks):
            chunk_terms = set(chunk.lower().split())
            relevance_score = (
                len(query_terms.intersection(chunk_terms)) / len(query_terms)
                if query_terms
                else 0
            )

            if relevance_score >= self.config["context_relevance_threshold"]:
                relevant_chunks.append(
                    {
                        "index": i,
                        "text": chunk,
                        "relevance_score": relevance_score,
                        "length": len(chunk),
                    }
                )

        # Sort by relevance and limit to max_context_chunks
        relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        relevant_chunks = relevant_chunks[: self.config["max_context_chunks"]]

        # Assess overall context characteristics
        context_info = {
            "total_chunks": len(chunks),
            "relevant_chunks": relevant_chunks,
            "total_length": len(context),
            "relevant_length": sum(chunk["length"] for chunk in relevant_chunks),
            "complexity_score": min(len(chunks) / 10, 1.0),
            "coverage_score": len(relevant_chunks) / len(chunks) if chunks else 0,
        }

        return context_info

    def _split_context_into_chunks(self, context: str) -> List[str]:
        """Split context into logical chunks for analysis"""

        # Try splitting by paragraphs first
        paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]

        if len(paragraphs) > 1:
            return paragraphs

        # Fall back to sentence splitting
        sentences = [s.strip() for s in context.split(".") if s.strip()]

        # Group sentences into chunks of 3-4
        chunks = []
        for i in range(0, len(sentences), 3):
            chunk = ". ".join(sentences[i : i + 3])
            if chunk:
                chunks.append(chunk + "." if not chunk.endswith(".") else chunk)

        return chunks if chunks else [context]

    def _determine_context_strategy(
        self, query: str, context_analysis: Dict[str, Any]
    ) -> str:
        """Determine optimal strategy for using context"""

        query_lower = query.lower()
        relevant_chunks = context_analysis.get("relevant_chunks", [])
        coverage_score = context_analysis.get("coverage_score", 0)

        # Summarization strategy
        if any(
            word in query_lower
            for word in ["summarize", "summary", "overview", "main points"]
        ):
            return "summarization"

        # Extraction strategy
        if any(
            word in query_lower for word in ["extract", "find", "locate", "identify"]
        ):
            return "extraction"

        # Synthesis strategy for complex queries with good context coverage
        if len(relevant_chunks) >= 3 and coverage_score > 0.6:
            return "synthesis"

        # Focused strategy for specific queries with limited relevant context
        if len(relevant_chunks) <= 2:
            return "focused"

        # Default comprehensive strategy
        return "comprehensive"

    def _create_contextual_prompt(
        self,
        query: str,
        context: str,
        context_analysis: Dict[str, Any],
        strategy: str,
        conversation_context: str = "",
    ) -> str:
        """Create a context-aware prompt based on strategy"""

        relevant_chunks = context_analysis.get("relevant_chunks", [])

        # Build focused context from most relevant chunks
        focused_context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        prompt_parts = []

        # Add conversation context if available
        if conversation_context:
            prompt_parts.extend(["Previous conversation:", conversation_context, ""])

        if strategy == "summarization":
            prompt_parts.extend(
                [
                    "Please provide a comprehensive summary based on the following information.",
                    "",
                    f"Question: {query}",
                    "",
                    "Source material:",
                    focused_context,
                    "",
                    "Summary:",
                ]
            )

        elif strategy == "extraction":
            prompt_parts.extend(
                [
                    "Extract specific information to answer this question from the provided context.",
                    "",
                    f"Question: {query}",
                    "",
                    "Context:",
                    focused_context,
                    "",
                    "Extracted answer:",
                ]
            )

        elif strategy == "synthesis":
            prompt_parts.extend(
                [
                    "Synthesize information from multiple sources to provide a comprehensive answer.",
                    "",
                    f"Question: {query}",
                    "",
                    "Available information:",
                    focused_context,
                    "",
                    "Synthesized response:",
                ]
            )

        elif strategy == "focused":
            prompt_parts.extend(
                [
                    "Answer this question using the most relevant information provided.",
                    "",
                    f"Question: {query}",
                    "",
                    "Relevant context:",
                    focused_context,
                    "",
                    "Answer:",
                ]
            )

        else:  # comprehensive
            prompt_parts.extend(
                [
                    "Use the provided context to give a thorough and well-informed answer.",
                    "",
                    f"Question: {query}",
                    "",
                    "Context information:",
                    focused_context,
                    "",
                    "Response:",
                ]
            )

        return "\n".join(prompt_parts)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()

    async def update_config(self, config: Dict[str, Any]) -> bool:
        """Update strategy configuration"""
        try:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
            return True
        except Exception as e:
            print(f"❌ Error updating ContextualReasoning config: {e}")
            return False
