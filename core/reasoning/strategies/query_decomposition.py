"""
Query Decomposition Strategy - Breaking complex queries into sub-questions
Handles multi-faceted questions by decomposing and addressing each part
"""

from typing import Any, Dict, List

from .base_strategy import BaseReasoningStrategy


class QueryDecompositionStrategy(BaseReasoningStrategy):
    """Strategy for decomposing complex multi-part queries"""

    def __init__(self):
        super().__init__(
            name="QueryDecomposition",
            description="Break complex multi-part questions into manageable sub-questions",
        )

        # Configuration
        self.config = {
            "max_subquestions": 5,
            "confidence_threshold": 0.6,
            "min_query_complexity": "moderate",
            "enable_subquestion_prioritization": True,
            "max_tokens_per_subquestion": 200,
        }

        # Patterns that indicate need for decomposition
        self.decomposition_indicators = [
            "and",
            "also",
            "additionally",
            "furthermore",
            "moreover",
            "both",
            "either",
            "multiple",
            "various",
            "several",
            "as well as",
            "in addition",
            "not only",
            "but also",
            "what about",
            "how about",
            "what if",
        ]

        # Question connectors that suggest multiple parts
        self.question_connectors = [
            ", and",
            "; and",
            ", or",
            "; or",
            ", but",
            "; but",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
        ]

    async def should_handle(
        self, query: str, context: str = "", analysis: Dict[str, Any] = None
    ) -> float:
        """Determine if query needs decomposition"""
        query_lower = query.lower()

        # Check for decomposition indicators
        indicator_count = sum(
            1 for indicator in self.decomposition_indicators if indicator in query_lower
        )
        indicator_score = min(indicator_count / 2, 1.0)

        # Check for multiple question words
        question_word_count = sum(
            1 for connector in self.question_connectors if connector in query_lower
        )
        question_score = min(question_word_count / 3, 1.0)

        # Check query length and structure
        sentences = query.split(".")
        clauses = query.split(",")
        structure_score = 0.0

        if len(sentences) > 2:
            structure_score += 0.3
        if len(clauses) > 3:
            structure_score += 0.2

        # Check for conjunctions that suggest multiple parts
        conjunctions = ["and", "or", "but", "however", "while", "whereas"]
        conjunction_count = sum(1 for conj in conjunctions if conj in query_lower)
        conjunction_score = min(conjunction_count / 2, 1.0)

        # Check analysis if provided
        complexity_score = 0.0
        if analysis:
            complexity = analysis.get("complexity", "simple")
            if complexity == "complex":
                complexity_score = 0.8
            elif complexity == "moderate":
                complexity_score = 0.5

            intent = analysis.get("intent", "factual")
            if intent == "analytical":
                complexity_score += 0.2

        # Combined confidence score
        confidence = (
            indicator_score * 0.25
            + question_score * 0.25
            + structure_score * 0.2
            + conjunction_score * 0.15
            + complexity_score * 0.15
        )

        return min(confidence, 1.0)

    async def process_query(
        self, query: str, context: str = "", analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process query by decomposing into sub-questions"""

        # Decompose the query
        subquestions = await self._decompose_query(query, context)

        # Prioritize sub-questions if enabled
        if self.config["enable_subquestion_prioritization"]:
            subquestions = await self._prioritize_subquestions(subquestions, context)

        result = {
            "subquestions": subquestions,
            "processing_type": "query_decomposition",
            "decomposition_strategy": "logical_split",
            "max_subquestions": len(subquestions),
            "reasoning_steps": [
                "query_parsing",
                "decomposition",
                "subquestion_prioritization",
                "sequential_processing",
                "answer_synthesis",
            ],
            "optimizations": [
                "parallel_subquestion_processing",
                "context_sharing_between_subquestions",
                "progressive_answer_building",
            ],
        }

        # Create decomposed prompt
        result["optimized_prompt"] = self._create_decomposed_prompt(
            query, context, subquestions
        )

        # Set token limits based on number of sub-questions
        total_tokens = len(subquestions) * self.config["max_tokens_per_subquestion"]
        result["max_tokens"] = min(total_tokens, 800)  # Cap at reasonable limit

        return result

    async def _decompose_query(self, query: str, context: str) -> List[Dict[str, Any]]:
        """Decompose query into logical sub-questions"""
        subquestions = []

        # Method 1: Split on conjunctions
        conjunctions = [" and ", " or ", " but ", ", and ", ", or ", ", but "]
        parts = [query]

        for conjunction in conjunctions:
            new_parts = []
            for part in parts:
                if conjunction in part:
                    new_parts.extend(part.split(conjunction))
                else:
                    new_parts.append(part)
            parts = new_parts

        # Method 2: Split on sentence boundaries for complex queries
        if len(parts) == 1 and len(query.split(".")) > 2:
            parts = [s.strip() for s in query.split(".") if s.strip()]

        # Method 3: Identify implicit sub-questions
        implicit_questions = self._identify_implicit_questions(query)
        if implicit_questions:
            parts.extend(implicit_questions)

        # Convert parts to structured sub-questions
        for i, part in enumerate(parts):
            part = part.strip()
            if part and len(part) > 10:  # Filter out very short fragments
                subquestion = {
                    "id": f"sub_{i+1}",
                    "question": self._format_as_question(part),
                    "original_part": part,
                    "priority": self._assess_priority(part, query),
                    "type": self._classify_subquestion(part),
                }
                subquestions.append(subquestion)

        # Limit to max_subquestions
        return subquestions[: self.config["max_subquestions"]]

    def _identify_implicit_questions(self, query: str) -> List[str]:
        """Identify implicit questions within the main query"""
        implicit_questions = []

        # Look for patterns that suggest additional questions
        query_lower = query.lower()

        # "What about X" patterns
        if "what about" in query_lower:
            # This suggests there might be a main question and a follow-up
            main_part = query_lower.split("what about")[0].strip()
            follow_up = "what about" + query_lower.split("what about")[1]
            if main_part:
                implicit_questions.append(main_part)
            implicit_questions.append(follow_up)

        # "How does X relate to Y" patterns suggest two questions
        if any(
            phrase in query_lower
            for phrase in ["relate to", "compared to", "different from"]
        ):
            # Could be decomposed into understanding X and understanding Y
            pass  # Keep this simple for now

        return implicit_questions

    def _format_as_question(self, part: str) -> str:
        """Ensure the part is formatted as a proper question"""
        part = part.strip()

        # If it already looks like a question, return as-is
        if part.endswith("?") or any(
            part.lower().startswith(qw)
            for qw in ["what", "how", "why", "when", "where", "which", "who"]
        ):
            return part

        # Try to convert statement to question
        if any(word in part.lower() for word in ["explain", "describe", "discuss"]):
            return f"Can you {part.lower()}?"

        # Default: add "What is" if it seems like a topic
        return f"What about {part}?" if not part.endswith("?") else part

    def _assess_priority(self, part: str, full_query: str) -> int:
        """Assess priority of sub-question (1=highest, 5=lowest)"""
        part_lower = part.lower()

        # Higher priority for questions that appear first
        position_in_query = full_query.lower().find(part_lower)
        position_score = 5 - min(4, position_in_query // (len(full_query) // 5))

        # Higher priority for questions with key question words
        key_words = ["what", "how", "why"]
        key_word_score = 1 if any(word in part_lower for word in key_words) else 2

        # Higher priority for longer, more substantive questions
        length_score = 1 if len(part.split()) > 8 else 2

        return min(5, max(1, (position_score + key_word_score + length_score) // 3))

    def _classify_subquestion(self, part: str) -> str:
        """Classify the type of sub-question"""
        part_lower = part.lower()

        if any(word in part_lower for word in ["what", "define", "explain"]):
            return "definitional"
        elif any(word in part_lower for word in ["how", "process", "method"]):
            return "procedural"
        elif any(word in part_lower for word in ["why", "because", "reason"]):
            return "causal"
        elif any(word in part_lower for word in ["when", "time", "date"]):
            return "temporal"
        elif any(word in part_lower for word in ["where", "location", "place"]):
            return "spatial"
        elif any(word in part_lower for word in ["compare", "different", "similar"]):
            return "comparative"
        else:
            return "general"

    async def _prioritize_subquestions(
        self, subquestions: List[Dict[str, Any]], context: str
    ) -> List[Dict[str, Any]]:
        """Prioritize sub-questions based on dependencies and importance"""
        # Sort by priority (lower number = higher priority)
        return sorted(subquestions, key=lambda x: x["priority"])

    def _create_decomposed_prompt(
        self, query: str, context: str, subquestions: List[Dict[str, Any]]
    ) -> str:
        """Create a prompt that addresses each sub-question systematically"""

        prompt_parts = [
            "I'll address this complex question by breaking it down into parts and answering each systematically.",
            "",
            f"Original question: {query}",
            "",
        ]

        if context:
            prompt_parts.extend(["Available context:", context, ""])

        prompt_parts.extend(["Breaking this down into sub-questions:", ""])

        for i, subq in enumerate(subquestions, 1):
            prompt_parts.append(f"{i}. {subq['question']} (Type: {subq['type']})")

        prompt_parts.extend(["", "Let me address each part:", ""])

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
            print(f"❌ Error updating QueryDecomposition config: {e}")
            return False
