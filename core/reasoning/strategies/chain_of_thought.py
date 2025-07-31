"""
Chain of Thought Strategy - Step-by-step logical reasoning
Implements explicit reasoning chains for complex logical problems
"""

from typing import Any, Dict, List

from .base_strategy import BaseReasoningStrategy


class ChainOfThoughtStrategy(BaseReasoningStrategy):
    """Strategy implementing Chain of Thought reasoning"""

    def __init__(self):
        super().__init__(
            name="ChainOfThought",
            description="Step-by-step logical reasoning for problems requiring explicit thought processes",
        )

        # Configuration
        self.config = {
            "max_reasoning_steps": 6,
            "confidence_threshold": 0.6,
            "enable_step_validation": True,
            "require_explicit_thinking": True,
            "max_tokens_per_step": 150,
        }

        # Patterns that indicate need for step-by-step reasoning
        self.cot_indicators = [
            "solve",
            "calculate",
            "work out",
            "figure out",
            "determine",
            "step by step",
            "how do",
            "how to",
            "process",
            "method",
            "reasoning",
            "logic",
            "because",
            "therefore",
            "thus",
            "first",
            "next",
            "then",
            "finally",
            "sequence",
        ]

        # Problem types that benefit from CoT
        self.problem_types = [
            "mathematical",
            "logical",
            "procedural",
            "causal",
            "sequential",
            "problem_solving",
        ]

    async def should_handle(
        self, query: str, context: str = "", analysis: Dict[str, Any] = None
    ) -> float:
        """Determine if query needs Chain of Thought reasoning"""
        query_lower = query.lower()

        # Check for CoT indicators
        indicator_count = sum(
            1 for indicator in self.cot_indicators if indicator in query_lower
        )
        indicator_score = min(indicator_count / 3, 1.0)  # Normalize to 0-1

        # Check for mathematical content
        math_indicators = [
            "calculate",
            "solve",
            "equation",
            "formula",
            "problem",
            "answer",
        ]
        math_score = sum(
            0.3 for indicator in math_indicators if indicator in query_lower
        )

        # Check for logical reasoning requirements
        logic_indicators = ["if", "then", "because", "therefore", "since", "given that"]
        logic_score = sum(
            0.2 for indicator in logic_indicators if indicator in query_lower
        )

        # Check analysis if provided
        complexity_score = 0.0
        if analysis:
            intent = analysis.get("intent", "factual")
            if intent in ["computational", "analytical"]:
                complexity_score = 0.7

            reasoning_type = analysis.get("reasoning_type", "direct")
            if reasoning_type == "step_by_step":
                complexity_score += 0.2

        # Check for question words that suggest multi-step thinking
        multi_step_words = ["how", "why", "explain", "describe", "show"]
        multi_step_score = sum(0.15 for word in multi_step_words if word in query_lower)

        # Combined confidence score
        confidence = (
            indicator_score * 0.3
            + math_score * 0.25
            + logic_score * 0.2
            + complexity_score * 0.15
            + multi_step_score * 0.1
        )

        return min(confidence, 1.0)

    async def process_query(
        self,
        query: str,
        context: str = "",
        analysis: Dict[str, Any] = None,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """Process query using Chain of Thought methodology"""

        # Identify the problem type
        problem_type = self._identify_problem_type(query, analysis)

        # Create reasoning structure
        reasoning_structure = await self._create_reasoning_structure(
            query, context, problem_type
        )

        result = {
            "problem_type": problem_type,
            "reasoning_structure": reasoning_structure,
            "processing_type": "chain_of_thought",
            "max_steps": self.config["max_reasoning_steps"],
            "step_validation": self.config["enable_step_validation"],
            "reasoning_steps": [
                "problem_identification",
                "step_planning",
                "sequential_reasoning",
                "step_validation",
                "conclusion_synthesis",
            ],
            "optimizations": [
                "explicit_thinking_steps",
                "intermediate_verification",
                "progressive_complexity",
            ],
        }

        # Create CoT prompt
        result["optimized_prompt"] = self._create_cot_prompt(
            query, context, reasoning_structure
        )

        # Set token limits based on problem complexity
        if problem_type in ["mathematical", "logical"]:
            result["max_tokens"] = 500
        else:
            result["max_tokens"] = 400

        return result

    def _identify_problem_type(self, query: str, analysis: Dict[str, Any]) -> str:
        """Identify the type of problem for appropriate CoT approach"""
        query_lower = query.lower()

        # Mathematical problems
        if any(
            word in query_lower
            for word in ["calculate", "solve", "equation", "math", "number"]
        ):
            return "mathematical"

        # Logical reasoning problems
        if any(
            word in query_lower
            for word in ["if", "then", "logic", "reasoning", "deduce"]
        ):
            return "logical"

        # Procedural problems
        if any(
            word in query_lower
            for word in ["how to", "process", "method", "procedure", "steps"]
        ):
            return "procedural"

        # Causal reasoning
        if any(
            word in query_lower
            for word in ["why", "because", "cause", "reason", "result"]
        ):
            return "causal"

        # Sequential problems
        if any(
            word in query_lower
            for word in ["sequence", "order", "first", "next", "then"]
        ):
            return "sequential"

        # Default to general problem solving
        return "problem_solving"

    async def _create_reasoning_structure(
        self, query: str, context: str, problem_type: str
    ) -> List[Dict[str, str]]:
        """Create a reasoning structure based on problem type"""
        structure = []

        # Common first step: understand the problem
        structure.append(
            {
                "step": "understand",
                "description": "Clearly identify what is being asked",
                "focus": "Parse the question and identify key elements",
            }
        )

        # Problem-specific steps
        if problem_type == "mathematical":
            structure.extend(
                [
                    {
                        "step": "identify_knowns",
                        "description": "List known values and relationships",
                        "focus": "Organize given information",
                    },
                    {
                        "step": "identify_unknowns",
                        "description": "Identify what needs to be found",
                        "focus": "Define target variables",
                    },
                    {
                        "step": "plan_solution",
                        "description": "Choose appropriate formulas or methods",
                        "focus": "Select solution approach",
                    },
                    {
                        "step": "execute_calculation",
                        "description": "Perform calculations step by step",
                        "focus": "Show each calculation",
                    },
                    {
                        "step": "verify_result",
                        "description": "Check if the answer makes sense",
                        "focus": "Validate solution",
                    },
                ]
            )
        elif problem_type == "logical":
            structure.extend(
                [
                    {
                        "step": "identify_premises",
                        "description": "List all given facts and assumptions",
                        "focus": "Establish logical foundation",
                    },
                    {
                        "step": "apply_logic",
                        "description": "Use logical rules to derive conclusions",
                        "focus": "Show logical connections",
                    },
                    {
                        "step": "check_consistency",
                        "description": "Verify logical consistency",
                        "focus": "Ensure no contradictions",
                    },
                ]
            )
        elif problem_type == "procedural":
            structure.extend(
                [
                    {
                        "step": "break_down_process",
                        "description": "Divide into manageable steps",
                        "focus": "Create step sequence",
                    },
                    {
                        "step": "detail_each_step",
                        "description": "Explain each step clearly",
                        "focus": "Provide specific instructions",
                    },
                    {
                        "step": "consider_alternatives",
                        "description": "Mention alternative approaches if relevant",
                        "focus": "Show flexibility",
                    },
                ]
            )
        else:
            # General problem solving structure
            structure.extend(
                [
                    {
                        "step": "analyze_components",
                        "description": "Break down into parts",
                        "focus": "Identify key components",
                    },
                    {
                        "step": "reason_through",
                        "description": "Work through systematically",
                        "focus": "Apply logical reasoning",
                    },
                    {
                        "step": "synthesize_conclusion",
                        "description": "Combine insights",
                        "focus": "Form coherent answer",
                    },
                ]
            )

        return structure

    def _create_cot_prompt(
        self, query: str, context: str, reasoning_structure: List[Dict[str, str]]
    ) -> str:
        """Create a Chain of Thought structured prompt"""

        prompt_parts = [
            "Think through this step by step, showing your reasoning clearly.",
            "",
            f"Question: {query}",
            "",
        ]

        if context:
            prompt_parts.extend(["Available information:", context, ""])

        prompt_parts.extend(["Let me work through this systematically:", ""])

        # Add reasoning steps
        for i, step in enumerate(reasoning_structure, 1):
            prompt_parts.append(f"Step {i}: {step['description']}")

        prompt_parts.extend(["", "Working through each step:", ""])

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
            print(f"❌ Error updating ChainOfThought config: {e}")
            return False
