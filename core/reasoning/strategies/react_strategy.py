"""
ReAct Strategy - Reasoning and Acting for complex analytical queries
Implements iterative reasoning with action planning
"""
from typing import Dict, Any, List
from .base_strategy import BaseReasoningStrategy

class ReActStrategy(BaseReasoningStrategy):
    """Strategy implementing ReAct (Reasoning + Acting) pattern"""
    
    def __init__(self):
        super().__init__(
            name="ReAct",
            description="Iterative reasoning with action planning for complex analytical tasks"
        )
        
        # Configuration
        self.config = {
            'max_reasoning_steps': 5,
            'confidence_threshold': 0.7,
            'min_query_complexity': 'moderate',
            'enable_self_reflection': True,
            'max_tokens_per_step': 200
        }
        
        # Patterns that indicate need for ReAct reasoning
        self.react_indicators = [
            'analyze', 'compare', 'evaluate', 'assess', 'investigate',
            'examine', 'determine', 'figure out', 'work through',
            'step by step', 'reasoning', 'logic', 'because', 'therefore',
            'pros and cons', 'advantages and disadvantages',
            'multiple', 'several', 'various', 'different'
        ]
    
    async def should_handle(self, query: str, context: str = "", analysis: Dict[str, Any] = None) -> float:
        """Determine if query needs ReAct reasoning"""
        query_lower = query.lower()
        
        # Check for ReAct indicators
        indicator_score = sum(1 for indicator in self.react_indicators if indicator in query_lower) / len(self.react_indicators)
        
        # Check complexity from analysis
        complexity_score = 0.0
        if analysis:
            complexity = analysis.get('complexity', 'simple')
            if complexity == 'complex':
                complexity_score = 0.8
            elif complexity == 'moderate':
                complexity_score = 0.6
            else:
                complexity_score = 0.2
            
            # Check intent
            intent = analysis.get('intent', 'factual')
            if intent == 'analytical':
                complexity_score += 0.3
        
        # Check query length and structure
        words = query.split()
        length_score = min(len(words) / 30, 1.0)  # Longer queries often need more reasoning
        
        # Check for question words that indicate complex reasoning
        complex_question_words = ['why', 'how', 'explain', 'describe', 'discuss']
        question_score = sum(0.2 for word in complex_question_words if word in query_lower)
        
        # Combined confidence score
        confidence = (indicator_score * 0.4 + complexity_score * 0.3 + length_score * 0.2 + question_score * 0.1)
        
        return min(confidence, 1.0)
    
    async def process_query(self, query: str, context: str = "", analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using ReAct methodology"""
        
        # Plan reasoning steps
        reasoning_plan = await self._create_reasoning_plan(query, context, analysis)
        
        result = {
            'reasoning_plan': reasoning_plan,
            'processing_type': 'iterative_reasoning',
            'max_steps': self.config['max_reasoning_steps'],
            'enable_reflection': self.config['enable_self_reflection'],
            'reasoning_steps': [
                'thought_generation',
                'action_planning', 
                'observation_analysis',
                'conclusion_synthesis'
            ],
            'optimizations': [
                'step_by_step_processing',
                'intermediate_validation',
                'adaptive_depth_control'
            ]
        }
        
        # Create enhanced prompt with ReAct structure
        result['optimized_prompt'] = self._create_react_prompt(query, context, reasoning_plan)
        
        # Set token limits based on complexity
        complexity = analysis.get('complexity', 'moderate') if analysis else 'moderate'
        if complexity == 'complex':
            result['max_tokens'] = 600
        else:
            result['max_tokens'] = 400
        
        return result
    
    async def _create_reasoning_plan(self, query: str, context: str, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create a step-by-step reasoning plan"""
        plan = []
        
        # Step 1: Understanding
        plan.append({
            'step': 'understand',
            'action': 'Parse and understand the core question',
            'focus': 'Identify key concepts and requirements'
        })
        
        # Step 2: Information gathering (if context available)
        if context:
            plan.append({
                'step': 'gather',
                'action': 'Review available context and information',
                'focus': 'Extract relevant facts and data'
            })
        
        # Step 3: Analysis planning
        if any(word in query.lower() for word in ['compare', 'analyze', 'evaluate']):
            plan.append({
                'step': 'analyze',
                'action': 'Break down the problem into components',
                'focus': 'Structure analytical approach'
            })
        
        # Step 4: Reasoning
        plan.append({
            'step': 'reason',
            'action': 'Apply logical reasoning to reach conclusions',
            'focus': 'Connect evidence to insights'
        })
        
        # Step 5: Synthesis
        plan.append({
            'step': 'synthesize',
            'action': 'Combine findings into coherent response',
            'focus': 'Ensure completeness and clarity'
        })
        
        return plan
    
    def _create_react_prompt(self, query: str, context: str, reasoning_plan: List[Dict[str, str]]) -> str:
        """Create a ReAct-structured prompt"""
        
        prompt_parts = [
            "Use step-by-step reasoning to answer this question thoroughly.",
            "",
            f"Question: {query}",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "Context information:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "Reasoning approach:",
            "Think through this step-by-step:"
        ])
        
        for i, step in enumerate(reasoning_plan, 1):
            prompt_parts.append(f"{i}. {step['action']} - {step['focus']}")
        
        prompt_parts.extend([
            "",
            "Provide your reasoning and conclusion:"
        ])
        
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
            print(f"❌ Error updating ReAct config: {e}")
            return False 