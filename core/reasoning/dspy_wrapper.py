"""
DSPy Wrapper - Integration layer for DSPy with MLX local models
Provides automatic prompt optimization and query reasoning
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json

try:
    import dspy
    from dspy import Signature, InputField, OutputField, Module
    from dspy.teleprompt import BootstrapFewShot
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available. Install with: pip install dspy-ai")
    
    # Create dummy classes for when DSPy is not available
    class dspy:
        class LM:
            def __init__(self, *args, **kwargs):
                pass
        class Signature:
            pass
        class Module:
            def __init__(self, *args, **kwargs):
                pass
        class InputField:
            @staticmethod
            def __call__(*args, **kwargs):
                return None
        class OutputField:
            @staticmethod
            def __call__(*args, **kwargs):
                return None
        class teleprompt:
            class BootstrapFewShot:
                def __init__(self, *args, **kwargs):
                    pass
                def compile(self, *args, **kwargs):
                    return None

from config import settings

if DSPY_AVAILABLE:
    class LocalMLXLM(dspy.LM):
        """Custom DSPy Language Model that wraps our MLX manager"""
        
        def __init__(self, mlx_manager):
            self.mlx_manager = mlx_manager
            self.model_name = "local-mlx"
            self.provider = "local"
            self.kwargs = {}  # Initialize kwargs for DSPy compatibility
            self.model = "local-mlx"  # Add model attribute for DSPy compatibility
        
        def __call__(self, prompt: str, **kwargs) -> str:
            """Generate response using MLX manager - OPTIMIZED for performance"""
            try:
                # Store kwargs for DSPy compatibility
                self.kwargs = kwargs
                
                # Extract parameters
                max_tokens = kwargs.get('max_tokens', settings.max_tokens)
                
                # CRITICAL: Use the synchronous method if available to avoid event loop overhead
                if hasattr(self.mlx_manager, 'generate_response_sync'):
                    response = self.mlx_manager.generate_response_sync(prompt, max_tokens)
                else:
                    # Fallback: Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, use create_task
                        import concurrent.futures
                        
                        # Use thread executor to avoid blocking the main loop
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.mlx_manager.generate_response(prompt, max_tokens)
                            )
                            response = future.result(timeout=30)
                            
                    except RuntimeError:
                        # No event loop running, safe to use asyncio.run
                        response = asyncio.run(
                            self.mlx_manager.generate_response(prompt, max_tokens)
                        )
                
                return response
                
            except Exception as e:
                print(f"❌ Error in LocalMLXLM: {e}")
                return "I apologize, but I encountered an error processing your request."
        
        def copy(self):
            """Create a copy of this LM"""
            return LocalMLXLM(self.mlx_manager)

    class QueryAnalysis(dspy.Signature):
        """Analyze query complexity and intent"""
        query = dspy.InputField(desc="User's input query")
        intent = dspy.OutputField(desc="Primary intent: factual, conversational, analytical, computational, creative")
        complexity = dspy.OutputField(desc="Complexity level: simple, moderate, complex")
        reasoning_type = dspy.OutputField(desc="Best reasoning approach: direct, step_by_step, contextual, decomposed")
        confidence = dspy.OutputField(desc="Confidence score 0-1 for analysis accuracy")

    class ResponseOptimization(dspy.Signature):
        """Optimize response generation based on query analysis"""
        query = dspy.InputField(desc="Original user query")
        intent = dspy.InputField(desc="Detected intent")
        complexity = dspy.InputField(desc="Complexity level")
        context = dspy.InputField(desc="Retrieved context from documents", default="")
        optimized_prompt = dspy.OutputField(desc="Optimized prompt for the specific reasoning strategy")
        strategy_params = dspy.OutputField(desc="JSON parameters for strategy execution")

    class DSPyOptimizedReasoner(dspy.Module):
        """DSPy-powered reasoning module with automatic optimization"""
        
        def __init__(self, mlx_manager):
            super().__init__()
            self.mlx_manager = mlx_manager
            self.lm = LocalMLXLM(mlx_manager)
            self.query_analyzer = dspy.ChainOfThought(QueryAnalysis)
            self.response_optimizer = dspy.ChainOfThought(ResponseOptimization)
            self.compiled = False
            self.training_examples = []
            
        def forward(self, query: str, context: str = "", reasoning_history: List[Dict] = None):
            """Process query with DSPy optimization"""
            try:
                # Analyze query
                analysis = self.query_analyzer(query=query)
                
                # Optimize response based on analysis
                optimization = self.response_optimizer(
                    query=query,
                    intent=analysis.intent,
                    complexity=analysis.complexity,
                    context=context
                )
                
                return {
                    'analysis': {
                        'intent': analysis.intent,
                        'complexity': analysis.complexity,
                        'reasoning_type': analysis.reasoning_type,
                        'confidence': float(analysis.confidence) if analysis.confidence.replace('.','',1).isdigit() else 0.7
                    },
                    'optimized_prompt': optimization.optimized_prompt,
                    'strategy_params': optimization.strategy_params,
                    'dspy_available': True
                }
                
            except Exception as e:
                print(f"❌ DSPy reasoning error: {e}")
                return {
                    'analysis': {
                        'intent': 'conversational',
                        'complexity': 'moderate',
                        'reasoning_type': 'direct',
                        'confidence': 0.5
                    },
                    'optimized_prompt': query,
                    'strategy_params': '{}',
                    'dspy_available': True,
                    'error': str(e)
                }

else:
    # Fallback classes when DSPy is not available
    class LocalMLXLM:
        """Fallback class when DSPy is not available"""
        
        def __init__(self, mlx_manager):
            self.mlx_manager = mlx_manager
            self.model_name = "local-mlx"
            self.provider = "local"
            self.kwargs = {}
            self.model = "local-mlx"
            
        def __call__(self, prompt: str, **kwargs) -> str:
            """Generate response using MLX manager"""
            try:
                max_tokens = kwargs.get('max_tokens', settings.max_tokens)
                if hasattr(self.mlx_manager, 'generate_response_sync'):
                    return self.mlx_manager.generate_response_sync(prompt, max_tokens)
                else:
                    return asyncio.run(self.mlx_manager.generate_response(prompt, max_tokens))
            except Exception as e:
                print(f"❌ Error in LocalMLXLM fallback: {e}")
                return "I apologize, but I encountered an error processing your request."
        
        def copy(self):
            return LocalMLXLM(self.mlx_manager)

    class QueryAnalysis:
        """Fallback query analysis when DSPy is not available"""
        pass

    class ResponseOptimization:
        """Fallback response optimization when DSPy is not available"""
        pass

    class DSPyOptimizedReasoner:
        """Fallback reasoner when DSPy is not available"""
        
        def __init__(self, mlx_manager):
            self.mlx_manager = mlx_manager
            self.lm = LocalMLXLM(mlx_manager)
            self.compiled = False
            self.training_examples = []
            
        def forward(self, query: str, context: str = "", reasoning_history: List[Dict] = None):
            """Fallback reasoning without DSPy"""
            return {
                'analysis': {
                    'intent': 'conversational',
                    'complexity': 'moderate',
                    'reasoning_type': 'direct',
                    'confidence': 0.5
                },
                'optimized_prompt': query,
                'strategy_params': '{}',
                'dspy_available': False
            }

class DSPyWrapper:
    """Wrapper class for DSPy integration with optional fallback"""
    
    def __init__(self, mlx_manager):
        self.mlx_manager = mlx_manager
        self.dspy_available = DSPY_AVAILABLE
        
        if DSPY_AVAILABLE:
            # Initialize DSPy with local MLX model
            self.lm = LocalMLXLM(mlx_manager)
            dspy.settings.configure(lm=self.lm)
            self.reasoner = DSPyOptimizedReasoner(mlx_manager)
            print("✅ DSPy reasoning system initialized!")
        else:
            # Fallback without DSPy
            self.lm = LocalMLXLM(mlx_manager)
            self.reasoner = DSPyOptimizedReasoner(mlx_manager)
            print("⚠️  Using fallback reasoning (DSPy not available)")
        
        self.optimization_history = []
        self.performance_stats = {
            'total_queries': 0,
            'avg_confidence': 0.0,
            'strategy_usage': {},
            'optimization_success_rate': 0.0
        }

    async def analyze_and_optimize(self, query: str, context: str = "", reasoning_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze query and optimize reasoning approach"""
        try:
            start_time = time.time()
            
            # Use DSPy reasoning if available, otherwise fallback
            result = self.reasoner.forward(query, context, reasoning_history)
            
            # Track performance
            processing_time = time.time() - start_time
            self._update_performance_stats(result, processing_time)
            
            # Add metadata
            result['processing_time'] = processing_time
            result['timestamp'] = datetime.now().isoformat()
            result['dspy_wrapper_version'] = '1.1.0'
            
            return result
            
        except Exception as e:
            print(f"❌ Error in DSPy analysis: {e}")
            return {
                'analysis': {
                    'intent': 'conversational',
                    'complexity': 'moderate',
                    'reasoning_type': 'direct',
                    'confidence': 0.5
                },
                'optimized_prompt': query,
                'strategy_params': '{}',
                'dspy_available': self.dspy_available,
                'error': str(e),
                'processing_time': 0.1,
                'timestamp': datetime.now().isoformat()
            }

    async def analyze_query(self, query: str, context: str = "") -> Dict[str, Any]:
        """Analyze query - alias for analyze_and_optimize for compatibility"""
        result = await self.analyze_and_optimize(query, context)
        # Return just the analysis part for compatibility
        return result.get('analysis', {
            'intent': 'conversational',
            'complexity': 'moderate', 
            'reasoning_type': 'direct',
            'confidence': 0.5
        })

    def _update_performance_stats(self, result: Dict[str, Any], processing_time: float):
        """Update performance statistics"""
        self.performance_stats['total_queries'] += 1
        
        if 'analysis' in result:
            confidence = result['analysis'].get('confidence', 0.5)
            reasoning_type = result['analysis'].get('reasoning_type', 'direct')
            
            # Update average confidence
            total = self.performance_stats['total_queries']
            current_avg = self.performance_stats['avg_confidence']
            self.performance_stats['avg_confidence'] = (current_avg * (total - 1) + confidence) / total
            
            # Update strategy usage
            if reasoning_type not in self.performance_stats['strategy_usage']:
                self.performance_stats['strategy_usage'][reasoning_type] = 0
            self.performance_stats['strategy_usage'][reasoning_type] += 1

    async def optimize_for_examples(self, examples: List[Dict[str, Any]]) -> bool:
        """Optimize DSPy system with example queries"""
        if not DSPY_AVAILABLE:
            print("⚠️  DSPy optimization skipped - DSPy not available")
            return False
            
        try:
            print("🧠 Starting DSPy optimization with examples...")
            
            # Convert examples to DSPy format
            training_examples = []
            for example in examples[:10]:  # Limit to prevent overwhelming
                if 'query' in example:
                    training_examples.append(
                        dspy.Example(
                            query=example['query'],
                            context=example.get('context', ''),
                            expected_strategy=example.get('expected_strategy', 'auto')
                        ).with_inputs('query', 'context')
                    )
            
            if not training_examples:
                print("⚠️  No valid training examples provided")
                return False
            
            # Optimize with BootstrapFewShot
            teleprompter = BootstrapFewShot(metric=None, max_bootstrapped_demos=4)
            
            try:
                optimized_reasoner = teleprompter.compile(self.reasoner, trainset=training_examples)
                self.reasoner = optimized_reasoner
                self.reasoner.compiled = True
                
                print(f"✅ DSPy optimization completed with {len(training_examples)} examples")
                return True
                
            except Exception as e:
                print(f"⚠️  DSPy optimization warning: {e}")
                return False
                
        except Exception as e:
            print(f"❌ DSPy optimization failed: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        stats['dspy_available'] = self.dspy_available
        stats['optimization_status'] = 'compiled' if hasattr(self.reasoner, 'compiled') and self.reasoner.compiled else 'uncompiled'
        return stats

    def get_available_strategies(self) -> List[str]:
        """Get list of available reasoning strategies"""
        if DSPY_AVAILABLE:
            return ['direct', 'step_by_step', 'contextual', 'decomposed', 'chain_of_thought', 'react']
        else:
            return ['direct', 'contextual']  # Fallback strategies

    async def create_optimized_prompt(self, query: str, strategy: str = "auto", context: str = "", 
                                    max_tokens: int = None) -> Dict[str, Any]:
        """Create an optimized prompt for a specific strategy"""
        try:
            # Get analysis and optimization
            analysis_result = await self.analyze_and_optimize(query, context)
            
            # Extract strategy if auto
            if strategy == "auto":
                strategy = analysis_result['analysis'].get('reasoning_type', 'direct')
            
            # Build optimized prompt based on strategy
            optimized_prompt = analysis_result.get('optimized_prompt', query)
            
            # Add context if available
            if context:
                optimized_prompt = f"Context: {context}\n\nQuery: {optimized_prompt}"
            
            # Determine max_tokens based on complexity
            if max_tokens is None:
                complexity = analysis_result['analysis'].get('complexity', 'moderate')
                if complexity == 'simple':
                    max_tokens = settings.max_tokens // 2
                elif complexity == 'complex':
                    max_tokens = min(settings.max_tokens * 2, 1024)
                else:
                    max_tokens = settings.max_tokens
            
            return {
                'optimized_prompt': optimized_prompt,
                'strategy': strategy,
                'max_tokens': max_tokens,
                'analysis': analysis_result['analysis'],
                'dspy_available': self.dspy_available,
                'processing_time': analysis_result.get('processing_time', 0.0)
            }
            
        except Exception as e:
            print(f"❌ Error creating optimized prompt: {e}")
            return {
                'optimized_prompt': query,
                'strategy': 'direct',
                'max_tokens': max_tokens or settings.max_tokens,
                'analysis': {
                    'intent': 'conversational',
                    'complexity': 'moderate',
                    'reasoning_type': 'direct',
                    'confidence': 0.5
                },
                'dspy_available': self.dspy_available,
                'error': str(e)
            } 