"""
Hybrid Stack Manager - Orchestrates multiple reasoning strategies
Selects and coordinates the best reasoning approach for each query
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from config import settings
from .dspy_wrapper import DSPyWrapper
from .strategies.base_strategy import BaseReasoningStrategy
from .strategies.fast_path import FastPathStrategy
from .strategies.react_strategy import ReActStrategy
from .strategies.chain_of_thought import ChainOfThoughtStrategy
from .strategies.query_decomposition import QueryDecompositionStrategy
from .strategies.contextual_reasoning import ContextualReasoningStrategy

class HybridStackManager:
    """Manager for hybrid reasoning stacks with strategy orchestration"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, mlx_manager):
        if cls._instance is None:
            cls._instance = super(HybridStackManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, mlx_manager):
        # Only initialize once
        if HybridStackManager._initialized:
            return
        
        self.mlx_manager = mlx_manager
        self.dspy_wrapper = DSPyWrapper(mlx_manager)
        
        # Initialize strategies
        self.strategies = {
            'fast_path': FastPathStrategy(),
            'react': ReActStrategy(),
            'chain_of_thought': ChainOfThoughtStrategy(),
            'query_decomposition': QueryDecompositionStrategy(),
            'contextual_reasoning': ContextualReasoningStrategy()
        }
        
        # Predefined hybrid stacks
        self.hybrid_stacks = {
            'auto': {
                'name': 'Auto Selection',
                'description': 'Automatically select the best strategy for each query',
                'strategy_order': ['fast_path', 'contextual_reasoning', 'chain_of_thought', 'react', 'query_decomposition'],
                'min_confidence': 0.7,
                'fallback_strategy': 'contextual_reasoning'
            },
            'speed_optimized': {
                'name': 'Speed Optimized',
                'description': 'Prioritize response speed over complexity',
                'strategy_order': ['fast_path', 'contextual_reasoning'],
                'min_confidence': 0.6,
                'fallback_strategy': 'fast_path'
            },
            'quality_optimized': {
                'name': 'Quality Optimized',
                'description': 'Prioritize response quality and thoroughness',
                'strategy_order': ['react', 'chain_of_thought', 'query_decomposition', 'contextual_reasoning'],
                'min_confidence': 0.8,
                'fallback_strategy': 'chain_of_thought'
            },
            'analytical': {
                'name': 'Analytical Focus',
                'description': 'Optimized for complex analytical and research queries',
                'strategy_order': ['query_decomposition', 'react', 'chain_of_thought', 'contextual_reasoning'],
                'min_confidence': 0.7,
                'fallback_strategy': 'react'
            },
            'contextual': {
                'name': 'Context-Aware',
                'description': 'Prioritize document context and knowledge integration',
                'strategy_order': ['contextual_reasoning', 'chain_of_thought', 'query_decomposition'],
                'min_confidence': 0.6,
                'fallback_strategy': 'contextual_reasoning'
            },
            'conversational': {
                'name': 'Conversational',
                'description': 'Optimized for natural conversation and quick responses',
                'strategy_order': ['fast_path', 'contextual_reasoning', 'chain_of_thought'],
                'min_confidence': 0.5,
                'fallback_strategy': 'fast_path'
            }
        }
        
        # Current configuration
        self.current_stack = 'auto'
        self.enable_dspy = True
        self.performance_tracking = True
        
        # Performance statistics
        self.stats = {
            'total_queries': 0,
            'strategy_usage': {name: 0 for name in self.strategies.keys()},
            'stack_usage': {name: 0 for name in self.hybrid_stacks.keys()},
            'average_processing_time': 0.0,
            'success_rate': 0.0,
            'recent_performance': []
        }
        
        # Mark as initialized
        HybridStackManager._initialized = True
    
    async def initialize(self):
        """Initialize the hybrid stack manager"""
        print("🧠 Initializing Hybrid Stack Manager...")
        
        # DSPy wrapper is now initialized in constructor
        # No additional async initialization needed
        
        print("✅ Hybrid Stack Manager initialized!")
    
    def _format_conversation_context(self, conversation_context: List[Dict] = None) -> str:
        """Format conversation context for inclusion in prompts"""
        if not conversation_context or len(conversation_context) == 0:
            return ""
        
        # Build conversation history string
        conversation_lines = []
        for msg in conversation_context[-6:]:  # Last 6 messages (3 exchanges)
            if isinstance(msg, dict) and 'type' in msg and 'message' in msg:
                if msg['type'] == 'user':
                    conversation_lines.append(f"Human: {msg['message']}")
                elif msg['type'] == 'assistant':
                    conversation_lines.append(f"Assistant: {msg['message']}")
        
        if conversation_lines:
            return "\n".join(conversation_lines) + "\n"
        
        return ""
    
    async def process_query(self, query: str, context: str = "", conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """Main query processing with hybrid strategy selection"""
        start_time = time.time()
        
        try:
            # Format conversation context for use in prompts
            conversation_history = self._format_conversation_context(conversation_context)
            
            # PERFORMANCE OPTIMIZATION: Check FastPath first for ultra-simple queries
            fastpath_confidence = await self.strategies['fast_path'].should_handle(query, context)
            
            # ULTRA-FAST BYPASS: Skip all reasoning for very simple queries
            if fastpath_confidence > 0.95:
                print(f"🚀 Ultra-fast bypass: FastPath confidence={fastpath_confidence:.3f}")
                
                # Process with minimal overhead, including conversation context
                result = await self.strategies['fast_path'].process_query(
                    query, context, conversation_context=conversation_history
                )
                result['bypass_reasoning'] = True
                result['total_processing_time'] = time.time() - start_time
                
                return result
            
            # Regular processing for queries that need reasoning
            analysis = {'intent': 'factual', 'complexity': 'moderate'}
            
            # Try DSPy analysis if enabled and available
            if self.enable_dspy and self.dspy_wrapper:
                try:
                    dspy_analysis = await self.dspy_wrapper.analyze_query(query, context)
                    if dspy_analysis:
                        analysis.update(dspy_analysis)
                        print(f"🧠 Query analyzed: {analysis.get('intent', 'unknown')} | {analysis.get('complexity', 'unknown')} | {time.time() - start_time:.3f}s")
                    else:
                        print("🧠 DSPy analysis returned empty result")
                except Exception as e:
                    print(f"❌ Error in DSPy reasoning: {e}")
                    # Continue with fallback analysis
            
            # Select best strategy
            selected_strategy, confidence = await self._select_strategy(query, context, analysis)
            strategy_name = selected_strategy.name
            
            print(f"🧠 Strategy: {strategy_name} | Confidence: {confidence:.3f}")
            if conversation_history:
                print(f"💬 Using conversation context: {len(conversation_history)} characters")
            
            # Execute strategy with conversation context
            result = await selected_strategy.process_query(
                query, context, analysis, conversation_context=conversation_history
            )
            
            # Add metadata
            result.update({
                'strategy': strategy_name,
                'confidence': confidence,
                'analysis': analysis,
                'conversation_context_used': bool(conversation_history),
                'total_processing_time': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            print(f"❌ Error in hybrid processing: {e}")
            # Fallback to direct processing
            return {
                'handled': True,
                'optimized_prompt': self._create_fallback_prompt(query, context, conversation_context),
                'max_tokens': settings.max_tokens,
                'strategy': 'fallback',
                'confidence': 0.5,
                'error': str(e),
                'total_processing_time': time.time() - start_time
            }
    
    def _create_fallback_prompt(self, query: str, context: str = "", conversation_context: List[Dict] = None) -> str:
        """Create a fallback prompt with conversation context when errors occur"""
        conversation_history = self._format_conversation_context(conversation_context)
        
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            prompt_parts.append(conversation_history)
        
        if context:
            prompt_parts.append("Relevant context:")
            prompt_parts.append(context)
            prompt_parts.append("")
        
        prompt_parts.append(f"Human: {query}")
        prompt_parts.append("Assistant: ")
        
        return "\n".join(prompt_parts)
    
    async def _select_strategy(self, query: str, context: str, analysis: Dict[str, Any]) -> Tuple[BaseReasoningStrategy, float]:
        """Select the best strategy based on current stack configuration"""
        
        current_stack_config = self.hybrid_stacks[self.current_stack]
        strategy_order = current_stack_config['strategy_order']
        min_confidence = current_stack_config['min_confidence']
        
        best_strategy = None
        best_confidence = 0.0
        
        # Try strategies in the configured order
        for strategy_name in strategy_order:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                confidence = await strategy.should_handle(query, context, analysis)
                
                print(f"🔍 {strategy.name}: confidence={confidence:.3f}")
                
                if confidence >= min_confidence:
                    best_strategy = strategy
                    best_confidence = confidence
                    break
                elif confidence > best_confidence:
                    best_strategy = strategy
                    best_confidence = confidence
        
        # If no strategy meets minimum confidence, use fallback
        if best_strategy is None or best_confidence < min_confidence:
            fallback_name = current_stack_config['fallback_strategy']
            if fallback_name in self.strategies:
                best_strategy = self.strategies[fallback_name]
                best_confidence = 0.5  # Default fallback confidence
                print(f"🔄 Using fallback strategy: {best_strategy.name}")
        
        # Ultimate fallback to fast_path if something goes wrong
        if best_strategy is None:
            best_strategy = self.strategies['fast_path']
            best_confidence = 0.3
            print("⚠️ Using ultimate fallback: FastPath")
        
        return best_strategy, best_confidence
    
    async def _update_performance_stats(self, strategy_name: str, processing_time: float, success: bool):
        """Update performance statistics"""
        self.stats['total_queries'] += 1
        
        if strategy_name in self.stats['strategy_usage']:
            self.stats['strategy_usage'][strategy_name] += 1
        
        if self.current_stack in self.stats['stack_usage']:
            self.stats['stack_usage'][self.current_stack] += 1
        
        # Update timing
        self.stats['recent_performance'].append({
            'strategy': strategy_name,
            'time': processing_time,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 50 performance records
        if len(self.stats['recent_performance']) > 50:
            self.stats['recent_performance'] = self.stats['recent_performance'][-50:]
        
        # Update averages
        recent_times = [p['time'] for p in self.stats['recent_performance']]
        if recent_times:
            self.stats['average_processing_time'] = sum(recent_times) / len(recent_times)
        
        recent_successes = [p['success'] for p in self.stats['recent_performance']]
        if recent_successes:
            self.stats['success_rate'] = sum(recent_successes) / len(recent_successes)
    
    async def switch_stack(self, stack_name: str) -> bool:
        """Switch to a different hybrid stack"""
        if stack_name in self.hybrid_stacks:
            self.current_stack = stack_name
            print(f"🔄 Switched to hybrid stack: {stack_name}")
            return True
        else:
            print(f"❌ Unknown hybrid stack: {stack_name}")
            return False
    
    def get_available_stacks(self) -> Dict[str, Dict[str, Any]]:
        """Get all available hybrid stacks"""
        return self.hybrid_stacks.copy()
    
    def get_current_stack(self) -> str:
        """Get current hybrid stack name"""
        return self.current_stack
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        strategy_stats = {}
        for name, strategy in self.strategies.items():
            strategy_stats[name] = strategy.get_stats()
        
        return {
            'overall_stats': self.stats.copy(),
            'strategy_stats': strategy_stats,
            'dspy_stats': self.dspy_wrapper.get_performance_stats() if self.enable_dspy else {},
            'current_configuration': {
                'current_stack': self.current_stack,
                'enable_dspy': self.enable_dspy,
                'performance_tracking': self.performance_tracking
            }
        }
    
    async def optimize_stack_for_examples(self, examples: List[Dict[str, Any]]) -> bool:
        """Optimize current stack based on example queries and expected outcomes"""
        try:
            print(f"🎯 Optimizing stack for {len(examples)} examples...")
            
            # Analyze examples to understand patterns
            analysis_results = []
            for example in examples:
                query = example.get('query', '')
                context = example.get('context', '')
                expected_strategy = example.get('expected_strategy', '')
                
                if query:
                    # Get current strategy selection
                    analysis = await self.dspy_wrapper.analyze_query(query, context) if self.enable_dspy else {}
                    selected_strategy, confidence = await self._select_strategy(query, context, analysis)
                    
                    analysis_results.append({
                        'query': query,
                        'selected_strategy': selected_strategy.name,
                        'confidence': confidence,
                        'expected_strategy': expected_strategy,
                        'match': selected_strategy.name == expected_strategy
                    })
            
            # Calculate current accuracy
            matches = sum(1 for result in analysis_results if result['match'])
            accuracy = matches / len(analysis_results) if analysis_results else 0
            
            print(f"📊 Current accuracy: {accuracy:.2%} ({matches}/{len(analysis_results)})")
            
            # If accuracy is low, suggest stack adjustments
            if accuracy < 0.7:
                print("🔧 Suggesting stack optimizations...")
                # This could be enhanced with ML-based optimization
                
            # Optimize DSPy components if available
            if self.enable_dspy and len(examples) >= 5:
                dspy_examples = [{'query': ex['query'], 'context': ex.get('context', '')} for ex in examples]
                optimization_success = await self.dspy_wrapper.optimize_for_examples(dspy_examples)
                if optimization_success:
                    print("✅ DSPy optimization completed!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error optimizing stack: {e}")
            return False
    
    async def create_custom_stack(self, name: str, config: Dict[str, Any]) -> bool:
        """Create a custom hybrid stack configuration"""
        try:
            required_fields = ['description', 'strategy_order', 'min_confidence', 'fallback_strategy']
            
            if not all(field in config for field in required_fields):
                print(f"❌ Missing required fields for custom stack: {required_fields}")
                return False
            
            # Validate strategy names
            for strategy_name in config['strategy_order']:
                if strategy_name not in self.strategies:
                    print(f"❌ Unknown strategy: {strategy_name}")
                    return False
            
            if config['fallback_strategy'] not in self.strategies:
                print(f"❌ Unknown fallback strategy: {config['fallback_strategy']}")
                return False
            
            # Add custom stack
            self.hybrid_stacks[name] = {
                'name': config.get('display_name', name),
                'description': config['description'],
                'strategy_order': config['strategy_order'],
                'min_confidence': config['min_confidence'],
                'fallback_strategy': config['fallback_strategy'],
                'custom': True
            }
            
            # Initialize usage tracking
            self.stats['stack_usage'][name] = 0
            
            print(f"✅ Created custom hybrid stack: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating custom stack: {e}")
            return False
    
    async def update_configuration(self, config: Dict[str, Any]) -> bool:
        """Update hybrid manager configuration"""
        try:
            if 'enable_dspy' in config:
                self.enable_dspy = config['enable_dspy']
            
            if 'performance_tracking' in config:
                self.performance_tracking = config['performance_tracking']
            
            if 'current_stack' in config and config['current_stack'] in self.hybrid_stacks:
                self.current_stack = config['current_stack']
            
            # Update strategy configurations
            if 'strategy_configs' in config:
                for strategy_name, strategy_config in config['strategy_configs'].items():
                    if strategy_name in self.strategies:
                        await self.strategies[strategy_name].update_config(strategy_config)
            
            print("✅ Hybrid manager configuration updated")
            return True
            
        except Exception as e:
            print(f"❌ Error updating configuration: {e}")
            return False 