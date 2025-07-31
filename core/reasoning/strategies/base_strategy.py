"""
Base Strategy - Abstract base class for all reasoning strategies
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class BaseReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.performance_stats = {
            "total_queries": 0,
            "success_rate": 0.0,
            "average_time": 0.0,
            "recent_times": [],
        }

    @abstractmethod
    async def should_handle(
        self, query: str, context: str = "", analysis: Dict[str, Any] = None
    ) -> float:
        """
        Determine if this strategy should handle the query
        Returns confidence score 0.0-1.0
        """
        pass

    @abstractmethod
    async def process_query(
        self,
        query: str,
        context: str = "",
        analysis: Dict[str, Any] = None,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Process the query using this strategy
        Returns enhanced analysis and processing instructions
        """
        pass

    def _build_conversational_prompt(
        self,
        query: str,
        context: str = "",
        conversation_context: str = "",
        system_prompt: str = "",
    ) -> str:
        """Build a prompt that includes conversation context for natural dialogue"""
        prompt_parts = []

        # Add system prompt if provided
        if system_prompt:
            prompt_parts.append(system_prompt)
            prompt_parts.append("")

        # Add conversation history if available
        if conversation_context:
            prompt_parts.append("Previous conversation:")
            prompt_parts.append(conversation_context)

        # Add document context if available
        if context:
            prompt_parts.append("Relevant context:")
            prompt_parts.append(context)
            prompt_parts.append("")

        # Add current query
        prompt_parts.append(f"Human: {query}")
        prompt_parts.append("Assistant: ")

        return "\n".join(prompt_parts)

    async def execute(
        self,
        query: str,
        context: str = "",
        analysis: Dict[str, Any] = None,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """Execute the strategy with performance tracking"""
        start_time = time.time()

        try:
            # Check if strategy should handle this query
            confidence = await self.should_handle(query, context, analysis)

            if confidence < 0.3:  # Low confidence threshold
                return {
                    "strategy": self.name,
                    "handled": False,
                    "confidence": confidence,
                    "reason": "Low confidence for this query type",
                }

            # Process the query with conversation context
            result = await self.process_query(
                query, context, analysis, conversation_context
            )

            # Add strategy metadata
            result.update(
                {
                    "strategy": self.name,
                    "handled": True,
                    "confidence": confidence,
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Update performance stats
            self._update_stats(time.time() - start_time, True)

            return result

        except Exception as e:
            error_result = {
                "strategy": self.name,
                "handled": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

            # Update performance stats for failure
            self._update_stats(time.time() - start_time, False)

            return error_result

    def _update_stats(self, execution_time: float, success: bool):
        """Update performance statistics"""
        self.performance_stats["total_queries"] += 1
        self.performance_stats["recent_times"].append(execution_time)

        # Keep only last 20 times
        if len(self.performance_stats["recent_times"]) > 20:
            self.performance_stats["recent_times"] = self.performance_stats[
                "recent_times"
            ][-20:]

        # Update averages
        if self.performance_stats["recent_times"]:
            self.performance_stats["average_time"] = sum(
                self.performance_stats["recent_times"]
            ) / len(self.performance_stats["recent_times"])

        # Update success rate (simplified)
        if success:
            current_rate = self.performance_stats["success_rate"]
            total = self.performance_stats["total_queries"]
            self.performance_stats["success_rate"] = (
                (current_rate * (total - 1)) + 1.0
            ) / total
        else:
            current_rate = self.performance_stats["success_rate"]
            total = self.performance_stats["total_queries"]
            self.performance_stats["success_rate"] = (
                current_rate * (total - 1)
            ) / total

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        return {
            "name": self.name,
            "description": self.description,
            "performance": self.performance_stats.copy(),
        }

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get strategy-specific configuration"""
        pass

    @abstractmethod
    async def update_config(self, config: Dict[str, Any]) -> bool:
        """Update strategy configuration"""
        pass
