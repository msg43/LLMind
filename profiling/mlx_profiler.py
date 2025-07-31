"""
MLX Performance Profiler and Optimizer
Comprehensive profiling and optimization for MLX generation on Apple Silicon
"""

import asyncio
import gc
import json
import statistics
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import psutil

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import generate, load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import sys

sys.path.append("../")
from config import settings
from core.mlx_manager import MLXManager


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single generation"""

    tokens_per_second: float
    total_time: float
    prompt_processing_time: float
    generation_time: float
    memory_usage: Dict[str, float]
    gpu_utilization: float
    cpu_usage: float
    model_name: str
    prompt_length: int
    output_length: int
    batch_size: int
    temperature: float
    max_tokens: int


class MLXProfiler:
    def __init__(self):
        self.mlx_manager = MLXManager()
        self.results: List[PerformanceMetrics] = []
        self.model_cache = {}

    async def initialize(self):
        """Initialize the profiler"""
        if not MLX_AVAILABLE:
            raise Exception("MLX not available. Please install mlx-lm package.")

        await self.mlx_manager.initialize()
        print("✅ MLX Profiler initialized")

    def profile_generation(
        self,
        prompt: str,
        model_name: str = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        runs: int = 3,
    ) -> List[PerformanceMetrics]:
        """Profile MLX generation with detailed metrics"""

        if model_name and model_name != self.mlx_manager.current_model_name:
            asyncio.run(self.mlx_manager.switch_model(model_name))

        results = []

        for run in range(runs):
            print(f"📊 Running profile {run + 1}/{runs}...")

            # Start monitoring
            tracemalloc.start()
            start_memory = psutil.virtual_memory().used / (1024**3)
            start_cpu = psutil.cpu_percent()

            # Time prompt processing vs generation
            total_start = time.perf_counter()

            # Measure prompt processing time (first forward pass)
            prompt_start = time.perf_counter()

            # Generate response with detailed timing
            try:
                # Use the existing MLX manager's generate function
                response = self.mlx_manager.generate_response(prompt, max_tokens)

                total_end = time.perf_counter()
                total_time = total_end - total_start

                # Estimate prompt processing vs generation time
                # This is approximate since MLX doesn't expose detailed timing
                prompt_processing_time = min(total_time * 0.1, 1.0)  # Estimated
                generation_time = total_time - prompt_processing_time

                # Calculate tokens per second
                output_tokens = len(response.split()) * 1.3  # Rough estimate
                tokens_per_second = (
                    output_tokens / generation_time if generation_time > 0 else 0
                )

                # Memory usage
                current_memory = psutil.virtual_memory().used / (1024**3)
                memory_delta = current_memory - start_memory

                # CPU usage
                end_cpu = psutil.cpu_percent()
                avg_cpu = (start_cpu + end_cpu) / 2

                # GPU utilization (estimated for Apple Silicon)
                gpu_util = self.mlx_manager.get_gpu_utilization()

                metrics = PerformanceMetrics(
                    tokens_per_second=tokens_per_second,
                    total_time=total_time,
                    prompt_processing_time=prompt_processing_time,
                    generation_time=generation_time,
                    memory_usage={
                        "used_gb": current_memory,
                        "delta_gb": memory_delta,
                        "peak_gb": tracemalloc.get_traced_memory()[1] / (1024**3),
                    },
                    gpu_utilization=gpu_util,
                    cpu_usage=avg_cpu,
                    model_name=self.mlx_manager.current_model_name,
                    prompt_length=len(prompt),
                    output_length=len(response),
                    batch_size=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                results.append(metrics)
                print(
                    f"  ⚡ {tokens_per_second:.1f} tokens/sec, {total_time:.2f}s total"
                )

            except Exception as e:
                print(f"❌ Error in run {run + 1}: {e}")

            finally:
                tracemalloc.stop()
                gc.collect()  # Force garbage collection between runs
                time.sleep(1)  # Cool down between runs

        self.results.extend(results)
        return results

    def benchmark_models(
        self, prompts: List[str], models: List[str]
    ) -> Dict[str, List[PerformanceMetrics]]:
        """Benchmark multiple models with multiple prompts"""
        results = {}

        for model in models:
            print(f"\n🔄 Benchmarking model: {model}")
            model_results = []

            try:
                asyncio.run(self.mlx_manager.switch_model(model))

                for i, prompt in enumerate(prompts):
                    print(f"  📝 Prompt {i + 1}/{len(prompts)}")
                    metrics = self.profile_generation(prompt, model, runs=2)
                    model_results.extend(metrics)

                results[model] = model_results

            except Exception as e:
                print(f"❌ Error benchmarking {model}: {e}")
                results[model] = []

        return results

    def test_optimization_flags(self, prompt: str) -> Dict[str, PerformanceMetrics]:
        """Test different MLX optimization configurations"""
        print("🧪 Testing MLX optimization configurations...")

        configurations = {
            "default": {},
            "high_temp": {"temperature": 0.9},
            "low_temp": {"temperature": 0.1},
            "max_tokens_50": {"max_tokens": 50},
            "max_tokens_200": {"max_tokens": 200},
        }

        results = {}

        for config_name, config in configurations.items():
            print(f"  Testing {config_name}...")

            max_tokens = config.get("max_tokens", 100)
            temperature = config.get("temperature", 0.7)

            metrics = self.profile_generation(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature, runs=2
            )

            if metrics:
                results[config_name] = metrics[0]  # Take first result

        return results

    def memory_stress_test(
        self, prompt: str, max_iterations: int = 10
    ) -> List[Dict[str, Any]]:
        """Test memory usage under continuous generation"""
        print("🧠 Running memory stress test...")

        results = []

        for i in range(max_iterations):
            start_memory = psutil.virtual_memory().used / (1024**3)

            # Generate response
            start_time = time.perf_counter()
            try:
                response = self.mlx_manager.generate_response(prompt, 100)
                end_time = time.perf_counter()

                end_memory = psutil.virtual_memory().used / (1024**3)
                memory_delta = end_memory - start_memory

                result = {
                    "iteration": i + 1,
                    "memory_before": start_memory,
                    "memory_after": end_memory,
                    "memory_delta": memory_delta,
                    "generation_time": end_time - start_time,
                    "response_length": len(response),
                }

                results.append(result)
                print(
                    f"  Iteration {i+1}: {memory_delta:+.2f}GB delta, {end_memory:.1f}GB total"
                )

                # Force cleanup
                del response
                gc.collect()

            except Exception as e:
                print(f"❌ Error in iteration {i+1}: {e}")
                break

        return results

    def analyze_bottlenecks(
        self, metrics_list: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Analyze performance metrics to identify bottlenecks"""
        if not metrics_list:
            return {"error": "No metrics to analyze"}

        # Calculate statistics
        tps_values = [m.tokens_per_second for m in metrics_list]
        total_times = [m.total_time for m in metrics_list]
        memory_usage = [m.memory_usage["used_gb"] for m in metrics_list]
        gpu_util = [m.gpu_utilization for m in metrics_list]

        analysis = {
            "performance_summary": {
                "avg_tokens_per_second": statistics.mean(tps_values),
                "max_tokens_per_second": max(tps_values),
                "min_tokens_per_second": min(tps_values),
                "std_tokens_per_second": (
                    statistics.stdev(tps_values) if len(tps_values) > 1 else 0
                ),
                "avg_total_time": statistics.mean(total_times),
                "avg_memory_usage": statistics.mean(memory_usage),
                "avg_gpu_utilization": statistics.mean(gpu_util),
            },
            "bottleneck_analysis": [],
            "optimization_recommendations": [],
        }

        # Identify bottlenecks
        avg_tps = analysis["performance_summary"]["avg_tokens_per_second"]
        avg_gpu = analysis["performance_summary"]["avg_gpu_utilization"]
        avg_memory = analysis["performance_summary"]["avg_memory_usage"]

        # GPU utilization bottlenecks
        if avg_gpu < 30:
            analysis["bottleneck_analysis"].append(
                {
                    "type": "Low GPU Utilization",
                    "severity": "High",
                    "value": f"{avg_gpu:.1f}%",
                    "description": "GPU is underutilized, may indicate CPU bottleneck or inefficient model loading",
                }
            )
            analysis["optimization_recommendations"].append(
                "Consider using smaller batch sizes or check for CPU-bound operations"
            )

        # Memory bottlenecks
        if avg_memory > 100:  # More than 100GB
            analysis["bottleneck_analysis"].append(
                {
                    "type": "High Memory Usage",
                    "severity": "Medium",
                    "value": f"{avg_memory:.1f}GB",
                    "description": "High memory usage may limit model size or batch size",
                }
            )
            analysis["optimization_recommendations"].append(
                "Consider using quantized models or reducing context length"
            )

        # Performance variability
        if len(tps_values) > 1:
            cv = statistics.stdev(tps_values) / statistics.mean(tps_values)
            if cv > 0.2:  # High coefficient of variation
                analysis["bottleneck_analysis"].append(
                    {
                        "type": "High Performance Variability",
                        "severity": "Medium",
                        "value": f"{cv:.2f}",
                        "description": "Inconsistent performance may indicate thermal throttling or resource contention",
                    }
                )
                analysis["optimization_recommendations"].append(
                    "Check for thermal throttling and ensure consistent system load"
                )

        # Low absolute performance
        if avg_tps < 10:
            analysis["bottleneck_analysis"].append(
                {
                    "type": "Low Tokens Per Second",
                    "severity": "High",
                    "value": f"{avg_tps:.1f} tokens/sec",
                    "description": "Overall low performance may indicate model size mismatch or optimization issues",
                }
            )
            analysis["optimization_recommendations"].append(
                "Consider using smaller models or enabling quantization"
            )

        return analysis

    def generate_report(self, output_path: str = "mlx_performance_report.json"):
        """Generate comprehensive performance report"""
        if not self.results:
            print("❌ No profiling results available")
            return

        analysis = self.analyze_bottlenecks(self.results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "python_version": sys.version,
            },
            "profiling_results": [
                {
                    "tokens_per_second": m.tokens_per_second,
                    "total_time": m.total_time,
                    "prompt_processing_time": m.prompt_processing_time,
                    "generation_time": m.generation_time,
                    "memory_usage": m.memory_usage,
                    "gpu_utilization": m.gpu_utilization,
                    "cpu_usage": m.cpu_usage,
                    "model_name": m.model_name,
                    "prompt_length": m.prompt_length,
                    "output_length": m.output_length,
                    "temperature": m.temperature,
                    "max_tokens": m.max_tokens,
                }
                for m in self.results
            ],
            "analysis": analysis,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📊 Performance report saved to {output_path}")
        return report

    def plot_performance(self, save_path: str = "mlx_performance_plots.png"):
        """Generate performance visualization plots"""
        if not self.results:
            print("❌ No results to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Tokens per second over time
        tps_values = [m.tokens_per_second for m in self.results]
        ax1.plot(range(len(tps_values)), tps_values, "b-o")
        ax1.set_title("Tokens Per Second Over Time")
        ax1.set_xlabel("Run Number")
        ax1.set_ylabel("Tokens/Second")
        ax1.grid(True)

        # Memory usage
        memory_values = [m.memory_usage["used_gb"] for m in self.results]
        ax2.plot(range(len(memory_values)), memory_values, "r-o")
        ax2.set_title("Memory Usage Over Time")
        ax2.set_xlabel("Run Number")
        ax2.set_ylabel("Memory (GB)")
        ax2.grid(True)

        # Performance distribution
        ax3.hist(tps_values, bins=10, alpha=0.7, color="green")
        ax3.set_title("Tokens Per Second Distribution")
        ax3.set_xlabel("Tokens/Second")
        ax3.set_ylabel("Frequency")
        ax3.grid(True)

        # GPU utilization vs Performance
        gpu_values = [m.gpu_utilization for m in self.results]
        ax4.scatter(gpu_values, tps_values, alpha=0.7, color="purple")
        ax4.set_title("GPU Utilization vs Performance")
        ax4.set_xlabel("GPU Utilization (%)")
        ax4.set_ylabel("Tokens/Second")
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📈 Performance plots saved to {save_path}")
        plt.close()


async def main():
    """Main profiling routine"""
    print("🚀 Starting MLX Performance Profiling")

    profiler = MLXProfiler()
    await profiler.initialize()

    # Test prompts of varying complexity
    test_prompts = [
        "Hello, how are you?",
        "Explain the concept of machine learning in simple terms.",
        "Write a detailed analysis of renewable energy technologies and their impact on global warming, including specific examples of solar, wind, and hydroelectric power implementations across different countries.",
    ]

    print("\n=== Single Model Profiling ===")

    # Profile current model with different prompts
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 Testing prompt {i+1}: {prompt[:50]}...")
        results = profiler.profile_generation(prompt, runs=3)

    print("\n=== Optimization Configuration Testing ===")

    # Test different optimization configurations
    config_results = profiler.test_optimization_flags(test_prompts[1])

    print("\n=== Memory Stress Testing ===")

    # Memory stress test
    memory_results = profiler.memory_stress_test(test_prompts[0], max_iterations=5)

    print("\n=== Generating Analysis and Report ===")

    # Generate comprehensive analysis
    analysis = profiler.analyze_bottlenecks(profiler.results)

    print("\n📊 Performance Analysis:")
    print(
        f"  Average Tokens/Second: {analysis['performance_summary']['avg_tokens_per_second']:.1f}"
    )
    print(
        f"  Average Memory Usage: {analysis['performance_summary']['avg_memory_usage']:.1f} GB"
    )
    print(
        f"  Average GPU Utilization: {analysis['performance_summary']['avg_gpu_utilization']:.1f}%"
    )

    if analysis["bottleneck_analysis"]:
        print("\n⚠️  Identified Bottlenecks:")
        for bottleneck in analysis["bottleneck_analysis"]:
            print(
                f"  - {bottleneck['type']}: {bottleneck['value']} ({bottleneck['severity']} severity)"
            )

    if analysis["optimization_recommendations"]:
        print("\n💡 Optimization Recommendations:")
        for rec in analysis["optimization_recommendations"]:
            print(f"  - {rec}")

    # Generate detailed report and plots
    report = profiler.generate_report()
    profiler.plot_performance()

    print("\n✅ Profiling complete! Check the generated report and plots.")


if __name__ == "__main__":
    asyncio.run(main())
