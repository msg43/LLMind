#!/usr/bin/env python3
"""
MLX Performance Profiling and Optimization Suite
Comprehensive script to profile MLX generation and apply optimizations
"""
import asyncio
import sys
import argparse
import time
from pathlib import Path

# Add profiling directory to path
sys.path.append('profiling')

from profiling.mlx_profiler import MLXProfiler
from profiling.mlx_optimization_guide import MLXOptimizationGuide
from core.mlx_manager import MLXManager
import psutil

class MLXPerformanceSuite:
    """Complete MLX performance analysis and optimization suite"""
    
    def __init__(self):
        self.profiler = MLXProfiler()
        self.optimizer = MLXOptimizationGuide()
        self.mlx_manager = None
    
    async def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing MLX Performance Suite...")
        
        # Initialize profiler
        await self.profiler.initialize()
        self.mlx_manager = self.profiler.mlx_manager
        
        print("✅ MLX Performance Suite ready!")
    
    async def run_comprehensive_analysis(self, 
                                       model_name: str = None,
                                       quick_mode: bool = False):
        """Run comprehensive performance analysis"""
        
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE MLX PERFORMANCE ANALYSIS")
        print("="*80)
        
        # System info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        print(f"💻 System: {memory_gb:.0f}GB RAM, {cpu_count} CPU cores")
        
        if model_name:
            print(f"🤖 Switching to model: {model_name}")
            await self.mlx_manager.switch_model(model_name)
        
        current_model = self.mlx_manager.current_model_name
        print(f"📋 Current model: {current_model}")
        
        # Test prompts for different scenarios
        test_scenarios = {
            "simple": "Hello, how are you today?",
            "medium": "Explain the concept of machine learning and its applications in modern technology.",
            "complex": "Write a comprehensive analysis of renewable energy technologies, including solar, wind, hydroelectric, and geothermal power. Discuss their environmental impact, economic viability, technological challenges, and potential for global adoption in the next decade."
        }
        
        print(f"\n📝 Testing {len(test_scenarios)} scenarios...")
        
        # Run profiling for each scenario
        all_results = {}
        for scenario, prompt in test_scenarios.items():
            print(f"\n🧪 Testing {scenario} scenario...")
            runs = 2 if quick_mode else 3
            results = self.profiler.profile_generation(prompt, runs=runs)
            all_results[scenario] = results
            
            if results:
                avg_tps = sum(r.tokens_per_second for r in results) / len(results)
                avg_time = sum(r.total_time for r in results) / len(results)
                print(f"   ⚡ Average: {avg_tps:.1f} tokens/sec, {avg_time:.2f}s")
        
        # Test optimization configurations
        print(f"\n⚙️  Testing optimization configurations...")
        config_results = self.profiler.test_optimization_flags(test_scenarios["medium"])
        
        # Memory stress test
        if not quick_mode:
            print(f"\n🧠 Running memory stress test...")
            memory_results = self.profiler.memory_stress_test(
                test_scenarios["simple"], 
                max_iterations=5
            )
        
        # Analyze bottlenecks
        print(f"\n🔍 Analyzing performance bottlenecks...")
        analysis = self.profiler.analyze_bottlenecks(self.profiler.results)
        
        # Print analysis results
        self._print_analysis_summary(analysis)
        
        # Get optimization recommendations
        hardware_profile = self._detect_hardware_profile(memory_gb)
        recommendations = self.optimizer.get_model_recommendations(hardware_profile, memory_gb)
        
        print(f"\n🤖 Model Recommendations for {hardware_profile} ({memory_gb:.0f}GB):")
        for model in recommendations[:3]:  # Show top 3
            print(f"   • {model['model']} ({model['size']}) - {model['use_case']}")
        
        # Generate reports
        print(f"\n📊 Generating performance reports...")
        report = self.profiler.generate_report("mlx_performance_report.json")
        self.profiler.plot_performance("mlx_performance_plots.png")
        
        # Print optimization guide
        print(f"\n📖 Printing optimization guide...")
        self.optimizer.print_optimization_guide()
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "report_path": "mlx_performance_report.json",
            "plots_path": "mlx_performance_plots.png"
        }
    
    def _detect_hardware_profile(self, memory_gb: float) -> str:
        """Detect hardware profile based on system specs"""
        if memory_gb >= 512:
            return "m3_ultra"
        elif memory_gb >= 192:
            return "m3"
        elif memory_gb >= 64:
            return "m2_max"
        elif memory_gb >= 32:
            return "m2"
        else:
            return "m1"
    
    def _print_analysis_summary(self, analysis: dict):
        """Print analysis summary"""
        print(f"\n📊 PERFORMANCE ANALYSIS SUMMARY")
        print("-" * 40)
        
        summary = analysis.get("performance_summary", {})
        print(f"Average Performance: {summary.get('avg_tokens_per_second', 0):.1f} tokens/sec")
        print(f"Memory Usage: {summary.get('avg_memory_usage', 0):.1f} GB")
        print(f"GPU Utilization: {summary.get('avg_gpu_utilization', 0):.1f}%")
        
        # Bottlenecks
        bottlenecks = analysis.get("bottleneck_analysis", [])
        if bottlenecks:
            print(f"\n⚠️  Identified Bottlenecks:")
            for bottleneck in bottlenecks:
                severity_emoji = "🔴" if bottleneck['severity'] == "High" else "🟡"
                print(f"   {severity_emoji} {bottleneck['type']}: {bottleneck['value']}")
        
        # Recommendations
        recommendations = analysis.get("optimization_recommendations", [])
        if recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
    
    async def run_model_comparison(self, models: list):
        """Compare performance across multiple models"""
        print(f"\n🔬 MODEL COMPARISON ANALYSIS")
        print("-" * 50)
        
        test_prompt = "Explain the benefits of artificial intelligence in healthcare."
        comparison_results = {}
        
        for model in models:
            print(f"\n📋 Testing {model}...")
            try:
                await self.mlx_manager.switch_model(model)
                results = self.profiler.profile_generation(test_prompt, runs=2)
                
                if results:
                    avg_tps = sum(r.tokens_per_second for r in results) / len(results)
                    avg_memory = sum(r.memory_usage["used_gb"] for r in results) / len(results)
                    
                    comparison_results[model] = {
                        "tokens_per_second": avg_tps,
                        "memory_usage": avg_memory,
                        "results": results
                    }
                    
                    print(f"   ⚡ {avg_tps:.1f} tok/s, {avg_memory:.1f}GB memory")
                else:
                    print(f"   ❌ No results")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                comparison_results[model] = {"error": str(e)}
        
        # Print comparison summary
        print(f"\n📊 Model Comparison Summary:")
        print("-" * 50)
        for model, result in comparison_results.items():
            if "error" not in result:
                print(f"{model}:")
                print(f"   Performance: {result['tokens_per_second']:.1f} tok/s")
                print(f"   Memory: {result['memory_usage']:.1f} GB")
            else:
                print(f"{model}: Error - {result['error']}")
        
        return comparison_results
    
    async def run_optimization_benchmark(self):
        """Run optimization technique benchmarks"""
        print(f"\n⚙️  OPTIMIZATION TECHNIQUE BENCHMARKS")
        print("-" * 50)
        
        test_prompt = "Explain machine learning concepts in simple terms."
        
        # Get optimized config for current hardware
        memory_gb = psutil.virtual_memory().total / (1024**3)
        hardware_profile = self._detect_hardware_profile(memory_gb)
        
        config = self.optimizer.get_optimized_generation_config(
            hardware_profile, "chat", "high"
        )
        
        print(f"🎯 Optimal config for {hardware_profile}: {config}")
        
        # Apply optimizations and benchmark
        results = self.optimizer.benchmark_optimizations(self.mlx_manager, test_prompt)
        
        # Find best configuration
        best_config = None
        best_tps = 0
        
        for config_name, result in results.items():
            if "error" not in result and result["tokens_per_second"] > best_tps:
                best_tps = result["tokens_per_second"]
                best_config = config_name
        
        print(f"\n🏆 Best configuration: {best_config} ({best_tps:.1f} tok/s)")
        
        return results

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MLX Performance Profiling Suite")
    parser.add_argument("--model", help="Model to profile")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer runs)")
    parser.add_argument("--compare-models", nargs="+", help="Compare multiple models")
    parser.add_argument("--optimization-only", action="store_true", help="Run only optimization benchmarks")
    
    args = parser.parse_args()
    
    # Initialize suite
    suite = MLXPerformanceSuite()
    await suite.initialize()
    
    if args.optimization_only:
        # Run only optimization benchmarks
        await suite.run_optimization_benchmark()
        
    elif args.compare_models:
        # Run model comparison
        await suite.run_model_comparison(args.compare_models)
        
    else:
        # Run comprehensive analysis
        results = await suite.run_comprehensive_analysis(
            model_name=args.model,
            quick_mode=args.quick
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Report saved to: {results['report_path']}")
        print(f"📈 Plots saved to: {results['plots_path']}")
        print(f"\n💡 Check the generated files for detailed analysis and recommendations.")

if __name__ == "__main__":
    asyncio.run(main()) 