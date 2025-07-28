# MLX Performance Profiling and Optimization Suite

This directory contains comprehensive tools for profiling MLX generation performance and applying optimization techniques for Apple Silicon.

## Overview

The profiling suite provides:

1. **Performance Profiling** - Detailed analysis of MLX generation performance
2. **Bottleneck Detection** - Identifies performance bottlenecks and issues
3. **Optimization Guide** - Latest MLX optimization flags and techniques
4. **Hardware-Specific Recommendations** - Tailored suggestions for different Apple Silicon chips

## Files

### Core Components

- `mlx_profiler.py` - Comprehensive MLX performance profiler with detailed metrics
- `mlx_optimization_guide.py` - Complete optimization guide with latest MLX flags and techniques
- `../run_mlx_profiling.py` - Main profiling script with comprehensive analysis

### Key Features

#### MLX Profiler (`mlx_profiler.py`)
- **Detailed Performance Metrics**: Tokens/second, memory usage, GPU utilization
- **Bottleneck Analysis**: Identifies performance issues and provides recommendations
- **Memory Stress Testing**: Tests memory usage under continuous generation
- **Visualization**: Generates performance plots and charts
- **JSON Reports**: Detailed profiling reports with all metrics

#### Optimization Guide (`mlx_optimization_guide.py`)
- **Latest MLX Flags**: 2024-2025 optimization flags and parameters
- **Apple Silicon Optimizations**: Specific techniques for M1/M2/M3/M4 chips
- **Hardware-Specific Configs**: Optimized configurations for different hardware profiles
- **Model Recommendations**: Suggests best models for your hardware

## Usage

### Quick Start

```bash
# Run comprehensive profiling on current model
python run_mlx_profiling.py

# Quick profiling (fewer runs)
python run_mlx_profiling.py --quick

# Profile specific model
python run_mlx_profiling.py --model "mlx-community/Llama-3.1-8B-Instruct-4bit"

# Compare multiple models
python run_mlx_profiling.py --compare-models \
    "mlx-community/Llama-3.1-8B-Instruct-4bit" \
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

# Run only optimization benchmarks
python run_mlx_profiling.py --optimization-only
```

### Detailed Usage

#### 1. Basic Profiling
```python
from profiling.mlx_profiler import MLXProfiler

profiler = MLXProfiler()
await profiler.initialize()

# Profile a generation
results = profiler.profile_generation(
    prompt="Explain machine learning",
    runs=3
)

# Analyze bottlenecks
analysis = profiler.analyze_bottlenecks(profiler.results)
```

#### 2. Apply Optimizations
```python
from profiling.mlx_optimization_guide import MLXOptimizationGuide

optimizer = MLXOptimizationGuide()

# Get optimized config for your hardware
config = optimizer.get_optimized_generation_config(
    hardware_profile="m2_max",
    use_case="chat",
    memory_constraint="high"
)

# Apply optimizations
optimizer.apply_optimizations(mlx_manager, config)
```

## Key Optimization Techniques

### 1. Memory Optimizations
- **Unified Memory**: Automatic in MLX - CPU/GPU share memory
- **Quantization**: 4-bit/8-bit quantization for memory reduction
- **KV Cache**: Optimized attention caching for conversations

### 2. Computation Optimizations
- **Metal GPU Acceleration**: Full Apple Silicon GPU utilization
- **Lazy Evaluation**: Compute only when needed
- **Graph Optimization**: Optimized computation paths

### 3. Apple Silicon Specific
- **Neural Engine**: Dedicated AI processing unit utilization
- **Memory Bandwidth**: High-bandwidth memory optimization
- **Thermal Management**: Sustained performance without throttling

## Latest MLX Flags (2024-2025)

### Generation Flags
- `--temp`: Temperature for sampling (0.0-1.0)
- `--top-p`: Nucleus sampling parameter
- `--repetition-penalty`: Reduce repetition in output
- `--max-tokens`: Maximum tokens to generate
- `--seed`: Random seed for reproducibility

### Quantization Flags
- `--q`: Enable 4-bit quantization
- `--q-bits`: Quantization bits (4, 8, 16)
- `--q-group-size`: Group size for quantization (32, 64, 128)
- `--q-dataset`: Calibration dataset for quantization

### Memory Flags
- `--cache-limit-gb`: KV cache memory limit
- `--low-memory`: Enable low memory mode
- `--memory-map`: Memory map model weights

### Performance Flags
- `--verbose`: Enable verbose output for profiling
- `--timing`: Show detailed timing information
- `--profile`: Enable performance profiling

## Hardware Recommendations

### M1 (16-32GB)
- **Recommended Models**: Llama-3.1-8B-4bit, Mistral-7B-4bit
- **Optimal Config**: 4-bit quantization, temperature 0.7, max_tokens 200

### M2 Max (64-128GB)
- **Recommended Models**: Llama-3.1-70B-4bit, CodeLlama-34B
- **Optimal Config**: 6-8bit quantization, temperature 0.7, max_tokens 500

### M3 (192GB+)
- **Recommended Models**: Llama-3.1-405B-4bit, Large reasoning models
- **Optimal Config**: 8-bit quantization, temperature 0.7, max_tokens 800

## Output Files

After running profiling, you'll get:

1. **`mlx_performance_report.json`** - Detailed JSON report with all metrics
2. **`mlx_performance_plots.png`** - Performance visualization charts
3. **Console output** - Real-time analysis and recommendations

## Understanding Results

### Performance Metrics
- **Tokens/Second**: Primary performance indicator
- **Memory Usage**: RAM consumption during generation
- **GPU Utilization**: Apple Silicon GPU usage percentage
- **Response Time**: Total time for generation

### Bottleneck Analysis
- **Low GPU Utilization**: May indicate CPU bottleneck
- **High Memory Usage**: Consider quantized models
- **Performance Variability**: Check for thermal throttling
- **Low Absolute Performance**: Model size mismatch

### Optimization Recommendations
- Use quantized models (4-bit minimum)
- Leverage unified memory architecture
- Monitor thermal state for sustained performance
- Use appropriate temperature settings (0.3-0.7)
- Enable KV cache for conversations

## Advanced Usage

### Custom Quantization
```python
def custom_quantization_config(layer_path, layer, model_config):
    if "embed" in layer_path or "lm_head" in layer_path:
        return {"bits": 8, "group_size": 32}  # Higher precision
    elif "attention" in layer_path:
        return {"bits": 6, "group_size": 64}  # Balanced
    else:
        return {"bits": 4, "group_size": 128}  # Aggressive
```

### Thermal Management
```python
def monitor_thermal_state():
    result = subprocess.run(['pmset', '-g', 'thermstate'], capture_output=True, text=True)
    return 'CPU_Speed_Limit' not in result.stdout

def adaptive_generation(model, prompt, base_tokens=100):
    if not monitor_thermal_state():
        return generate(model, prompt, max_tokens=base_tokens // 2)
    else:
        return generate(model, prompt, max_tokens=base_tokens)
```

## Troubleshooting

### Common Issues

1. **MLX Not Available**
   ```bash
   pip install mlx-lm mlx
   ```

2. **Memory Errors**
   - Use quantized models
   - Reduce max_tokens
   - Enable low-memory mode

3. **Low Performance**
   - Check thermal throttling
   - Verify model quantization
   - Ensure Metal GPU acceleration

4. **Import Errors**
   ```bash
   pip install matplotlib psutil
   ```

## Contributing

To add new optimization techniques or profiling metrics:

1. Add to `MLXOptimizationGuide.optimization_techniques`
2. Update `MLXProfiler.profile_generation()` for new metrics
3. Add hardware-specific configurations
4. Update this README with new features

## See Also

- [MLX Documentation](https://github.com/ml-explore/mlx)
- [Apple Silicon Optimization Guide](https://developer.apple.com/documentation/metal/)
- [MLX Community Models](https://huggingface.co/mlx-community) 