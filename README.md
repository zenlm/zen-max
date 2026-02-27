---
library_name: transformers
license: apache-2.0
tags:
- zen
- text-generation
- reasoning
- agentic
- tool-calling
- moe
- compressed-tensors
pipeline_tag: text-generation
---

# Zen Max

**Organization**: [Zen LM](https://zenlm.org) (Hanzo AI × Zoo Labs Foundation)
**Parameters**: 1.04T total (1,044B — MoE with 32B active per token)
**License**: Apache 2.0
**Context Window**: 256K tokens
**Thinking Capacity**: 96K-128K thinking tokens per step
**Architecture**: MoE (Mixture of Experts)

## Model Overview

Zen Max is the largest model in the Zen family — a 1T+ reasoning-first language model designed for **test-time scaling** through extended thinking and tool-calling capabilities.

Built as a **thinking agent**, Zen Max reasons step-by-step while using tools, executing **200-300 sequential tool calls** without human interference, reasoning coherently across hundreds of steps to solve complex problems.

### Key Capabilities

#### 1. Agentic Reasoning (HLE: 44.9%)
- Extended chain-of-thought reasoning with `<think>` tags
- Multi-step planning and execution
- Adaptive reasoning with hypothesis generation and refinement
- Think → search → code → verify → think cycles

#### 2. Agentic Search & Browsing (BrowseComp: 60.2%)
- Goal-directed web-based reasoning
- 200-300 sequential tool calls for information gathering
- Real-world information collection and synthesis
- Dynamic search → browser → reasoning loops

#### 3. Agentic Coding (SWE-Bench Verified: 71.3%)
- Multi-language support (100+ languages)
- Agentic coding workflows with tool integration
- Component-heavy web development (React, HTML)
- Terminal automation (Terminal-Bench: 47.1%)

#### 4. Mathematical Reasoning
- AIME 2025: 99.1% (with Python)
- HMMT 2025: 95.1% (with Python)
- IMO-AnswerBench: 78.6%
- GPQA-Diamond: 84.5%

### Architecture Features

#### Test-Time Scaling
- **Thinking Tokens**: 96K-128K per reasoning step
- **Extended Context**: 256K tokens
- **Sequential Tool Calls**: 200-300 without human intervention
- **Parallel Rollouts**: Heavy mode with 8 simultaneous trajectories

#### INT4 Quantization-Aware Training
- Native INT4 inference support
- 2x generation speed improvement
- State-of-the-art performance at INT4 precision
- Optimized for low-bit quantization during post-training

#### Inference Efficiency
- Quantization-aware training (QAT) for MoE components
- INT4 weight-only quantization
- ~50% latency reduction
- Minimal performance degradation

## Benchmark Performance

### Reasoning Tasks
| Benchmark | Score | Notes |
|-----------|-------|-------|
| HLE (with tools) | 44.9% | vs Human baseline 29.2% |
| AIME 2025 (with Python) | 99.1% | 75.2% without tools |
| HMMT 2025 (with Python) | 95.1% | 70.4% without tools |
| IMO-AnswerBench | 78.6% | Mathematical olympiad |
| GPQA-Diamond | 84.5% | Expert-level questions |

### Agentic Search
| Benchmark | Score | Notes |
|-----------|-------|-------|
| BrowseComp | 60.2% | vs Human 29.2% |
| BrowseComp-ZH | 62.3% | Chinese browsing |
| Seal-0 | 56.3% | Real-world info |
| FinSearchComp-T3 | 47.4% | Financial search |
| Frames | 87.0% | Multi-step search |

### Coding
| Benchmark | Score | Notes |
|-----------|-------|-------|
| SWE-Bench Verified | 71.3% | Software engineering |
| SWE-Multilingual | 61.1% | Multi-language coding |
| Multi-SWE-Bench | 41.9% | Multiple repositories |
| LiveCodeBench v6 | 83.1% | Competitive programming |
| Terminal-Bench | 47.1% | Shell automation |

### General Capabilities
| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU-Pro | 84.6% | Professional knowledge |
| MMLU-Redux | 94.4% | General knowledge |
| Longform Writing | 73.8% | Creative writing |
| HealthBench | 58.0% | Medical knowledge |

## Training Approach

### Architecture
- 1.04T parameter Mixture of Experts
- 32B active parameters per token
- Extended thinking token support
- Multi-modal reasoning capabilities

### Zen Identity Fine-Tuning
1. **Constitutional AI Training**: Hanzo AI principles and values
2. **Tool-Calling Specialization**: 200-300 step sequences
3. **Thinking Mode Optimization**: Extended reasoning patterns
4. **Multi-Agent Workflows**: Coordinated task execution

### Optimization
- INT4 quantization-aware training
- MoE component optimization
- Context management strategies
- Parallel trajectory aggregation (Heavy Mode)

## Usage Examples

### 1. Extended Reasoning with Tools
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-max")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-max")

# Enable thinking mode with tool access
messages = [
    {
        "role": "user",
        "content": "Research and analyze the latest developments in quantum computing, then write a comprehensive report."
    }
]

# Model will:
# 1. Think about search strategy
# 2. Execute 50+ web searches
# 3. Browse relevant pages
# 4. Synthesize information
# 5. Generate structured report
response = model.chat(tokenizer, messages, thinking_budget=128000, max_tool_calls=300)
```

### 2. Agentic Coding Workflow
```python
# Component-heavy web development
messages = [
    {
        "role": "user",
        "content": "Build a fully functional Word clone with React, including document editing, formatting, and export features."
    }
]

# Model will:
# 1. Plan component architecture
# 2. Generate HTML/React code
# 3. Implement styling and interactions
# 4. Test and debug iteratively
# 5. Deliver production-ready application
response = model.chat(tokenizer, messages, thinking_budget=96000, enable_tools=True)
```

### 3. Mathematical Problem Solving
```python
# PhD-level mathematics with Python
messages = [
    {
        "role": "user",
        "content": "Solve the hyperbolic space sampling problem involving Lorentz model and Brownian bridge covariance."
    }
]

# Model will:
# 1. Analyze mathematical structure
# 2. Execute Python computations
# 3. Derive closed-form solutions
# 4. Verify results numerically
response = model.chat(tokenizer, messages, thinking_budget=128000, python_enabled=True)
```

### 4. Heavy Mode (Parallel Reasoning)
```python
# 8 parallel trajectories with reflective aggregation
messages = [
    {
        "role": "user",
        "content": "Comprehensive analysis of climate change solutions across economics, technology, and policy."
    }
]

response = model.chat(
    tokenizer,
    messages,
    mode="heavy",  # 8 parallel rollouts
    thinking_budget=128000,
    enable_reflection=True
)
```

## Configuration

### Thinking Budget
- **Low**: 32K thinking tokens (fast responses)
- **Medium**: 96K thinking tokens (balanced)
- **High**: 128K thinking tokens (complex reasoning)
- **Heavy Mode**: 8 × 128K parallel trajectories

### Tool Configuration
```python
tools = {
    "search": True,          # Web search
    "browser": True,         # Page browsing
    "python": True,          # Code execution
    "bash": True,            # Shell commands
    "file_operations": True, # File I/O
}
```

### Context Management
- **Context Window**: 256K tokens
- **Auto-hiding**: Tool outputs hidden when exceeding context
- **Smart truncation**: Preserves reasoning chain and key results

## Hardware Requirements

### Inference (INT4 from HuggingFace)
- **Model Size**: ~370GB (62 safetensors shards, INT4 quantized)
- **Minimum**: 247GB combined RAM+VRAM+Disk
- **Optimal**: 370GB+ RAM+VRAM for 5+ tokens/s
- **Budget Setup**: 1x 24GB GPU + 256GB RAM (~1-2 tokens/s)
- **High Performance**: 4x A100 80GB or 8x A100 40GB

### Alternative: GGUF Quantizations
- **1.66-bit (UD-TQ1_0)**: 245GB - fits on 247GB combined RAM+VRAM
- **2.71-bit (UD-Q2_K_XL)**: 381GB - recommended for accuracy
- **4.5-bit (UD-Q4_K_XL)**: 588GB - near full precision

### QLoRA Training
- **VRAM**: ~500GB total (370GB model + 130GB activations)
- **GPUs**: 4x A100 80GB or 8x A100 40GB
- **Training Time**: 4-8 hours for 1000 steps
- **Output**: LoRA adapters (~100MB)

## Format Availability

### Current
- SafeTensors (BF16, full precision)
- INT4 Quantized (native QAT)

### Coming Soon
- GGUF quantizations (Q4_K_M, Q5_K_M, Q8_0)
- MLX optimized formats (4-bit, 8-bit for Apple Silicon)
- ONNX export for edge deployment

## Special Features

### 1. Thinking Mode
- Chain-of-thought reasoning with `<think>` tags
- Explicit reasoning traces
- Up to 128K thinking tokens per step
- Adaptive depth based on problem complexity

### 2. Tool-Calling Agent
- 200-300 sequential tool invocations
- No human intervention required
- Dynamic tool selection
- Error recovery and retry logic

### 3. Parallel Reasoning (Heavy Mode)
- 8 simultaneous reasoning trajectories
- Reflective aggregation of outputs
- Consensus-based answer selection
- 2-3x accuracy improvement on hard problems

### 4. Multi-Modal Extensions
- Vision-language understanding (future)
- Audio processing (future)
- Code → execution → analysis loops

## Limitations

1. **Thinking Token Overhead**: Extended reasoning increases latency
2. **Tool Call Limits**: 300 steps may not suffice for extremely complex tasks
3. **Context Management**: Auto-hiding may lose important intermediate results
4. **Quantization**: INT4 optimized, but BF16 still preferred for maximum accuracy

## Training Data

- **Zen Fine-Tuning**:
  - Zoo-Gym framework with RAIS technology
  - Constitutional AI alignment data
  - Multi-turn tool-calling trajectories
  - Agentic workflow demonstrations
- **Verification**: Human expert validation on HLE, AIME, coding tasks

## Citation

```bibtex
@misc{zenmax2025,
  title={Zen Max: Reasoning-First Language Model with Test-Time Scaling},
  author={Hanzo AI and Zoo Labs Foundation},
  year={2025},
  url={https://zenlm.org}
}
```

## Links

- **Website**: https://zenlm.org
- **API**: https://api.hanzo.ai/v1
- **HuggingFace**: https://huggingface.co/zenlm
- **GitHub**: https://github.com/zenlm

---

**Zen AI**: Clarity Through Intelligence
*Now with reasoning at test-time*
