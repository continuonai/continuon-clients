# Math & Physics Models for Continuon Brain

## ✅ Currently Downloaded

### Math Models
- **microsoft/phi-2** (2.7B parameters) - Strong math reasoning, excellent for Pi 5
  - Size: ~5.4GB (can be quantized to ~1.4GB with 4-bit)
  - Strengths: Mathematical reasoning, code generation, logical problem solving
  - Use case: Physics calculations, world model training, symbolic reasoning

## 🔍 Recommended Math Models

### Small Math Models (< 3B)
1. **microsoft/phi-2** ✅ **DOWNLOADED**
   - 2.7B parameters
   - Excellent math and physics reasoning
   - Good for Pi 5 (with quantization)
   - HuggingFace: https://huggingface.co/microsoft/phi-2

2. **microsoft/phi-3-mini-4k-instruct**
   - 3.8B parameters
   - More efficient than Phi-2
   - Strong math/code performance
   - HuggingFace: https://huggingface.co/microsoft/phi-3-mini-4k-instruct

3. **huggingface/SmolLM-1.7B-Instruct**
   - 1.7B parameters
   - Trained on FineMath dataset
   - Optimized for local devices
   - HuggingFace: https://huggingface.co/huggingface/SmolLM-1.7B-Instruct

4. **huggingface/SmolLM2-135M-Instruct**
   - 135M parameters (ultra-small)
   - Math-capable despite small size
   - Perfect for resource-constrained devices
   - HuggingFace: https://huggingface.co/huggingface/SmolLM2-135M-Instruct

### Medium Math Models (3B - 8B)
5. **meta-llama/Llama-3.1-8B-Instruct**
   - 8B parameters
   - Good math performance
   - Can be quantized to ~4GB for Pi 5
   - Requires quantization for Pi 5 8GB

6. **WizardLM/WizardMath-7B-V1.1**
   - 7B parameters
   - Specialized for math problems
   - Requires quantization for Pi 5

## 🔬 Physics Models

### General Physics Reasoning
- **microsoft/phi-2** ✅ **DOWNLOADED** - Strong physics reasoning
- **microsoft/phi-3-mini-4k-instruct** - Good physics understanding

### Physics-Informed Neural Networks (PINNs)
Search HuggingFace for:
- "physics-informed" models
- "PINN" (Physics-Informed Neural Networks)
- Models trained on PDEs (partial differential equations)

### Physics Datasets
- **scientific_papers** - Physics/ML papers corpus
- **physics-datasets/mechanics** - Classical mechanics problems
- **physics-datasets/quantum** - Quantum mechanics datasets
- **physics-datasets/thermodynamics** - Thermodynamics problems

## 📥 How to Download

### Using the Discovery Script
```bash
# Download Phi-2 (already done)
python3 scripts/discover_and_download_models.py --download microsoft/phi-2

# Download Phi-3 Mini
python3 scripts/discover_and_download_models.py --download microsoft/phi-3-mini-4k-instruct

# Download SmolLM
python3 scripts/discover_and_download_models.py --download huggingface/SmolLM-1.7B-Instruct
```

### Using HuggingFace CLI
```bash
# Install if needed
pip install huggingface_hub

# Download model
huggingface-cli download microsoft/phi-2

# Download with token (if gated)
huggingface-cli download microsoft/phi-2 --token YOUR_TOKEN
```

## 🎯 Use Cases for Continuon Brain

### 1. World Model Training
- Use Phi-2 for physics reasoning in world model
- Train on physics equations and physical laws
- Integrate with HOPE architecture for symbolic reasoning

### 2. Physics Simulation
- Use math models to predict object trajectories
- Calculate forces, velocities, and collisions
- Enhance robot's understanding of physical interactions

### 3. Symbolic Reasoning
- Replace curve-fitting with symbolic search
- Use math models for planning and problem-solving
- Enable "System 2 thinking" as mentioned in README

### 4. Training Data Generation
- Generate synthetic physics scenarios
- Create training data for world model
- Augment RLDS episodes with physics knowledge

## 💾 Memory Considerations (Pi 5)

### 8GB Pi 5
- **Phi-2**: Requires 4-bit quantization (~1.4GB)
- **SmolLM-1.7B**: Can run with 4-bit quantization (~900MB)
- **SmolLM2-135M**: Can run natively (~270MB)

### 16GB Pi 5
- **Phi-2**: Can run with 8-bit quantization (~2.7GB)
- **Phi-3-mini**: Can run with 4-bit quantization (~1.9GB)
- **SmolLM-1.7B**: Can run with 8-bit quantization (~1.7GB)

## 🔧 Integration with Training Pipeline

### For JAX Training
- Check if models have JAX/Flax weights
- Phi-2: PyTorch only (use transformers)
- SmolLM: Check for JAX weights

### For PyTorch Training
- All models work with transformers library
- Use `bitsandbytes` for quantization
- Load with: `AutoModelForCausalLM.from_pretrained()`

### Example Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Phi-2
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # or use quantization
    device_map="auto"
)

# Use for physics reasoning
prompt = "Calculate the force needed to move a 5kg object at 2 m/s²"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## 📊 Model Comparison

| Model | Size | Math Score | Physics | Pi 5 Compatible |
|-------|------|------------|---------|-----------------|
| Phi-2 | 2.7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ (quantized) |
| Phi-3-mini | 3.8B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ (quantized) |
| SmolLM-1.7B | 1.7B | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ (quantized) |
| SmolLM2-135M | 135M | ⭐⭐⭐ | ⭐⭐ | ✅ (native) |
| Llama-3.1-8B | 8B | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⚠️ (needs 16GB) |

## 🚀 Next Steps

1. ✅ **Downloaded Phi-2** - Ready to use
2. **Test Phi-2** with physics problems
3. **Integrate into training pipeline** for world model
4. **Download SmolLM-1.7B** for smaller footprint option
5. **Create physics reasoning module** using Phi-2
6. **Add to Chat Learn Tests** UI for testing

## 📚 References

- [Phi-2 Model Card](https://huggingface.co/microsoft/phi-2)
- [Phi-3-mini Model Card](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)
- [SmolLM Models](https://huggingface.co/huggingface/SmolLM-1.7B-Instruct)
- [HuggingFace Math Models](https://huggingface.co/models?other=math)
