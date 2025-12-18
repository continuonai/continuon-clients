# Small Models & Datasets for Multi-Source Learning

This document catalogs small models and datasets available or recommended for training Continuon Brain from diverse sources (VLA, VLM, LM, embeddings, Wikipedia, physics).

## Currently Downloaded Models

### Language Models (LM)
- **google/gemma-2-2b-it** - 2B parameter instruction-tuned Gemma
- **google/gemma-3-270m-it** - 270M parameter Gemma 3 (ultra-small)
- **google/gemma-3n-E2B-it** - Gemma 3 Nano E2B instruction-tuned
- **google/gemma-3n-E2B-it-litert-lm** - LiteRT quantized version
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - 1.1B parameter chat model

### Currently Downloaded Datasets
- **HuggingFaceVLA/community_dataset_v3** - Vision-Language-Action robot trajectories
- **philschmid/amazon-product-descriptions-vlm** - Product image/text pairs
- **rahular/simple-wikipedia** - Simplified English Wikipedia

## Recommended Small Models by Category

### Vision-Language Models (VLM)

#### Ultra-Small (< 500M params)
- **google/gemma-3-270m-it** ✅ (already downloaded)
- **microsoft/Phi-3-mini-4k-instruct** - 3.8B but efficient, supports vision via extensions
- **Qwen/Qwen2-VL-2B-Instruct** - 2B vision-language model
- **Salesforce/blip-image-captioning-base** - 247M image captioning

#### Small (500M - 2B params)
- **google/gemma-2-2b-it** ✅ (already downloaded)
- **liuhaotian/LLaVA-1.5-7B** - Popular VLM (7B, but can be quantized)
- **microsoft/kosmos-2-patch14-224** - Multimodal foundation model

### Vision-Language-Action (VLA)

#### Robot-Specific
- **HuggingFaceVLA/community_dataset_v3** ✅ (already downloaded)
- **Open-X-Embodiment/bridge_data_v2** - Large robot demo dataset (subsample for Pi)
- **robotican/robotican-dataset** - Household manipulation tasks

#### General VLA
- **allenai/robocook** - Cooking manipulation dataset
- **stanford-oval/robocat** - Cat manipulation dataset

### Language Models (LM) - Small & Efficient

#### Ultra-Small (< 1B)
- **google/gemma-3-270m-it** ✅ (already downloaded)
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** ✅ (already downloaded)
- **microsoft/Phi-3-mini-4k-instruct** - 3.8B but very efficient
- **Qwen/Qwen2-0.5B-Instruct** - 0.5B instruction-tuned
- **stabilityai/stablelm-2-1_6b** - 1.6B base model

#### Small (1B - 3B)
- **google/gemma-2-2b-it** ✅ (already downloaded)
- **google/gemma-3n-2b-it** - 2B Gemma 3 Nano
- **mistralai/Mistral-7B-Instruct-v0.2** - 7B but can be quantized to <2GB
- **meta-llama/Llama-3.2-1B-Instruct** - 1B Llama 3.2

### Text Embeddings

#### Sentence Embeddings (Small)
- **sentence-transformers/all-MiniLM-L6-v2** - 80M, 384-dim, fast
- **sentence-transformers/all-mpnet-base-v2** - 110M, 768-dim, better quality
- **intfloat/e5-small-v2** - 118M, 384-dim, multilingual
- **BAAI/bge-small-en-v1.5** - 33M, 384-dim, very fast
- **thenlper/gte-small** - 33M, 384-dim, efficient

#### Multilingual Embeddings
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** - 118M
- **intfloat/multilingual-e5-small** - 118M, 384-dim

### Wikipedia & Knowledge

#### English Wikipedia
- **rahular/simple-wikipedia** ✅ (already downloaded)
- **wikimedia/wikipedia** - Full English Wikipedia (filter by domain)
- **huggingface/wikipedia** - Processed Wikipedia dumps
- **wikimedia/simple_wikipedia** - Simple English Wikipedia

#### Domain-Specific Knowledge
- **scientific_papers** - Physics/ML papers
- **c4** - Colossal Clean Crawled Corpus (filtered subset)

### Physics Models

#### Physics-Informed Models
- **microsoft/phi-2** - Strong physics reasoning (2.7B) ✅ RECOMMENDED
- **microsoft/phi-3-mini-4k-instruct** - Good physics understanding (3.8B)
- **deepmind/physics-simulation** - Physics simulation models (if available)

#### Physics Datasets
- **scientific_papers** - Physics/ML papers corpus
- **physics-datasets/mechanics** - Classical mechanics problems
- **physics-datasets/quantum** - Quantum mechanics datasets
- **physics-datasets/thermodynamics** - Thermodynamics problems

#### Physics-Informed Neural Networks (PINNs)
- Look for models trained on PDEs (partial differential equations)
- Search HuggingFace for "physics-informed" or "PINN"

### Specialized Small Models

#### Code Models
- **bigcode/starcoder2-3b** - 3B code generation
- **Salesforce/codegen-350M-mono** - 350M code model
- **WizardLM/WizardCoder-3B-V1.1** - 3B instruction-tuned coder

#### Math Models
- **microsoft/phi-2** - 2.7B, strong math reasoning, excellent for Pi 5 ✅ RECOMMENDED
- **microsoft/phi-3-mini-4k-instruct** - 3.8B, efficient, strong math/code
- **huggingface/SmolLM-1.7B-Instruct** - 1.7B, trained on FineMath dataset
- **huggingface/SmolLM2-135M-Instruct** - 135M ultra-small, math-capable
- **meta-llama/Llama-3.1-8B-Instruct** - 8B, good at math (can be quantized to ~4GB)
- **WizardLM/WizardMath-7B-V1.1** - 7B, specialized for math (quantize for Pi)

## Tools & Scripts

### HuggingFace CLI Tools
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# List downloaded models
huggingface-cli scan-cache

# Download a model
huggingface-cli download MODEL_ID --local-dir ./models/MODEL_ID

# Download dataset
huggingface-cli download DATASET_ID --repo-type dataset --local-dir ./datasets/DATASET_ID
```

### Model Verification Script
Located at: `verify_hf_models.py` - Checks which models are in local cache

### Model Download Script
Located at: `download_model.py` - Downloads Gemma models with token support

## Recommended Download List for Pi 5

### Priority 1: Essential Small Models
```bash
# Ultra-small language models
huggingface-cli download google/gemma-3-270m-it
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Small embeddings
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
huggingface-cli download BAAI/bge-small-en-v1.5

# Vision-language
huggingface-cli download Salesforce/blip-image-captioning-base
```

### Priority 2: Domain-Specific
```bash
# Wikipedia (already have simple-wikipedia)
huggingface-cli download wikimedia/wikipedia --repo-type dataset

# VLA datasets
huggingface-cli download Open-X-Embodiment/bridge_data_v2 --repo-type dataset

# Chat datasets
huggingface-cli download HuggingFaceH4/ultrachat_200k --repo-type dataset
```

### Priority 3: Specialized
```bash
# Code models
huggingface-cli download bigcode/starcoder2-3b

# Math models
huggingface-cli download microsoft/phi-2
```

## Integration with Training UI

The training page (`/training`) already references these datasets in the "Seed Model Loop" section:
- **VLM**: `liuhaotian/LLaVA-Instruct-150K`
- **VLA**: `Open-X-Embodiment/bridge_data_v2`
- **LM**: `HuggingFaceH4/ultrachat_200k`

## Usage in Training Pipeline

### For JAX Training
Models with JAX/Flax weights are preferred:
- `google/gemma-3n-E2B-it` (already downloaded)
- Check for Flax weights: `huggingface-cli repo-info MODEL_ID`

### For PyTorch Training
All models with PyTorch weights work:
- Use `transformers` library to load
- Quantize with `bitsandbytes` for Pi 5 memory constraints

### For Embeddings
Use sentence-transformers for fast inference:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(['text to embed'])
```

## Memory Considerations (Pi 5)

- **8GB Pi 5**: Use models < 2B params, quantized to 4-bit
- **16GB Pi 5**: Can handle 2-3B params, 8-bit quantization
- **Embeddings**: All listed embedding models fit easily (< 500MB)

## Next Steps

1. **Download priority models** using HuggingFace CLI
2. **Update training configs** to reference new models
3. **Add model selection** to Chat Learn Tests UI
4. **Create embedding service** for Wikipedia/knowledge retrieval
5. **Integrate physics models** for world model training

## References

- [HuggingFace Models](https://huggingface.co/models)
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Sentence Transformers](https://www.sbert.net/)
- [Model Cards](https://huggingface.co/docs/hub/model-cards)
