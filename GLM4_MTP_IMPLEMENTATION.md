# GLM-4 Multi-Token Prediction (MTP) Support Implementation

This document summarizes the changes made to support Multi-Token Prediction (MTP) layers in GLM-4 models.

## Changes Made

### 1. gguf-py/gguf/constants.py
- Added NextN/MTP tensor definitions to the GLM4 model architecture
- Added the following tensors to `MODEL_ARCH.GLM4`:
  - `MODEL_TENSOR.NEXTN_EH_PROJ`
  - `MODEL_TENSOR.NEXTN_EMBED_TOKENS`
  - `MODEL_TENSOR.NEXTN_ENORM`
  - `MODEL_TENSOR.NEXTN_HNORM`
  - `MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD`
  - `MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM`

### 2. convert_hf_to_gguf.py - Glm4Model class
Enhanced the standard GLM-4 model class to support MTP layers:
- Added `__init__` method to adjust block count when MTP layers are present
- Enhanced `set_gguf_parameters()` to write NextN prediction layer count to GGUF
- Enhanced `modify_tensors()` to handle NextN/MTP layer tensors

### 3. convert_hf_to_gguf.py - Glm4MoeModel class
Enhanced the GLM-4 MoE model class to better handle MTP tensors:
- Improved `modify_tensors()` to properly map NextN layer tensor names

## Model Support

The implementation now supports:
- GLM-4 models with MTP layers (Glm4ForCausalLM)
- GLM-4 MoE models with MTP layers (Glm4MoeForCausalLM)
- GLM-4v multimodal models (Glm4vForConditionalGeneration)

The GLM-4.5-Air model hash is already configured:
- Hash: `9ca2dd618e8afaf09731a7cf6e2105b373ba6a1821559f258b272fe83e6eb902`
- Tokenizer: `glm4` (BPE-based)
- Repository: https://huggingface.co/zai-org/GLM-4.5-Air

## Key Features

### Multi-Token Prediction Support
- Handles `num_nextn_predict_layers` parameter from model configuration
- Correctly maps NextN layer tensors to GGUF format
- Supports various MTP components:
  - Embedding projection (`eh_proj`)
  - Token embeddings (`embed_tokens`)
  - Input normalization (`enorm`)
  - Hidden normalization (`hnorm`)
  - Shared head components (`shared_head_head`, `shared_head_norm`)

### Tensor Name Mapping
The implementation correctly maps HuggingFace tensor names to GGUF format:
- `model.layers.{bid}.nextn.eh_proj.weight` → `blk.{bid}.nextn.eh_proj.weight`
- `model.layers.{bid}.nextn.embed_tokens.weight` → `blk.{bid}.nextn.embed_tokens.weight`
- And similarly for other NextN components

## Usage

To convert a GLM-4 model with MTP layers:

```bash
python convert_hf_to_gguf.py /path/to/glm4-model --outfile glm4-model.gguf
```

Use the test script for validation:
```bash
python test_glm4_mtp_conversion.py /path/to/glm4-model output.gguf --type f16
```

## Technical Notes

- The block count is automatically adjusted when MTP layers are detected
- Both standard GLM-4 and GLM-4 MoE models are supported
- Visual components in GLM-4v models are ignored during conversion
- The implementation preserves backward compatibility with models without MTP layers
