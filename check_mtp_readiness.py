#!/usr/bin/env python3
"""
MTP Implementation Readiness Checker
Verifies that all MTP components are properly implemented before building.
"""

import os
import sys
import re

def check_file_contains(file_path, patterns, description):
    """Check if file contains required patterns"""
    if not os.path.exists(file_path):
        print(f"❌ {description}: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    missing_patterns = []
    for pattern, desc in patterns:
        if isinstance(pattern, str):
            if pattern not in content:
                missing_patterns.append(desc)
        else:  # regex pattern
            if not pattern.search(content):
                missing_patterns.append(desc)
    
    if missing_patterns:
        print(f"❌ {description}: Missing {', '.join(missing_patterns)}")
        return False
    else:
        print(f"✅ {description}: All required components present")
        return True

def main():
    print("🔍 MTP Implementation Readiness Check")
    print("="*50)
    
    all_good = True
    
    # Check 1: GGUF Conversion
    conversion_patterns = [
        ("TENSOR_SKIP", "TENSOR_SKIP should not be present (MTP tensors should be enabled)"),
        (re.compile(r'nextn.*predict.*layers'), "NextN predict layers handling"),
        (re.compile(r'eh_proj.*embed_tokens'), "MTP tensor types"),
    ]
    
    if not check_file_contains("convert_hf_to_gguf.py", 
                              [(re.compile(r'(?<!TENSOR_SKIP.*)(nextn|shared_head)'), "MTP tensor processing")],
                              "GGUF Conversion"):
        all_good = False
    
    # Check 2: Model Loading
    model_patterns = [
        (re.compile(r'llm_load_tensor.*nextn'), "MTP tensor loading"),
        (re.compile(r'model\.layers.*nextn.*predict'), "MTP layer structure"),
    ]
    
    if not check_file_contains("src/llama.cpp", model_patterns, "Model Loading"):
        all_good = False
    
    # Check 3: Forward Pass
    forward_patterns = [
        (re.compile(r'GLM_4.*build_forward'), "GLM4 forward pass"),
        (re.compile(r'nextn.*predict.*layer'), "MTP layer processing"),
    ]
    
    if not check_file_contains("src/llama-model.cpp", forward_patterns, "Forward Pass"):
        all_good = False
    
    # Check 4: API Support
    api_patterns = [
        ("llama_model_has_mtp_support", "MTP support detection"),
        ("llama_predict_mtp_tokens", "MTP prediction API"),
        (re.compile(r'LLAMA_API.*mtp'), "MTP API declarations"),
    ]
    
    if not check_file_contains("include/llama.h", api_patterns, "API Headers"):
        all_good = False
    
    # Check 5: Sampling Integration
    sampling_patterns = [
        ("common_sampler_sample_mtp", "MTP sampling function"),
        ("common_sampler_can_use_mtp", "MTP capability check"),
    ]
    
    if not check_file_contains("common/sampling.cpp", sampling_patterns, "Sampling"):
        all_good = False
    
    # Check 6: Build System
    build_patterns = [
        ("mtp-benchmark", "MTP benchmark tool"),
    ]
    
    if not check_file_contains("examples/CMakeLists.txt", build_patterns, "Build System"):
        all_good = False
    
    print("\n" + "="*50)
    
    if all_good:
        print("🎉 MTP IMPLEMENTATION READY!")
        print()
        print("Next steps:")
        print("1. mkdir build && cd build")
        print("2. cmake ..")
        print("3. make -j$(nproc)")
        print("4. ./examples/mtp-benchmark/mtp-benchmark --help")
        print()
        print("For GLM4 model testing:")
        print("python3 test_glm4_mtp_conversion.py path/to/glm4-model")
    else:
        print("❌ MTP IMPLEMENTATION INCOMPLETE")
        print("Please review the missing components above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
