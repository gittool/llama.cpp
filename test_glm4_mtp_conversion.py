#!/usr/bin/env python3
"""
Test script to convert GLM-4.5-Air model with MTP (Multi-Token Prediction) support.

Usage:
    python test_glm4_mtp_conversion.py [model_path] [output_path]
"""

import sys
import argparse
from pathlib import Path

def test_conversion():
    parser = argparse.ArgumentParser(description="Convert GLM-4.5-Air model with MTP support")
    parser.add_argument("model_path", type=str, help="Path to the HuggingFace model directory")
    parser.add_argument("output_path", type=str, help="Output path for GGUF file")
    parser.add_argument("--type", choices=["f16", "f32", "q8_0"], default="f16", help="Quantization type")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        return 1
    
    # Check if this is a GLM-4 model with MTP support
    config_file = model_path / "config.json"
    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        model_type = config.get('model_type', 'unknown')
        num_nextn_predict_layers = config.get('num_nextn_predict_layers', 0)
        
        print(f"Model type: {model_type}")
        print(f"NextN/MTP layers: {num_nextn_predict_layers}")
        
        if model_type in ['glm', 'chatglm']:
            print("✓ Detected GLM model")
        else:
            print(f"Warning: Model type '{model_type}' may not fully support MTP features")
        
        if num_nextn_predict_layers > 0:
            print(f"✓ Model has {num_nextn_predict_layers} MTP layers")
        else:
            print("Note: Model does not have MTP layers")
    
    # Build conversion command
    cmd = [
        sys.executable,
        "convert_hf_to_gguf.py",
        str(model_path),
        "--outdir", str(output_path.parent),
        "--outfile", output_path.name,
        "--type", args.type
    ]
    
    print(f"Running conversion command:")
    print(" ".join(cmd))
    
    # Execute conversion
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Conversion completed successfully!")
        print(result.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        return 1

if __name__ == "__main__":
    exit_code = test_conversion()
    sys.exit(exit_code)
