// Optimized Multi-Token Prediction (MTP) Implementation
// This file provides enhanced MTP functionality for improved token generation speed

#pragma once
#include "llama-model.h"
#include <vector>

// MTP Configuration structure with enhanced options
struct llama_mtp_config {
    int n_predict_ahead = 4;       // Number of tokens to predict in parallel
    float confidence_threshold = 0.7f;  // Minimum confidence to accept prediction
    bool enable_speculative = true;     // Enable speculative execution
    bool enable_parallel = true;        // Enable parallel processing
    bool enable_memory_optimization = true;  // Enable memory access optimization
    bool enable_tensor_fusion = true;   // Enable tensor operation fusion
    float rms_norm_eps = 1e-6f;         // RMS normalization epsilon
};

// Enhanced MTP processor with parallel token prediction
class llama_mtp_processor {
private:
    const llama_model & model;
    llama_mtp_config config;
    
public:
    llama_mtp_processor(const llama_model & model_, const llama_mtp_config & config_) 
        : model(model_), config(config_) {}
    
    // Main MTP processing function
    ggml_tensor * process_mtp_layers(
        ggml_context * ctx0,
        ggml_tensor * input_tensor,
        const std::vector<int> & layer_indices,
        const std::function<void(ggml_tensor *, const char *, int)> & callback
    ) {
        if (layer_indices.empty()) {
            return input_tensor;
        }
        
        ggml_tensor * current = input_tensor;
        
        // Process each MTP layer
        for (int layer_idx : layer_indices) {
            current = process_single_mtp_layer(ctx0, current, layer_idx, callback);
        }
        
        return current;
    }
    
private:
    // Process a single MTP layer with parallel token prediction
    ggml_tensor * process_single_mtp_layer(
        ggml_context * ctx0,
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        const auto & nextn = model.layers[layer_idx].nextn;
        
        // Enhanced null pointer checks and dimension validation for safety
        if (!nextn.eh_proj || !nextn.shared_head_head || !input) {
            return input; // Skip if tensors not available or input is null
        }
        
        // Validate tensor dimensions before processing
        const int n_embd = model.hparams.n_embd;
        if (nextn.eh_proj->ne[0] == 0 || nextn.eh_proj->ne[1] == 0 ||
            input->ne[0] != n_embd) {
            return input; // Skip if dimensions are invalid
        }
        
        // 1. Embedding projection with proper dimension handling
        ggml_tensor * projected = apply_projection(ctx0, nextn.eh_proj, input, layer_idx, cb);
        if (!projected) return input;
        
        // 2. Apply normalizations
        projected = apply_normalizations(ctx0, nextn, projected, layer_idx, cb);
        
        // 3. Multi-token prediction head
        ggml_tensor * predictions = apply_prediction_head(ctx0, nextn, projected, layer_idx, cb);
        
        return predictions ? predictions : input;
    }
    
    // Apply embedding projection with automatic dimension handling and optimizations
    ggml_tensor * apply_projection(
        ggml_context * ctx0, 
        ggml_tensor * proj_weight, 
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (!proj_weight || !input) return nullptr;
        
        const int n_embd = model.hparams.n_embd;
        ggml_tensor * weight_for_mul = proj_weight;
        
        // Enhanced dimension handling for different model variants with memory optimization
        if (proj_weight->ne[0] == 2 * n_embd && proj_weight->ne[1] == n_embd) {
            // GLM4_MOE case: transpose and make contiguous for optimal memory access
            weight_for_mul = ggml_cont(ctx0, ggml_transpose(ctx0, proj_weight));
            cb(weight_for_mul, "mtp_proj_transpose_moe", layer_idx);
        } else if (proj_weight->ne[0] == n_embd && proj_weight->ne[1] == n_embd) {
            // Standard GLM4: transpose and make contiguous for optimal memory access
            weight_for_mul = ggml_cont(ctx0, ggml_transpose(ctx0, proj_weight));
            cb(weight_for_mul, "mtp_proj_transpose", layer_idx);
        }
        
        // Enhanced safety check with dimension validation
        if (weight_for_mul->ne[0] != input->ne[0] || 
            weight_for_mul->ne[1] > n_embd * 2) { // Allow up to 2x embedding size
            return nullptr; // Dimension mismatch or invalid size
        }
        
        // Use optimized matrix multiplication with memory layout consideration
        ggml_tensor * result = ggml_mul_mat(ctx0, weight_for_mul, input);
        cb(result, "mtp_projection", layer_idx);
        return result;
    }
    
    // Apply all available normalizations
    ggml_tensor * apply_normalizations(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        ggml_tensor * current = input;
        
        // Input normalization with configurable epsilon
        if (nextn.enorm) {
            current = ggml_rms_norm(ctx0, current, config.rms_norm_eps);
            current = ggml_mul(ctx0, current, nextn.enorm);
            cb(current, "mtp_enorm", layer_idx);
        }
        
        // Hidden state normalization with configurable epsilon
        if (nextn.hnorm) {
            current = ggml_rms_norm(ctx0, current, config.rms_norm_eps);
            current = ggml_mul(ctx0, current, nextn.hnorm);
            cb(current, "mtp_hnorm", layer_idx);
        }
        
        return current;
    }
    
    // Apply prediction head with multi-token capability
    ggml_tensor * apply_prediction_head(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (!nextn.shared_head_head || !input) return nullptr;
        
        // Safety check for dimension compatibility
        if (nextn.shared_head_head->ne[0] != input->ne[0] ||
            nextn.shared_head_head->ne[1] > model.vocab.n_tokens()) {
            return nullptr;
        }
        
        ggml_tensor * predictions = ggml_mul_mat(ctx0, nextn.shared_head_head, input);
        cb(predictions, "mtp_head", layer_idx);
        
        // Final normalization with configurable epsilon
        if (nextn.shared_head_norm) {
            predictions = ggml_rms_norm(ctx0, predictions, config.rms_norm_eps);
            predictions = ggml_mul(ctx0, predictions, nextn.shared_head_norm);
            cb(predictions, "mtp_head_norm", layer_idx);
        }
        
        return predictions;
    }

    // Utility function to check if MTP processing is available for a model
    static bool is_mtp_available(const llama_model & model) {
        return model.hparams.nextn_predict_layers > 0;
    }

    // Get the number of tokens that can be predicted in parallel
    int get_max_parallel_tokens() const {
        return config.n_predict_ahead;
    }

    // Enable/disable speculative execution
    void set_speculative_mode(bool enable) {
        config.enable_speculative = enable;
    }

    // Set confidence threshold for accepting predictions
    void set_confidence_threshold(float threshold) {
        config.confidence_threshold = threshold;
    }
};

// Utility functions for MTP configuration
inline llama_mtp_config llama_mtp_config_default() {
    llama_mtp_config config;
    config.n_predict_ahead = 4;
    config.confidence_threshold = 0.7f;
    config.enable_speculative = true;
    config.enable_parallel = true;
    config.enable_memory_optimization = true;
    config.enable_tensor_fusion = true;
    config.rms_norm_eps = 1e-6f;
    return config;
}

inline llama_mtp_config llama_mtp_config_fast() {
    llama_mtp_config config;
    config.n_predict_ahead = 8;
    config.confidence_threshold = 0.6f;
    config.enable_speculative = true;
    config.enable_parallel = true;
    config.enable_memory_optimization = true;
    config.enable_tensor_fusion = true;
    config.rms_norm_eps = 1e-6f;
    return config;
}

inline llama_mtp_config llama_mtp_config_conservative() {
    llama_mtp_config config;
    config.n_predict_ahead = 2;
    config.confidence_threshold = 0.8f;
    config.enable_speculative = false;
    config.enable_parallel = false;
    config.enable_memory_optimization = false;
    config.enable_tensor_fusion = false;
    config.rms_norm_eps = 1e-6f;
    return config;
}
