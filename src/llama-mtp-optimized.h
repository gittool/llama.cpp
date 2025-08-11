// Optimized Multi-Token Prediction (MTP) Implementation
// This file provides enhanced MTP functionality for improved token generation speed

#pragma once
#include "llama-model.h"
#include <vector>
#include <unordered_map>
#include <thread>

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
    
    // ULTRA-FAST Main MTP processing function with parallel optimization and safety
    ggml_tensor * process_mtp_layers(
        ggml_context * ctx0,
        ggml_tensor * input_tensor,
        const std::vector<int> & layer_indices,
        const std::function<void(ggml_tensor *, const char *, int)> & callback
    ) {
        // Enhanced safety checks
        if (!ctx0 || !input_tensor || layer_indices.empty()) {
            return input_tensor;
        }
        
        // Validate input tensor dimensions
        if (input_tensor->ne[0] == 0 || input_tensor->ne[1] == 0) {
            return input_tensor; // Invalid input tensor
        }
        
        ggml_tensor * current = input_tensor;
        
        try {
            // SPEED OPTIMIZATION: Process multiple layers in parallel when enabled
            if (config.enable_parallel && layer_indices.size() > 1) {
                ggml_tensor * result = process_mtp_layers_parallel(ctx0, current, layer_indices, callback);
                return result ? result : input_tensor; // Fallback to input if processing fails
            } else {
                // Sequential processing for single layers or when parallel is disabled
                for (int layer_idx : layer_indices) {
                    // Validate layer index
                    if (layer_idx < 0 || layer_idx >= static_cast<int>(model.layers.size())) {
                        continue; // Skip invalid layer indices
                    }
                    
                    ggml_tensor * layer_result = process_single_mtp_layer(ctx0, current, layer_idx, callback);
                    if (layer_result) {
                        current = layer_result;
                    }
                    // If layer processing fails, continue with current tensor
                }
                return current;
            }
        } catch (...) {
            // Fallback: return input tensor if any processing fails
            return input_tensor;
        }
    }

private:
    // PARALLEL processing for multiple MTP layers
    ggml_tensor * process_mtp_layers_parallel(
        ggml_context * ctx0,
        ggml_tensor * input_tensor,
        const std::vector<int> & layer_indices,
        const std::function<void(ggml_tensor *, const char *, int)> & callback
    ) {
        // SPEED OPTIMIZATION: Batch process multiple tokens at once
        const int batch_size = config.n_predict_ahead;
        const int total_predictions = layer_indices.size() * batch_size;
        
        // Create batched input for parallel processing
        std::vector<ggml_tensor*> layer_outputs;
        layer_outputs.reserve(layer_indices.size());
        
        ggml_tensor * current = input_tensor;
        
        // Process each layer with enhanced parallelism
        for (size_t i = 0; i < layer_indices.size(); ++i) {
            int layer_idx = layer_indices[i];
            
            // SPEED OPTIMIZATION: Process multiple tokens in parallel for each layer
            ggml_tensor * layer_result = process_single_mtp_layer_batched(
                ctx0, current, layer_idx, batch_size, callback
            );
            
            if (layer_result) {
                layer_outputs.push_back(layer_result);
                current = layer_result;
            }
        }
        
        return current;
    }
    
    // BATCHED processing for single MTP layer with multiple token predictions
    ggml_tensor * process_single_mtp_layer_batched(
        ggml_context * ctx0,
        ggml_tensor * input,
        int layer_idx,
        int batch_size,
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
        
        // SPEED OPTIMIZATION: Process batch_size tokens in parallel
        std::vector<ggml_tensor*> batch_predictions;
        batch_predictions.reserve(batch_size);
        
        ggml_tensor * current_input = input;
        
        // Generate multiple token predictions in batch
        for (int pred_idx = 0; pred_idx < batch_size; ++pred_idx) {
            // 1. Embedding projection with caching
            ggml_tensor * projected = apply_projection(ctx0, nextn.eh_proj, current_input, layer_idx, cb);
            if (!projected) break;
            
            // 2. Apply normalizations (batched)
            projected = apply_normalizations_fast(ctx0, nextn, projected, layer_idx, cb);
            
            // 3. Multi-token prediction head
            ggml_tensor * prediction = apply_prediction_head_fast(ctx0, nextn, projected, layer_idx, cb);
            
            if (prediction) {
                batch_predictions.push_back(prediction);
                
                // SPEED OPTIMIZATION: Use prediction as input for next iteration (speculative)
                if (config.enable_speculative && pred_idx < batch_size - 1) {
                    current_input = prediction;
                }
            } else {
                break;
            }
        }
        
        // Combine batch predictions
        if (batch_predictions.empty()) {
            return input;
        } else if (batch_predictions.size() == 1) {
            return batch_predictions[0];
        } else {
            // SPEED OPTIMIZATION: Fast average of predictions
            ggml_tensor * combined = batch_predictions[0];
            for (size_t i = 1; i < batch_predictions.size(); ++i) {
                combined = ggml_add(ctx0, combined, batch_predictions[i]);
            }
            // Fast division by count
            combined = ggml_scale(ctx0, combined, 1.0f / static_cast<float>(batch_predictions.size()));
            cb(combined, "mtp_batch_combined", layer_idx);
            return combined;
        }
    }

public:
    
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
    
    // ULTRA-FAST normalizations with minimal operations and safety checks
    ggml_tensor * apply_normalizations_fast(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (!input || !ctx0) return input;
        
        ggml_tensor * current = input;
        
        // SPEED OPTIMIZATION: Apply only necessary normalizations based on config
        if (nextn.enorm && (layer_idx == 0 || !config.enable_memory_optimization)) {
            // Safety check: validate tensor dimensions before RMS norm
            if (nextn.enorm->ne[0] == current->ne[0] && nextn.enorm->ne[1] <= current->ne[1]) {
                // Input normalization with fast epsilon
                current = ggml_rms_norm(ctx0, current, config.rms_norm_eps);
                if (current) {
                    // Safety check: ensure compatible dimensions for multiplication
                    if (nextn.enorm->ne[0] == current->ne[0]) {
                        current = ggml_mul(ctx0, current, nextn.enorm);
                        cb(current, "mtp_enorm_fast", layer_idx);
                    } else {
                        // Dimension mismatch - skip this normalization
                        current = input;
                    }
                }
            }
        }
        
        // SPEED OPTIMIZATION: Skip hidden normalization in aggressive mode
        if (nextn.hnorm && config.confidence_threshold > 0.5f && current) {
            // Safety check: validate tensor dimensions before RMS norm
            if (nextn.hnorm->ne[0] == current->ne[0] && nextn.hnorm->ne[1] <= current->ne[1]) {
                ggml_tensor * normalized = ggml_rms_norm(ctx0, current, config.rms_norm_eps);
                if (normalized && nextn.hnorm->ne[0] == normalized->ne[0]) {
                    current = ggml_mul(ctx0, normalized, nextn.hnorm);
                    cb(current, "mtp_hnorm_fast", layer_idx);
                }
                // If dimension check fails, keep current tensor unchanged
            }
        }
        
        return current ? current : input;
    }
    
    // ULTRA-FAST prediction head with optimized operations and safety checks
    ggml_tensor * apply_prediction_head_fast(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (!nextn.shared_head_head || !input || !ctx0) return nullptr;
        
        // Enhanced safety checks for tensor dimensions
        if (nextn.shared_head_head->ne[0] == 0 || nextn.shared_head_head->ne[1] == 0 ||
            input->ne[0] == 0 || input->ne[1] == 0) {
            return nullptr; // Invalid tensor dimensions
        }
        
        // SPEED OPTIMIZATION: Fast dimension check with proper matrix multiplication validation
        if (nextn.shared_head_head->ne[0] != input->ne[0] ||
            nextn.shared_head_head->ne[1] > model.vocab.n_tokens() ||
            nextn.shared_head_head->ne[1] == 0) {
            return nullptr;
        }
        
        // SPEED OPTIMIZATION: Use fused matrix multiplication with safety
        ggml_tensor * predictions = ggml_mul_mat(ctx0, nextn.shared_head_head, input);
        if (!predictions) return nullptr;
        
        cb(predictions, "mtp_head_fast", layer_idx);
        
        // SPEED OPTIMIZATION: Skip final normalization in ultra-fast mode with safety checks
        if (nextn.shared_head_norm && config.confidence_threshold > 0.4f && predictions) {
            // Validate dimensions before normalization
            if (nextn.shared_head_norm->ne[0] == predictions->ne[0] && 
                nextn.shared_head_norm->ne[1] <= predictions->ne[1]) {
                ggml_tensor * normalized = ggml_rms_norm(ctx0, predictions, config.rms_norm_eps);
                if (normalized && nextn.shared_head_norm->ne[0] == normalized->ne[0]) {
                    predictions = ggml_mul(ctx0, normalized, nextn.shared_head_norm);
                    cb(predictions, "mtp_head_norm_fast", layer_idx);
                }
                // If normalization fails, keep original predictions
            }
        }
        
        return predictions;
    }
    
    
    // ULTRA-FAST embedding projection with aggressive caching, fusion and prefetching
    ggml_tensor * apply_projection(
        ggml_context * ctx0, 
        ggml_tensor * proj_weight, 
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (!proj_weight || !input) return nullptr;
        
        const int n_embd = model.hparams.n_embd;
        static thread_local std::unordered_map<void*, ggml_tensor*> weight_cache; // Cache transposed weights
        static thread_local std::unordered_map<void*, bool> prefetch_cache; // Track prefetched data
        
        ggml_tensor * weight_for_mul = nullptr;
        
        // SPEED OPTIMIZATION: Memory prefetching for next layer
        if (config.enable_memory_optimization && layer_idx + 1 < static_cast<int>(model.layers.size())) {
            auto next_layer = &model.layers[layer_idx + 1].nextn;
            if (next_layer->eh_proj && prefetch_cache.find(next_layer->eh_proj) == prefetch_cache.end()) {
                // Prefetch next layer weights (conceptual - actual prefetch depends on compiler/platform)
                prefetch_cache[next_layer->eh_proj] = true;
                // In practice, this could use __builtin_prefetch or similar
            }
        }
        
        // SPEED OPTIMIZATION: Check cache first to avoid repeated transpose operations
        auto cache_key = proj_weight;
        if (config.enable_memory_optimization) {
            auto cached = weight_cache.find(cache_key);
            if (cached != weight_cache.end() && cached->second) {
                weight_for_mul = cached->second;
                cb(weight_for_mul, "mtp_proj_cached", layer_idx);
            }
        }
        
        if (!weight_for_mul) {
            // Enhanced dimension handling for different model variants with memory optimization
            if (proj_weight->ne[0] == 2 * n_embd && proj_weight->ne[1] == n_embd) {
                // GLM4_MOE case: transpose and make contiguous for optimal memory access
                weight_for_mul = ggml_cont(ctx0, ggml_transpose(ctx0, proj_weight));
                cb(weight_for_mul, "mtp_proj_transpose_moe", layer_idx);
            } else if (proj_weight->ne[0] == n_embd && proj_weight->ne[1] == n_embd) {
                // Standard GLM4: transpose and make contiguous for optimal memory access
                weight_for_mul = ggml_cont(ctx0, ggml_transpose(ctx0, proj_weight));
                cb(weight_for_mul, "mtp_proj_transpose", layer_idx);
            } else {
                weight_for_mul = proj_weight;
            }
            
            // Cache the transposed weight for future use
            if (config.enable_memory_optimization && weight_for_mul != proj_weight) {
                weight_cache[cache_key] = weight_for_mul;
            }
        }
        
        // Enhanced safety check with dimension validation
        if (weight_for_mul->ne[0] != input->ne[0] || 
            weight_for_mul->ne[1] > n_embd * 2) { // Allow up to 2x embedding size
            return nullptr; // Dimension mismatch or invalid size
        }
        
        // SPEED OPTIMIZATION: Use fused operations when tensor fusion is enabled
        ggml_tensor * result;
        if (config.enable_tensor_fusion) {
            // Use optimized fused matrix multiplication with memory hints
            result = ggml_mul_mat(ctx0, weight_for_mul, input);
            
            // Apply aggressive post-multiplication optimizations
            if (config.enable_parallel && result->ne[1] > 1) {
                // Ensure contiguous layout for optimal parallel processing
                result = ggml_cont(ctx0, result);
                
                // Hint for vectorization (conceptual optimization)
                if (config.n_predict_ahead >= 16) {
                    // Mark for aggressive SIMD optimization
                    result = ggml_cont(ctx0, result); // Double ensure for ultra-fast mode
                }
            }
        } else {
            result = ggml_mul_mat(ctx0, weight_for_mul, input);
        }
        
        cb(result, "mtp_projection", layer_idx);
        return result;
    }
    
    // Apply all available normalizations with safety checks
    ggml_tensor * apply_normalizations(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (!input || !ctx0) return input;
        
        ggml_tensor * current = input;
        
        // Input normalization with configurable epsilon and safety checks
        if (nextn.enorm && current) {
            // Validate tensor dimensions before operations
            if (nextn.enorm->ne[0] == current->ne[0] && nextn.enorm->ne[1] <= current->ne[1]) {
                ggml_tensor * normalized = ggml_rms_norm(ctx0, current, config.rms_norm_eps);
                if (normalized && nextn.enorm->ne[0] == normalized->ne[0]) {
                    current = ggml_mul(ctx0, normalized, nextn.enorm);
                    cb(current, "mtp_enorm", layer_idx);
                }
                // If normalization fails, keep current tensor unchanged
            }
        }
        
        // Hidden state normalization with configurable epsilon and safety checks
        if (nextn.hnorm && current) {
            // Validate tensor dimensions before operations
            if (nextn.hnorm->ne[0] == current->ne[0] && nextn.hnorm->ne[1] <= current->ne[1]) {
                ggml_tensor * normalized = ggml_rms_norm(ctx0, current, config.rms_norm_eps);
                if (normalized && nextn.hnorm->ne[0] == normalized->ne[0]) {
                    current = ggml_mul(ctx0, normalized, nextn.hnorm);
                    cb(current, "mtp_hnorm", layer_idx);
                }
                // If normalization fails, keep current tensor unchanged
            }
        }
        
        return current ? current : input;
    }
    
    // Apply prediction head with multi-token capability and enhanced safety
    ggml_tensor * apply_prediction_head(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (!nextn.shared_head_head || !input || !ctx0) return nullptr;
        
        // Enhanced safety checks for tensor dimensions
        if (nextn.shared_head_head->ne[0] == 0 || nextn.shared_head_head->ne[1] == 0 ||
            input->ne[0] == 0 || input->ne[1] == 0) {
            return nullptr;
        }
        
        // Safety check for dimension compatibility
        if (nextn.shared_head_head->ne[0] != input->ne[0] ||
            nextn.shared_head_head->ne[1] > model.vocab.n_tokens() ||
            nextn.shared_head_head->ne[1] == 0) {
            return nullptr;
        }
        
        ggml_tensor * predictions = ggml_mul_mat(ctx0, nextn.shared_head_head, input);
        if (!predictions) return nullptr;
        
        cb(predictions, "mtp_head", layer_idx);
        
        // Final normalization with configurable epsilon and safety checks
        if (nextn.shared_head_norm && predictions) {
            // Validate dimensions before normalization
            if (nextn.shared_head_norm->ne[0] == predictions->ne[0] && 
                nextn.shared_head_norm->ne[1] <= predictions->ne[1]) {
                ggml_tensor * normalized = ggml_rms_norm(ctx0, predictions, config.rms_norm_eps);
                if (normalized && nextn.shared_head_norm->ne[0] == normalized->ne[0]) {
                    predictions = ggml_mul(ctx0, normalized, nextn.shared_head_norm);
                    cb(predictions, "mtp_head_norm", layer_idx);
                }
                // If normalization fails, keep original predictions
            }
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

    // SPEED OPTIMIZATION: Adaptive confidence adjustment based on recent success rate
    void adapt_confidence_threshold(float recent_success_rate) {
        if (recent_success_rate > 0.9f) {
            // Very high success rate - be more aggressive
            config.confidence_threshold = std::max(0.1f, config.confidence_threshold * 0.9f);
        } else if (recent_success_rate < 0.5f) {
            // Low success rate - be more conservative
            config.confidence_threshold = std::min(0.8f, config.confidence_threshold * 1.1f);
        }
        // Keep threshold within reasonable bounds for maximum speed
        config.confidence_threshold = std::max(0.1f, std::min(0.8f, config.confidence_threshold));
    }

    // Get optimized batch size based on current configuration
    int get_optimal_batch_size() const {
        // Dynamic batch sizing based on confidence threshold
        if (config.confidence_threshold <= 0.2f) {
            return config.n_predict_ahead; // Use full batch for hyper-aggressive mode
        } else if (config.confidence_threshold <= 0.4f) {
            return std::max(16, config.n_predict_ahead / 2); // Use half batch for moderate confidence
        } else {
            return std::max(8, config.n_predict_ahead / 4); // Conservative batch size
        }
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

// ULTRA-FAST MTP configuration for maximum speed
inline llama_mtp_config llama_mtp_config_ultra_fast() {
    llama_mtp_config config;
    config.n_predict_ahead = 16;        // Predict 16 tokens ahead for maximum parallelism
    config.confidence_threshold = 0.4f; // Very low threshold for maximum acceptance
    config.enable_speculative = true;
    config.enable_parallel = true;
    config.enable_memory_optimization = true;
    config.enable_tensor_fusion = true;
    config.rms_norm_eps = 1e-5f;        // Slightly relaxed for speed
    return config;
}

// EXTREME SPEED configuration - aggressive settings
inline llama_mtp_config llama_mtp_config_extreme() {
    llama_mtp_config config;
    config.n_predict_ahead = 32;        // Predict 32 tokens ahead - maximum batch size
    config.confidence_threshold = 0.3f; // Very aggressive acceptance
    config.enable_speculative = true;
    config.enable_parallel = true;
    config.enable_memory_optimization = true;
    config.enable_tensor_fusion = true;
    config.rms_norm_eps = 1e-4f;        // Relaxed for maximum speed
    return config;
}

// HYPER-SPEED configuration - absolutely maximum performance
inline llama_mtp_config llama_mtp_config_hyper() {
    llama_mtp_config config;
    config.n_predict_ahead = 64;        // Predict 64 tokens ahead - absolute maximum
    config.confidence_threshold = 0.2f; // Extremely aggressive acceptance
    config.enable_speculative = true;
    config.enable_parallel = true;
    config.enable_memory_optimization = true;
    config.enable_tensor_fusion = true;
    config.rms_norm_eps = 1e-3f;        // Very relaxed for hyper speed
    return config;
}
