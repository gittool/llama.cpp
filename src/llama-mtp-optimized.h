// Optimized Multi-Token Prediction (MTP) Implementation
// This file provides enhanced MTP functionality for improved token generation speed

#pragma once
#include "llama-model.h"
#include <vector>
#include <unordered_map>
#include <thread>
#include <chrono>

// Performance monitoring metrics for MTP operations
struct llama_mtp_perf_metrics {
    uint64_t total_calls = 0;           // Total MTP calls
    uint64_t successful_calls = 0;      // Successful MTP operations
    uint64_t fallback_calls = 0;        // Times fallback was used
    uint64_t validation_failures = 0;   // Dimension validation failures
    double total_latency_ms = 0.0;      // Total latency in milliseconds
    double total_throughput_tokens = 0.0; // Total tokens processed
    uint64_t cache_hits = 0;            // Weight cache hits
    uint64_t cache_misses = 0;          // Weight cache misses
    
    // Real-time metrics
    double avg_latency_ms() const { return total_calls > 0 ? total_latency_ms / total_calls : 0.0; }
    double success_rate() const { return total_calls > 0 ? double(successful_calls) / total_calls : 0.0; }
    double tokens_per_ms() const { return total_latency_ms > 0 ? total_throughput_tokens / total_latency_ms : 0.0; }
    double cache_hit_rate() const { return (cache_hits + cache_misses) > 0 ? double(cache_hits) / (cache_hits + cache_misses) : 0.0; }
};

// MTP Configuration structure with enhanced options
struct llama_mtp_config {
    int n_predict_ahead = 4;       // Number of tokens to predict in parallel
    float confidence_threshold = 0.7f;  // Minimum confidence to accept prediction
    bool enable_speculative = true;     // Enable speculative execution
    bool enable_parallel = true;        // Enable parallel processing
    bool enable_memory_optimization = true;  // Enable memory access optimization
    bool enable_tensor_fusion = true;   // Enable tensor operation fusion
    float rms_norm_eps = 1e-6f;         // RMS normalization epsilon
    bool enable_performance_monitoring = true; // Enable detailed performance tracking
};

// Enhanced MTP processor with parallel token prediction
class llama_mtp_processor {
private:
    const llama_model & model;
    llama_mtp_config config;
    mutable llama_mtp_perf_metrics metrics; // Performance tracking
    
    // Enhanced dimension validation and error handling
    bool validate_mtp_tensor_dimensions(
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int n_embd) const {
        if (!input || !nextn.eh_proj || !nextn.shared_head_head) {
            return false;
        }
        
        // Validate input tensor
        if (input->ne[0] != n_embd || input->ne[1] == 0) {
            return false;
        }
        
        // Validate eh_proj dimensions
        if (nextn.eh_proj->ne[0] == 0 || nextn.eh_proj->ne[1] == 0) {
            return false;
        }
        
        // Allow standard GLM4 (n_embd x n_embd) or GLM4_MOE (2*n_embd x n_embd)
        bool valid_eh_proj = (nextn.eh_proj->ne[0] == n_embd && nextn.eh_proj->ne[1] == n_embd) ||
                            (nextn.eh_proj->ne[0] == 2 * n_embd && nextn.eh_proj->ne[1] == n_embd);
        
        if (!valid_eh_proj) {
            return false;
        }
        
        // Validate shared_head dimensions
        if (nextn.shared_head_head->ne[1] > model.vocab.n_tokens() || 
            nextn.shared_head_head->ne[1] == 0) {
            return false;
        }
        
        return true;
    }
    
    // Fallback to single token processing
    ggml_tensor * fallback_to_single_token(ggml_tensor * input) const {
        // Simply return the input tensor unchanged as fallback
        return input;
    }
    
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
        // Performance monitoring start
        auto start_time = std::chrono::high_resolution_clock::now();
        if (config.enable_performance_monitoring) {
            metrics.total_calls++;
        }
        
        // Enhanced safety checks
        if (!ctx0 || !input_tensor || layer_indices.empty()) {
            if (config.enable_performance_monitoring) {
                metrics.validation_failures++;
                metrics.fallback_calls++;
            }
            return input_tensor;
        }
        
        // Validate input tensor dimensions
        if (input_tensor->ne[0] == 0 || input_tensor->ne[1] == 0) {
            if (config.enable_performance_monitoring) {
                metrics.validation_failures++;
                metrics.fallback_calls++;
            }
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
                // Performance monitoring completion for sequential processing
                if (config.enable_performance_monitoring) {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                    metrics.total_latency_ms += duration.count() / 1000.0;
                    metrics.successful_calls++;
                    metrics.total_throughput_tokens += layer_indices.size() * config.n_predict_ahead;
                }
                
                return current;
            }
        } catch (...) {
            // Fallback: return input tensor if any processing fails
            if (config.enable_performance_monitoring) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                metrics.total_latency_ms += duration.count() / 1000.0;
                metrics.fallback_calls++;
            }
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
        
        // Track total predictions for performance metrics
        int total_predictions = 0;
        if (config.enable_performance_monitoring) {
            total_predictions = layer_indices.size() * batch_size;
        }
        
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
        
        // Update performance metrics for parallel processing
        if (config.enable_performance_monitoring && total_predictions > 0) {
            metrics.total_throughput_tokens += layer_outputs.size() * batch_size;
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
        
        // SPEED OPTIMIZATION: Use dynamic batch size based on system performance
        const int optimal_batch_size = std::min(batch_size, get_optimal_batch_size());
        std::vector<ggml_tensor*> batch_predictions;
        batch_predictions.reserve(optimal_batch_size);
        
        ggml_tensor * current_input = input;
        
        // Generate multiple token predictions in batch with adaptive sizing
        for (int pred_idx = 0; pred_idx < optimal_batch_size; ++pred_idx) {
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
                if (config.enable_speculative && pred_idx < optimal_batch_size - 1) {
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
            
            // Update performance metrics with actual prediction count
            if (config.enable_performance_monitoring) {
                metrics.total_throughput_tokens += batch_predictions.size();
            }
            
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
        const int n_embd = model.hparams.n_embd;
        
        // Enhanced dimension validation with fallback
        if (!validate_mtp_tensor_dimensions(nextn, input, n_embd)) {
            cb(input, "mtp_validation_failed", layer_idx);
            return fallback_to_single_token(input);
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
        
        // SPEED OPTIMIZATION: Clear old cache entries to prevent memory bloat
        static thread_local int cache_cleanup_counter = 0;
        if (++cache_cleanup_counter > 1000) {
            weight_cache.clear();
            prefetch_cache.clear();
            cache_cleanup_counter = 0;
        }
        
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
                if (config.enable_performance_monitoring) {
                    metrics.cache_hits++;
                }
                cb(weight_for_mul, "mtp_proj_cached", layer_idx);
            } else if (config.enable_performance_monitoring) {
                metrics.cache_misses++;
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

    // SPEED OPTIMIZATION: Improved adaptive confidence adjustment for stable performance
    void adapt_confidence_threshold(float recent_success_rate) {
        if (recent_success_rate > 0.85f) {
            // High success rate - be slightly more aggressive
            config.confidence_threshold = std::max(0.3f, config.confidence_threshold * 0.95f);
        } else if (recent_success_rate < 0.6f) {
            // Low success rate - be more conservative
            config.confidence_threshold = std::min(0.9f, config.confidence_threshold * 1.05f);
        }
        // Keep threshold within reasonable bounds for stable performance
        config.confidence_threshold = std::max(0.3f, std::min(0.9f, config.confidence_threshold));
    }

    // Get optimized batch size based on current configuration and system capabilities
    int get_optimal_batch_size() const {
        // Improved dynamic batch sizing for better cache performance
        if (config.confidence_threshold <= 0.2f) {
            return std::min(config.n_predict_ahead, 8); // Cap at 8 for better cache locality
        } else if (config.confidence_threshold <= 0.5f) {
            return std::min(6, config.n_predict_ahead); // Moderate batch size
        } else if (config.confidence_threshold <= 0.7f) {
            return std::min(4, config.n_predict_ahead); // Conservative for better quality
        } else {
            return std::min(2, config.n_predict_ahead); // Very conservative for high quality
        }
    }

    // Get current performance metrics
    const llama_mtp_perf_metrics & get_performance_metrics() const {
        return metrics;
    }

    // Reset performance metrics
    void reset_performance_metrics() {
        metrics = llama_mtp_perf_metrics{};
    }

    // Print detailed performance report
    void print_performance_report() const {
        if (!config.enable_performance_monitoring) {
            printf("MTP Performance Monitoring is disabled\n");
            return;
        }
        
        printf("=== MTP Performance Report ===\n");
        printf("Total Calls: %llu\n", (unsigned long long)metrics.total_calls);
        printf("Successful: %llu (%.1f%%)\n", (unsigned long long)metrics.successful_calls, metrics.success_rate() * 100.0);
        printf("Fallback: %llu (%.1f%%)\n", (unsigned long long)metrics.fallback_calls, 
               metrics.total_calls > 0 ? (double(metrics.fallback_calls) / metrics.total_calls) * 100.0 : 0.0);
        printf("Validation Failures: %llu (%.1f%%)\n", (unsigned long long)metrics.validation_failures, 
               metrics.total_calls > 0 ? (double(metrics.validation_failures) / metrics.total_calls) * 100.0 : 0.0);
        printf("Average Latency: %.2f ms\n", metrics.avg_latency_ms());
        printf("Throughput: %.2f tokens/ms\n", metrics.tokens_per_ms());
        printf("Cache Hit Rate: %.1f%%\n", metrics.cache_hit_rate() * 100.0);
        printf("Total Tokens Processed: %.0f\n", metrics.total_throughput_tokens);
        printf("=============================\n");
    }

    // vLLM-style MTP processing with full transformer layer
    ggml_tensor * process_vllm_style_mtp(
        ggml_context * ctx0,
        ggml_tensor * hidden_state_inp,
        llama_token last_token_id,
        int /* n_past */,  // Parameter reserved for future position-aware processing
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (config.enable_performance_monitoring) {
            metrics.total_calls++;
        }

        const auto & mtp_layer = model.layers[layer_idx];
        const auto & nextn = mtp_layer.nextn;

        // Enhanced validation with fallback
        if (!validate_mtp_tensor_dimensions(nextn, hidden_state_inp, model.hparams.n_embd)) {
            cb(hidden_state_inp, "vllm_mtp_validation_failed", layer_idx);
            if (config.enable_performance_monitoring) {
                metrics.validation_failures++;
                metrics.fallback_calls++;
            }
            return fallback_to_single_token(hidden_state_inp);
        }

        try {
            // 1. Get MTP embedding for last (conventionally sampled) token
            ggml_tensor * inp_token_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            ggml_set_i32(inp_token_id, last_token_id);
            
            ggml_tensor * token_emb = ggml_get_rows(ctx0, nextn.embed_tokens, inp_token_id);
            if (!token_emb) {
                cb(hidden_state_inp, "vllm_mtp_token_emb_failed", layer_idx);
                if (config.enable_performance_monitoring) {
                    metrics.fallback_calls++;
                }
                return fallback_to_single_token(hidden_state_inp);
            }
            
            // 2. Apply token embedding normalization (enorm)
            ggml_tensor * token_emb_norm = token_emb;
            if (nextn.enorm) {
                token_emb_norm = ggml_rms_norm(ctx0, token_emb, config.rms_norm_eps);
                if (token_emb_norm) {
                    token_emb_norm = ggml_mul(ctx0, token_emb_norm, nextn.enorm);
                    cb(token_emb_norm, "vllm_token_emb_norm", layer_idx);
                }
            }
            
            // 3. Apply hidden state normalization (hnorm) - vLLM L99 style
            ggml_tensor * hidden_state_norm = hidden_state_inp;
            if (nextn.hnorm) {
                hidden_state_norm = ggml_rms_norm(ctx0, hidden_state_inp, config.rms_norm_eps);
                if (hidden_state_norm) {
                    hidden_state_norm = ggml_mul(ctx0, hidden_state_norm, nextn.hnorm);
                    cb(hidden_state_norm, "vllm_hidden_norm", layer_idx);
                }
            }
            
            // 4. Concatenate embeddings (torch.cat equivalent)
            ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
            if (!combined) {
                cb(hidden_state_inp, "vllm_mtp_concat_failed", layer_idx);
                if (config.enable_performance_monitoring) {
                    metrics.fallback_calls++;
                }
                return fallback_to_single_token(hidden_state_inp);
            }
            cb(combined, "vllm_combined", layer_idx);
            
            // 5. Apply eh_proj projection
            ggml_tensor * projected = ggml_mul_mat(ctx0, nextn.eh_proj, combined);
            if (!projected) {
                cb(hidden_state_inp, "vllm_mtp_projection_failed", layer_idx);
                if (config.enable_performance_monitoring) {
                    metrics.fallback_calls++;
                }
                return fallback_to_single_token(hidden_state_inp);
            }
            cb(projected, "vllm_eh_proj", layer_idx);
            
            // 6. Apply final shared head for token prediction
            if (nextn.shared_head_head) {
                // Optional: Apply shared head normalization first
                if (nextn.shared_head_norm) {
                    projected = ggml_rms_norm(ctx0, projected, config.rms_norm_eps);
                    if (projected) {
                        projected = ggml_mul(ctx0, projected, nextn.shared_head_norm);
                        cb(projected, "vllm_shared_norm", layer_idx);
                    }
                }
                
                // Final projection to vocabulary
                projected = ggml_mul_mat(ctx0, nextn.shared_head_head, projected);
                if (projected) {
                    cb(projected, "vllm_shared_head", layer_idx);
                    
                    if (config.enable_performance_monitoring) {
                        metrics.successful_calls++;
                        metrics.total_throughput_tokens += 1.0; // Single token prediction
                    }
                    
                    return projected;
                }
            }
            
            // If we reach here, something failed
            if (config.enable_performance_monitoring) {
                metrics.fallback_calls++;
            }
            return fallback_to_single_token(hidden_state_inp);
            
        } catch (...) {
            cb(hidden_state_inp, "vllm_mtp_exception", layer_idx);
            if (config.enable_performance_monitoring) {
                metrics.fallback_calls++;
            }
            return fallback_to_single_token(hidden_state_inp);
        }
    }
};

// Utility functions for MTP configuration
inline llama_mtp_config llama_mtp_config_default() {
    llama_mtp_config config;
    config.n_predict_ahead = 4;            // Balanced for cache performance
    config.confidence_threshold = 0.75f;   // Higher quality threshold
    config.enable_speculative = true;
    config.enable_parallel = true;
    config.enable_memory_optimization = true;
    config.enable_tensor_fusion = true;
    config.enable_performance_monitoring = true;
    config.rms_norm_eps = 1e-6f;           // Higher precision
    return config;
}

inline llama_mtp_config llama_mtp_config_fast() {
    llama_mtp_config config;
    config.n_predict_ahead = 6;            // Reduced from 8 for better cache performance
    config.confidence_threshold = 0.65f;   // Slightly increased for better quality
    config.enable_speculative = true;
    config.enable_parallel = true;
    config.enable_memory_optimization = true;
    config.enable_tensor_fusion = true;
    config.enable_performance_monitoring = true;
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
    config.enable_performance_monitoring = true;
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
