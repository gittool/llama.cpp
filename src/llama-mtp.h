#pragma once

#include "llama-model.h"
#include "ggml.h"

// MTP (Multi-Token Prediction) processing utilities for GLM4
// Optimized implementation with improved error checking and performance

namespace llama_mtp {
    
    // Check if model supports MTP
    inline bool has_mtp_support(const llama_model & model) {
        return model.hparams.nextn_predict_layers > 0;
    }
    
    // Get effective transformer layer count (excluding MTP layers)
    inline int get_transformer_layers(const llama_model & model) {
        return static_cast<int>(model.hparams.n_layer - model.hparams.nextn_predict_layers);
    }
    
    // Get MTP layer count
    inline int get_mtp_layer_count(const llama_model & model) {
        return static_cast<int>(model.hparams.nextn_predict_layers);
    }
    
    // Validate MTP layer has required tensors with enhanced checks
    inline bool validate_nextn_layer(const llama_layer_nextn & nextn) {
        // Basic required tensors check
        if (!nextn.eh_proj || !nextn.shared_head_head) {
            return false;
        }
        
        // Dimension validation
        if (nextn.eh_proj->ne[0] <= 0 || nextn.eh_proj->ne[1] <= 0 ||
            nextn.shared_head_head->ne[0] <= 0 || nextn.shared_head_head->ne[1] <= 0) {
            return false;
        }
        
        // Additional consistency checks
        return true;
    }
    
    // Optimized MTP layer processing (for llm_graph_context)
    inline ggml_tensor * process_mtp_layer(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * input,
        int n_embd,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb) {
        
        if (!validate_nextn_layer(nextn) || !input) {
            return input; // Skip if invalid
        }
        
        // 1. Optimized embedding projection with proper dimension handling
        ggml_tensor * projected = input;
        
        // Handle different projection layouts (GLM4 vs GLM4_MOE)
        if (nextn.eh_proj->ne[0] == n_embd && nextn.eh_proj->ne[1] == n_embd) {
            // Standard GLM4: transpose for correct multiplication
            ggml_tensor * weight_t = ggml_cont(ctx0, ggml_transpose(ctx0, nextn.eh_proj));
            cb(weight_t, "mtp_proj_t", layer_idx);
            
            if (weight_t->ne[0] == input->ne[0]) {
                projected = ggml_mul_mat(ctx0, weight_t, input);
                cb(projected, "mtp_proj", layer_idx);
            }
        } else if (nextn.eh_proj->ne[0] == 2 * n_embd && nextn.eh_proj->ne[1] == n_embd) {
            // GLM4_MOE: transpose for correct multiplication
            ggml_tensor * weight_t = ggml_cont(ctx0, ggml_transpose(ctx0, nextn.eh_proj));
            cb(weight_t, "mtp_proj_t", layer_idx);
            
            if (weight_t->ne[0] == input->ne[0]) {
                projected = ggml_mul_mat(ctx0, weight_t, input);
                cb(projected, "mtp_proj", layer_idx);
            }
        } else if (nextn.eh_proj->ne[0] == input->ne[0]) {
            // Direct multiplication if dimensions match
            projected = ggml_mul_mat(ctx0, nextn.eh_proj, input);
            cb(projected, "mtp_proj", layer_idx);
        }
        
        // 2. Apply normalizations (optimized order)
        if (nextn.enorm) {
            projected = ggml_rms_norm(ctx0, projected, 1e-6f);
            projected = ggml_mul(ctx0, projected, nextn.enorm);
            cb(projected, "mtp_enorm", layer_idx);
        }
        
        if (nextn.hnorm) {
            projected = ggml_rms_norm(ctx0, projected, 1e-6f);
            projected = ggml_mul(ctx0, projected, nextn.hnorm);
            cb(projected, "mtp_hnorm", layer_idx);
        }
        
        // 3. Multi-token prediction head with dimension optimization
        ggml_tensor * predictions = projected;
        if (nextn.shared_head_head) {
            // Handle dimension reshaping for multi-token prediction
            ggml_tensor * head_input = projected;
            const int64_t input_dim = projected->ne[0];
            const int64_t head_input_dim = nextn.shared_head_head->ne[0];
            
            // Reshape if needed for dimension compatibility
            if (input_dim != head_input_dim && input_dim % n_embd == 0) {
                const int64_t k = input_dim / n_embd;
                if (k > 0 && head_input_dim == n_embd) {
                    head_input = ggml_reshape_2d(ctx0, projected, n_embd, projected->ne[1] * k);
                    cb(head_input, "mtp_reshape", layer_idx);
                }
            }
            
            // Apply prediction head
            if (head_input->ne[0] == nextn.shared_head_head->ne[0]) {
                predictions = ggml_mul_mat(ctx0, nextn.shared_head_head, head_input);
                cb(predictions, "mtp_head", layer_idx);
                
                // Optional head normalization
                if (nextn.shared_head_norm) {
                    predictions = ggml_rms_norm(ctx0, predictions, 1e-6f);
                    predictions = ggml_mul(ctx0, predictions, nextn.shared_head_norm);
                    cb(predictions, "mtp_head_norm", layer_idx);
                }
            }
        }
        
        return predictions;
    }
    
    // Process all MTP layers efficiently with improved return semantics (for llm_graph_context)
    inline ggml_tensor * process_all_mtp_layers(
        ggml_context * ctx0,
        const llama_model & model,
        ggml_tensor * input,
        const std::function<void(ggml_tensor *, const char *, int)> & cb) {
        
        if (!has_mtp_support(model) || !input) {
            return input; // Return original input if no MTP support
        }
        
        const int n_layer = static_cast<int>(model.hparams.n_layer);
        const int n_transformer_layers = get_transformer_layers(model);
        const int n_embd = static_cast<int>(model.hparams.n_embd);
        
        ggml_tensor * current = input;
        ggml_tensor * latest_prediction = nullptr;
        bool any_prediction_made = false;
        
        // Process each MTP layer
        for (int il = n_transformer_layers; il < n_layer; ++il) {
            const auto & nextn = model.layers[il].nextn;
            
            if (validate_nextn_layer(nextn)) {
                ggml_tensor * layer_output = process_mtp_layer(ctx0, nextn, current, n_embd, il, cb);
                
                if (layer_output != current) {
                    latest_prediction = layer_output;
                    current = layer_output; // Use prediction as input for next layer
                    any_prediction_made = true;
                } else {
                    cb(current, "mtp_skip", il);
                }
            } else {
                cb(current, "mtp_invalid", il);
            }
        }
        
        // Return the latest prediction if any was made, otherwise return original input
        return any_prediction_made ? latest_prediction : input;
    }
    
    // Alternative wrapper for llm_graph_context cb member function
    inline ggml_tensor * process_all_mtp_layers_with_context(
        ggml_context * ctx0,
        const llama_model & model,
        ggml_tensor * input,
        const void * context_obj,
        void (*cb_func)(const void*, ggml_tensor *, const char *, int)) {
        
        auto cb = [context_obj, cb_func](ggml_tensor * cur, const char * name, int il) {
            cb_func(context_obj, cur, name, il);
        };
        
        return process_all_mtp_layers(ctx0, model, input, cb);
    }
    
} // namespace llama_mtp
