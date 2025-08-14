// MTP (Multi-Token Prediction) Utility Functions
// Helper functions for MTP validation and diagnostics

#pragma once
#include "llama-model.h"

namespace llama_mtp_utils {
    // Validate MTP layer configuration
    static bool validate_nextn_layer(const llama_layer_nextn & nextn) {
        if (!nextn.eh_proj || !nextn.shared_head_head) {
            return false; // Essential tensors missing
        }
        
        // Check tensor dimension consistency
        const int64_t eh_proj_rows = nextn.eh_proj->ne[0];
        const int64_t eh_proj_cols = nextn.eh_proj->ne[1];
        const int64_t head_rows = nextn.shared_head_head->ne[0];
        const int64_t head_cols = nextn.shared_head_head->ne[1];
        
        // Basic sanity checks
        if (eh_proj_rows <= 0 || eh_proj_cols <= 0 || head_rows <= 0 || head_cols <= 0) {
            return false;
        }
        
        return true;
    }
    
    // Check if model supports MTP functionality
    static bool is_mtp_model(const llama_model & model) {
        return model.hparams.nextn_predict_layers > 0;
    }
    
    // Get effective transformer layers (excluding MTP layers)
    static int get_transformer_layer_count(const llama_model & model) {
        return static_cast<int>(model.hparams.n_layer) - 
               static_cast<int>(model.hparams.nextn_predict_layers);
    }
    
    // Validate MTP layer indices
    static bool validate_mtp_layer_indices(const llama_model & model, 
                                         const std::vector<int> & layer_indices) {
        const int n_transformer_layers = get_transformer_layer_count(model);
        const int total_layers = static_cast<int>(model.hparams.n_layer);
        
        for (int idx : layer_indices) {
            if (idx < n_transformer_layers || idx >= total_layers) {
                return false; // Index out of MTP range
            }
        }
        return true;
    }
    
    // Get MTP layer indices for a model
    static std::vector<int> get_mtp_layer_indices(const llama_model & model) {
        std::vector<int> indices;
        
        if (!is_mtp_model(model)) {
            return indices;
        }
        
        const int n_transformer_layers = get_transformer_layer_count(model);
        const int total_layers = static_cast<int>(model.hparams.n_layer);
        
        for (int i = n_transformer_layers; i < total_layers; ++i) {
            indices.push_back(i);
        }
        
        return indices;
    }
}