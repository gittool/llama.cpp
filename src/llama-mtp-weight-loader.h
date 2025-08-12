// Enhanced Weight Loading for MTP (Multi-Token Prediction)
// Based on vLLM's weight mapping and rewriting system
// Handles shared weights and proper layer index mapping

#pragma once
#include "llama-mtp-enhanced.h"
#include <string>
#include <vector>
#include <unordered_set>

// Weight name rewriting utilities (similar to vLLM's _rewrite_spec_layer_name)
class llama_mtp_weight_mapper {
private:
    const llama_model & model;
    const llama_hparams & hparams;
    
    // MTP-specific weight names that need special handling
    static const std::vector<std::string> mtp_layer_weight_names;
    static const std::vector<std::string> shared_weight_names;
    
public:
    llama_mtp_weight_mapper(const llama_model & model_) 
        : model(model_), hparams(model_.hparams) {}
    
    // Check if this is an MTP layer index
    bool is_mtp_layer(int32_t layer_idx) const {
        int32_t mtp_start = hparams.n_layer;
        int32_t mtp_end = mtp_start + hparams.nextn_predict_layers;
        return layer_idx >= mtp_start && layer_idx < mtp_end;
    }
    
    // Extract layer index from weight name
    int32_t extract_layer_index(const std::string & weight_name) const {
        // Look for pattern like "model.layers.XX."
        size_t layers_pos = weight_name.find("model.layers.");
        if (layers_pos == std::string::npos) return -1;
        
        size_t start = layers_pos + 13; // Length of "model.layers."
        size_t end = weight_name.find('.', start);
        if (end == std::string::npos) return -1;
        
        std::string layer_str = weight_name.substr(start, end - start);
        try {
            return std::stoi(layer_str);
        } catch (...) {
            return -1;
        }
    }
    
    // Check if weight name contains MTP-specific components
    bool is_mtp_weight(const std::string & weight_name) const {
        for (const auto & mtp_name : mtp_layer_weight_names) {
            if (weight_name.find(mtp_name) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    
    // Check if weight is shared across MTP layers
    bool is_shared_weight(const std::string & weight_name) const {
        for (const auto & shared_name : shared_weight_names) {
            if (weight_name.find(shared_name) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    
    // Rewrite weight name for proper mapping
    std::string rewrite_weight_name(const std::string & original_name, int32_t layer_idx) const {
        if (!is_mtp_layer(layer_idx)) {
            return original_name; // No rewriting needed for non-MTP layers
        }
        
        std::string name = original_name;
        
        // Check if this is an MTP-specific weight
        bool is_mtp_specific = is_mtp_weight(name);
        bool is_shared = is_shared_weight(name);
        
        if (!is_mtp_specific) {
            // This is a transformer block weight, add .mtp_block prefix
            std::string layer_prefix = "model.layers." + std::to_string(layer_idx) + ".";
            std::string new_layer_prefix = layer_prefix + "mtp_block.";
            
            size_t pos = name.find(layer_prefix);
            if (pos != std::string::npos) {
                name.replace(pos, layer_prefix.length(), new_layer_prefix);
            }
        } else if (is_shared) {
            // This is a shared weight, move to top level
            std::string layer_prefix = "model.layers." + std::to_string(layer_idx) + ".";
            std::string new_prefix = "model.";
            
            size_t pos = name.find(layer_prefix);
            if (pos != std::string::npos) {
                name.replace(pos, layer_prefix.length(), new_prefix);
            }
        }
        
        return name;
    }
    
    // Get the canonical weight name for shared weights
    std::string get_canonical_shared_name(const std::string & weight_name) const {
        if (!is_shared_weight(weight_name)) {
            return weight_name;
        }
        
        // For shared weights, use the first MTP layer's weight name as canonical
        int32_t first_mtp_layer = hparams.n_layer;
        
        // Replace any layer index with the first MTP layer index
        std::string canonical = weight_name;
        int32_t layer_idx = extract_layer_index(weight_name);
        
        if (layer_idx >= 0 && is_mtp_layer(layer_idx)) {
            std::string old_layer = "model.layers." + std::to_string(layer_idx) + ".";
            std::string new_layer = "model.layers." + std::to_string(first_mtp_layer) + ".";
            
            size_t pos = canonical.find(old_layer);
            if (pos != std::string::npos) {
                canonical.replace(pos, old_layer.length(), new_layer);
            }
        }
        
        return canonical;
    }
};

// Static member definitions
const std::vector<std::string> llama_mtp_weight_mapper::mtp_layer_weight_names = {
    "embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"
};

const std::vector<std::string> llama_mtp_weight_mapper::shared_weight_names = {
    "embed_tokens"
};

// Enhanced weight loader for MTP models
class llama_mtp_weight_loader {
private:
    llama_mtp_enhanced & mtp_processor;
    llama_mtp_weight_mapper mapper;
    
    // Track loaded shared weights to avoid duplicates
    std::unordered_set<std::string> loaded_shared_weights;
    
    // Stacked parameter mappings (from vLLM)
    struct stacked_param_mapping {
        std::string param_name;
        std::string weight_name;
        int shard_id;
    };
    
    std::vector<stacked_param_mapping> stacked_params_mapping = {
        {"qkv_proj", "q_proj", 0},
        {"qkv_proj", "k_proj", 1},
        {"qkv_proj", "v_proj", 2},
        {"gate_up_proj", "gate_proj", 0},
        {"gate_up_proj", "up_proj", 1},
    };
    
public:
    llama_mtp_weight_loader(llama_mtp_enhanced & mtp_processor_, const llama_model & model)
        : mtp_processor(mtp_processor_), mapper(model) {}
    
    // Load weights with proper mapping and sharing
    std::unordered_set<std::string> load_weights(
        const std::vector<std::pair<std::string, ggml_tensor *>> & weights
    ) {
        std::unordered_set<std::string> loaded_params;
        
        for (const auto & [name, loaded_weight] : weights) {
            int32_t layer_idx = mapper.extract_layer_index(name);
            
            // Skip non-MTP layers
            if (layer_idx < 0 || !mapper.is_mtp_layer(layer_idx)) {
                continue;
            }
            
            // Rewrite weight name
            std::string rewritten_name = mapper.rewrite_weight_name(name, layer_idx);
            
            // Handle shared weights
            if (mapper.is_shared_weight(name)) {
                std::string canonical_name = mapper.get_canonical_shared_name(name);
                
                // Only load shared weights once
                if (loaded_shared_weights.find(canonical_name) != loaded_shared_weights.end()) {
                    continue;
                }
                loaded_shared_weights.insert(canonical_name);
                
                // Load shared embedding weights
                if (name.find("embed_tokens") != std::string::npos) {
                    mtp_processor.get_predictor()->set_embed_tokens(loaded_weight);
                    loaded_params.insert(rewritten_name);
                    continue;
                }
            }
            
            // Handle stacked parameters
            bool handled_stacked = false;
            for (const auto & mapping : stacked_params_mapping) {
                if (name.find(mapping.weight_name) != std::string::npos) {
                    // TODO: Implement stacked parameter loading
                    // This would require more complex tensor reshaping
                    handled_stacked = true;
                    break;
                }
            }
            
            if (handled_stacked) {
                loaded_params.insert(rewritten_name);
                continue;
            }
            
            // Load individual MTP layer weights
            if (!load_mtp_layer_weight(layer_idx, rewritten_name, loaded_weight)) {
                // Skip if loading failed
                continue;
            }
            
            loaded_params.insert(rewritten_name);
        }
        
        return loaded_params;
    }
    
private:
    // Load weight for specific MTP layer
    bool load_mtp_layer_weight(int32_t layer_idx, const std::string & weight_name, ggml_tensor * weight) {
        auto * predictor = mtp_processor.get_predictor();
        if (!predictor) return false;
        
        // Extract weight type from name
        if (weight_name.find("enorm") != std::string::npos) {
            predictor->set_layer_weights(layer_idx, weight, nullptr, nullptr, nullptr, nullptr);
            return true;
        } else if (weight_name.find("hnorm") != std::string::npos) {
            predictor->set_layer_weights(layer_idx, nullptr, weight, nullptr, nullptr, nullptr);
            return true;
        } else if (weight_name.find("eh_proj") != std::string::npos) {
            predictor->set_layer_weights(layer_idx, nullptr, nullptr, weight, nullptr, nullptr);
            return true;
        } else if (weight_name.find("shared_head.norm") != std::string::npos || 
                   weight_name.find("shared_head_norm") != std::string::npos) {
            predictor->set_layer_weights(layer_idx, nullptr, nullptr, nullptr, weight, nullptr);
            return true;
        } else if (weight_name.find("shared_head.head") != std::string::npos || 
                   weight_name.find("shared_head_head") != std::string::npos) {
            predictor->set_layer_weights(layer_idx, nullptr, nullptr, nullptr, nullptr, weight);
            return true;
        }
        
        // Handle transformer block weights
        if (weight_name.find("mtp_block") != std::string::npos) {
            // TODO: Load transformer block weights (attention, FFN, etc.)
            return true;
        }
        
        return false;
    }
};

// Utility function to integrate with existing weight loading system
inline std::unordered_set<std::string> llama_load_mtp_weights(
    llama_mtp_enhanced & mtp_processor,
    const llama_model & model,
    const std::vector<std::pair<std::string, ggml_tensor *>> & weights
) {
    llama_mtp_weight_loader loader(mtp_processor, model);
    return loader.load_weights(weights);
}