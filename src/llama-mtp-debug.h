#pragma once

#include "llama-mtp.h"
#include <iostream>
#include <iomanip>

// MTP debugging and diagnostics utilities
namespace llama_mtp_debug {
    
    // Print comprehensive MTP model information
    inline void print_mtp_info(const llama_model & model) {
        std::cout << "=== MTP Model Information ===\n";
        std::cout << "MTP Support: " << (llama_mtp::has_mtp_support(model) ? "✓ Yes" : "✗ No") << "\n";
        std::cout << "Total layers: " << model.hparams.n_layer << "\n";
        std::cout << "Transformer layers: " << llama_mtp::get_transformer_layers(model) << "\n";
        std::cout << "MTP layers: " << llama_mtp::get_mtp_layer_count(model) << "\n";
        std::cout << "Embedding dimension: " << model.hparams.n_embd << "\n";
        
        if (llama_mtp::has_mtp_support(model)) {
            std::cout << "\n=== MTP Layer Details ===\n";
            const int n_transformer = llama_mtp::get_transformer_layers(model);
            const int n_total = static_cast<int>(model.hparams.n_layer);
            
            for (int il = n_transformer; il < n_total; ++il) {
                const auto & nextn = model.layers[il].nextn;
                bool valid = llama_mtp::validate_nextn_layer(nextn);
                
                std::cout << "Layer " << il << " (MTP " << (il - n_transformer) << "): " 
                         << (valid ? "✓ Valid" : "✗ Invalid") << "\n";
                
                if (valid) {
                    std::cout << "  eh_proj: [" << nextn.eh_proj->ne[0] << ", " 
                             << nextn.eh_proj->ne[1] << "]\n";
                    std::cout << "  shared_head: [" << nextn.shared_head_head->ne[0] << ", " 
                             << nextn.shared_head_head->ne[1] << "]\n";
                    std::cout << "  enorm: " << (nextn.enorm ? "Present" : "Absent") << "\n";
                    std::cout << "  hnorm: " << (nextn.hnorm ? "Present" : "Absent") << "\n";
                    std::cout << "  head_norm: " << (nextn.shared_head_norm ? "Present" : "Absent") << "\n";
                }
            }
        }
        std::cout << "=============================\n\n";
    }
    
    // Validate MTP layer dimensions
    inline bool validate_mtp_dimensions(const llama_model & model, bool verbose = false) {
        if (!llama_mtp::has_mtp_support(model)) {
            if (verbose) std::cout << "No MTP layers to validate.\n";
            return true;
        }
        
        const int n_transformer = llama_mtp::get_transformer_layers(model);
        const int n_total = static_cast<int>(model.hparams.n_layer);
        const int n_embd = static_cast<int>(model.hparams.n_embd);
        
        bool all_valid = true;
        
        for (int il = n_transformer; il < n_total; ++il) {
            const auto & nextn = model.layers[il].nextn;
            
            if (!llama_mtp::validate_nextn_layer(nextn)) {
                if (verbose) std::cout << "❌ Layer " << il << ": Missing required tensors\n";
                all_valid = false;
                continue;
            }
            
            // Check eh_proj dimensions
            const auto & eh = nextn.eh_proj;
            bool eh_valid = false;
            
            if (eh->ne[0] == n_embd && eh->ne[1] == n_embd) {
                eh_valid = true; // Standard GLM4
                if (verbose) std::cout << "✅ Layer " << il << ": Standard GLM4 projection [" 
                                      << n_embd << ", " << n_embd << "]\n";
            } else if (eh->ne[0] == 2 * n_embd && eh->ne[1] == n_embd) {
                eh_valid = true; // GLM4 MOE
                if (verbose) std::cout << "✅ Layer " << il << ": GLM4-MOE projection [" 
                                      << 2 * n_embd << ", " << n_embd << "]\n";
            } else {
                if (verbose) std::cout << "⚠️  Layer " << il << ": Unusual projection dimensions [" 
                                      << eh->ne[0] << ", " << eh->ne[1] << "]\n";
            }
            
            // Check head dimensions
            const auto & head = nextn.shared_head_head;
            if (head->ne[0] != n_embd) {
                if (verbose) std::cout << "⚠️  Layer " << il << ": Head input dimension " 
                                      << head->ne[0] << " != n_embd " << n_embd << "\n";
            }
            
            all_valid = all_valid && eh_valid;
        }
        
        return all_valid;
    }
    
    // Performance timing callback for MTP operations
    class mtp_profiler {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        std::string current_op;
        bool enabled;
        
    public:
        mtp_profiler(bool enable = false) : enabled(enable) {}
        
        void operator()(ggml_tensor * tensor, const char * name, int layer) {
            if (!enabled) return;
            
            auto now = std::chrono::high_resolution_clock::now();
            if (!current_op.empty()) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time);
                std::cout << std::setw(20) << current_op << " (layer " << std::setw(2) << layer 
                         << "): " << std::setw(6) << duration.count() << " μs\n";
            }
            
            current_op = std::string(name);
            start_time = now;
        }
        
        void enable() { enabled = true; }
        void disable() { enabled = false; }
    };
    
    // Check matrix multiplication compatibility for debugging
    inline bool check_mul_mat_compatibility(const ggml_tensor* a, const ggml_tensor* b, bool verbose = false) {
        if (!a || !b) {
            if (verbose) std::cout << "❌ Null tensor in mul_mat check\n";
            return false;
        }
        
        // Basic dimension requirements for matrix multiplication: a[0] == b[0]
        bool compatible = (a->ne[0] == b->ne[0]) && (a->ne[1] >= 1) && (b->ne[1] >= 1);
        
        if (verbose) {
            std::cout << "Matrix multiplication compatibility check:\n";
            std::cout << "  Tensor A: [" << a->ne[0] << ", " << a->ne[1] << ", " << a->ne[2] << ", " << a->ne[3] << "]\n";
            std::cout << "  Tensor B: [" << b->ne[0] << ", " << b->ne[1] << ", " << b->ne[2] << ", " << b->ne[3] << "]\n";
            std::cout << "  Compatible: " << (compatible ? "✅ Yes" : "❌ No") << "\n";
        }
        
        return compatible;
    }
    
    // Enhanced MTP layer dimension validation
    inline void debug_mtp_layer_dimensions(const llama_layer_nextn & nextn, int layer_idx, int n_embd, bool verbose = true) {
        if (!verbose) return;
        
        std::cout << "=== MTP Layer " << layer_idx << " Dimension Debug ===\n";
        
        if (nextn.eh_proj) {
            std::cout << "eh_proj: [" << nextn.eh_proj->ne[0] << ", " << nextn.eh_proj->ne[1] << "]\n";
        }
        if (nextn.shared_head_head) {
            std::cout << "shared_head: [" << nextn.shared_head_head->ne[0] << ", " << nextn.shared_head_head->ne[1] << "]\n";
        }
        std::cout << "Expected n_embd: " << n_embd << "\n";
        std::cout << "==========================================\n";
    }
    
} // namespace llama_mtp_debug
