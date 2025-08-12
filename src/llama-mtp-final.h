// Final MTP Integration Header
// This file provides the complete integration of MTP functionality into llama.cpp
// Based on vLLM's GLM4.5 MTP implementation

#pragma once

#include "llama-model.h"
#include "llama-mtp-optimized.h"
#include "llama-mtp-processor.h"
#include "llama-mtp-integration.h"

// Complete MTP API for external usage
namespace llama_mtp {

// Initialize MTP for a model (call once after model loading)
bool initialize_mtp(const llama_model & model);

// Check if MTP is available for this model
bool is_mtp_available(const llama_model & model);

// Get number of MTP layers
int get_mtp_layer_count(const llama_model & model);

// Process MTP layers for multi-token prediction
struct mtp_result {
    bool success = false;
    std::vector<llama_token> predicted_tokens;
    std::vector<float> prediction_scores;
    llama_mtp_perf_metrics metrics;
};

// Main MTP processing function
mtp_result process_mtp_prediction(
    struct llama_context * ctx,
    const llama_model & model,
    ggml_tensor * hidden_states,
    llama_token last_token,
    int n_predict = 4
);

// Configuration management
llama_mtp_config get_default_mtp_config();
llama_mtp_config get_fast_mtp_config();
llama_mtp_config get_conservative_mtp_config();

// Performance monitoring
void print_mtp_performance_report(const llama_mtp_perf_metrics & metrics);

} // namespace llama_mtp

// Integration modifications for existing llama.cpp functions

// Modified build_graph function for GLM4_MOE with MTP support
// This should be integrated into llama_model::build_graph()
inline std::unique_ptr<llm_graph_context> build_glm4_moe_with_mtp_support(
    const llama_model & model,
    const llm_graph_params & params
) {
    // Check if model has MTP layers
    if (model.hparams.nextn_predict_layers > 0) {
        // Use MTP-enhanced builder
        return std::make_unique<llm_build_glm4_moe_with_mtp>(model, params);
    } else {
        // Use standard builder
        return std::make_unique<llm_build_glm4_moe>(model, params);
    }
}

// Extension to llama_context for MTP state management
struct llama_context_mtp_state {
    std::unique_ptr<llama_mtp_context> mtp_ctx;
    std::unique_ptr<llama_mtp_vllm_processor> mtp_processor;
    bool mtp_enabled = false;
    
    void initialize(const llama_model & model) {
        if (llama_mtp::is_mtp_available(model)) {
            mtp_ctx = std::make_unique<llama_mtp_context>();
            mtp_processor = std::make_unique<llama_mtp_vllm_processor>(
                model, mtp_ctx->config
            );
            mtp_enabled = true;
        }
    }
    
    void reset() {
        if (mtp_ctx) {
            mtp_ctx->reset();
        }
    }
    
    bool has_predictions() const {
        return mtp_ctx && mtp_ctx->has_predictions();
    }
    
    llama_token get_next_prediction() {
        return mtp_ctx ? mtp_ctx->get_next_prediction() : -1;
    }
    
    const llama_mtp_perf_metrics & get_metrics() const {
        if (mtp_processor) {
            return mtp_processor->get_performance_metrics();
        }
        static llama_mtp_perf_metrics empty_metrics;
        return empty_metrics;
    }
};

// New API functions to be added to llama.h and implemented in llama.cpp

// Check if model supports MTP
LLAMA_API bool llama_model_has_mtp(const struct llama_model * model);

// Get number of MTP layers
LLAMA_API int32_t llama_model_mtp_layer_count(const struct llama_model * model);

// Enable/disable MTP processing for context
LLAMA_API void llama_context_set_mtp_enabled(struct llama_context * ctx, bool enabled);

// Check if MTP is enabled for context
LLAMA_API bool llama_context_is_mtp_enabled(const struct llama_context * ctx);

// Get MTP performance metrics
LLAMA_API void llama_context_get_mtp_metrics(
    const struct llama_context * ctx,
    struct llama_mtp_perf_metrics * metrics
);

// Reset MTP performance metrics
LLAMA_API void llama_context_reset_mtp_metrics(struct llama_context * ctx);

// Process multiple tokens with MTP (returns number of tokens predicted)
LLAMA_API int32_t llama_mtp_predict_tokens(
    struct llama_context * ctx,
    llama_token * tokens,           // output buffer for predicted tokens
    float * scores,                 // output buffer for prediction scores (optional)
    int32_t max_tokens,            // maximum tokens to predict
    float confidence_threshold     // minimum confidence to accept predictions
);

// Speculative decoding integration
LLAMA_API int32_t llama_mtp_speculative_decode(
    struct llama_context * ctx,
    llama_token * tokens,          // input/output token buffer
    int32_t n_input_tokens,        // number of input tokens
    int32_t max_new_tokens,        // maximum new tokens to generate
    float acceptance_threshold     // threshold for accepting predictions
);

// Implementation notes for integration:
/*
1. Add the following includes to llama-model.cpp:
   #include "llama-mtp-final.h"
   #include "llama-mtp-patch.cpp"

2. In llama_model::build_graph(), replace the GLM4_MOE case:
   case LLM_ARCH_GLM4_MOE:
       {
           llm = build_glm4_moe_with_mtp_support(*this, params);
       } break;

3. Add llama_context_mtp_state to the llama_context structure:
   struct llama_context {
       // ... existing members ...
       llama_context_mtp_state mtp_state;
   };

4. Initialize MTP in llama_new_context_with_model():
   ctx->mtp_state.initialize(model);

5. Add the new API functions to llama.cpp with proper error handling

6. Update the sampling loop in examples to use MTP when available:
   if (llama_context_is_mtp_enabled(ctx)) {
       // Try MTP prediction first
       int n_predicted = llama_mtp_predict_tokens(ctx, predicted_tokens, 
                                                 scores, 4, 0.7f);
       if (n_predicted > 0) {
           // Use predicted tokens
           for (int i = 0; i < n_predicted; ++i) {
               // Process predicted token
           }
       }
   }
*/