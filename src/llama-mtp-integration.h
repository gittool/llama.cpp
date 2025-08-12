#pragma once

#include "llama-model.h"
#include "llama-mtp-processor.h"
#include "llama-graph.h"

// Enhanced GLM4 MoE builder with integrated MTP support
// Based on vLLM's GLM4MoeMTP implementation
struct llm_build_glm4_moe_mtp : public llm_graph_context {
    llm_build_glm4_moe_mtp(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);
        ggml_tensor * inp_pos = build_inp_pos();
        auto * inp_attn = build_attn_inp_kv_unified();
        ggml_tensor * inp_out_ids = build_inp_out_ids();

        // Split between standard transformer layers and MTP layers
        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;

        // Phase 1: Standard transformer layers processing
        for (int il = 0; il < n_transformer_layers; ++il) {
            ggml_tensor * inpSA = inpL;

            // Pre-attention norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // Self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                }
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                }
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                }
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                // Apply Q/K norm if available
                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                    cb(Qcur, "Qcur_normed", il);
                }
                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                    cb(Kcur, "Kcur_normed", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_transformer_layers - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // Post-attention norm
            cur = build_norm(ffn_inp, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "post_attn_norm", il);

            // FFN processing (MoE or dense)
            if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
                // Dense FFN layer
                cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE FFN layer
                ggml_tensor * routed_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    hparams.n_expert, hparams.n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
                    il);
                cb(routed_out, "ffn_moe_out", il);

                // Process shared expert if available
                if (model.layers[il].ffn_up_shexp) {
                    ggml_tensor * shared_out = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(shared_out, "ffn_shexp_out", il);

                    cur = ggml_add(ctx0, routed_out, shared_out);
                } else {
                    cur = routed_out;
                }
                cb(cur, "ffn_out", il);
            }

            // Final residual connection
            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        // Phase 2: MTP layers processing (if available)
        ggml_tensor * mtp_output = nullptr;
        if (hparams.nextn_predict_layers > 0) {
            // Initialize MTP processor with optimized configuration
            llama_mtp_config mtp_config = llama_mtp_config_fast();
            llama_mtp_vllm_processor mtp_processor(model, mtp_config);

            // Get the last sampled token for MTP processing
            // Note: This would need to be passed from the inference context
            llama_token last_token_id = 0; // Placeholder - should come from context
            int n_past = n_tokens - 1; // Current position

            // Process each MTP layer
            ggml_tensor * current_hidden = inpL;
            for (int il = n_transformer_layers; il < n_layer; ++il) {
                // Apply vLLM-style MTP processing
                ggml_tensor * mtp_layer_output = mtp_processor.process_mtp_layer_vllm_style(
                    ctx0, current_hidden, last_token_id, n_past, il, cb
                );

                if (mtp_layer_output) {
                    current_hidden = mtp_layer_output;
                    cb(current_hidden, "mtp_layer_out", il);
                } else {
                    // Fallback to standard processing if MTP fails
                    cb(current_hidden, "mtp_fallback", il);
                    break;
                }
            }

            mtp_output = current_hidden;
            cb(mtp_output, "mtp_final", -1);
        }

        // Final processing
        cur = mtp_output ? mtp_output : inpL;

        // Final norm
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // Output projection
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// Utility function to create MTP-enabled GLM4 MoE model
inline std::unique_ptr<llm_graph_context> create_glm4_moe_mtp_builder(
    const llama_model & model, 
    const llm_graph_params & params
) {
    if (model.hparams.nextn_predict_layers > 0) {
        return std::make_unique<llm_build_glm4_moe_mtp>(model, params);
    } else {
        // Fallback to standard GLM4 MoE builder
        return std::make_unique<llm_build_glm4_moe>(model, params);
    }
}

// MTP context state for maintaining prediction state across calls
struct llama_mtp_context {
    std::vector<llama_token> predicted_tokens;
    std::vector<float> prediction_scores;
    llama_mtp_perf_metrics metrics;
    llama_mtp_config config;
    
    // Track current MTP processing state
    bool mtp_active = false;
    int current_step = 0;
    int max_parallel_predictions = 4;
    
    llama_mtp_context() : config(llama_mtp_config_default()) {}
    
    void reset() {
        predicted_tokens.clear();
        prediction_scores.clear();
        mtp_active = false;
        current_step = 0;
    }
    
    bool has_predictions() const {
        return !predicted_tokens.empty();
    }
    
    llama_token get_next_prediction() {
        if (predicted_tokens.empty()) {
            return -1;
        }
        llama_token token = predicted_tokens.front();
        predicted_tokens.erase(predicted_tokens.begin());
        if (!prediction_scores.empty()) {
            prediction_scores.erase(prediction_scores.begin());
        }
        return token;
    }
    
    void add_prediction(llama_token token, float score) {
        predicted_tokens.push_back(token);
        prediction_scores.push_back(score);
    }
    
    float get_average_confidence() const {
        if (prediction_scores.empty()) return 0.0f;
        float sum = 0.0f;
        for (float score : prediction_scores) {
            sum += score;
        }
        return sum / prediction_scores.size();
    }
};

// Integration with existing llama_context
struct llama_context_mtp_extension {
    llama_mtp_context mtp_state;
    std::unique_ptr<llama_mtp_vllm_processor> mtp_processor;
    
    explicit llama_context_mtp_extension(const llama_model & model) {
        if (llama_mtp_vllm_processor::is_mtp_available(model)) {
            mtp_processor = std::make_unique<llama_mtp_vllm_processor>(model, mtp_state.config);
        }
    }
    
    bool is_mtp_enabled() const {
        return mtp_processor != nullptr;
    }
    
    void reset_mtp() {
        mtp_state.reset();
    }
    
    const llama_mtp_perf_metrics & get_mtp_metrics() const {
        if (mtp_processor) {
            return mtp_processor->get_performance_metrics();
        }
        static llama_mtp_perf_metrics empty_metrics;
        return empty_metrics;
    }
};