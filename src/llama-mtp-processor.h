#pragma once

#include "llama-model.h"
#include "llama-mtp-optimized.h"
#include <vector>
#include <memory>

// vLLM-style MTP Processor based on GLM4.5 implementation
class llama_mtp_vllm_processor {
private:
    const llama_model & model;
    llama_mtp_config config;
    mutable llama_mtp_perf_metrics metrics;

public:
    llama_mtp_vllm_processor(const llama_model & model_, const llama_mtp_config & config_)
        : model(model_), config(config_) {}

    // Main vLLM-style MTP processing function
    // Based on GLM4MoeMultiTokenPredictorLayer.forward() in vLLM
    ggml_tensor * process_mtp_layer_vllm_style(
        ggml_context * ctx0,
        ggml_tensor * previous_hidden_states,  // from last transformer layer
        llama_token last_token_id,             // input_ids[positions == n_past]
        int n_past,                           // current position
        int layer_idx,                        // MTP layer index
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        if (config.enable_performance_monitoring) {
            metrics.total_calls++;
        }

        const auto & mtp_layer = model.layers[layer_idx];
        const auto & nextn = mtp_layer.nextn;

        // Enhanced validation
        if (!nextn.embed_tokens || !nextn.eh_proj || !nextn.shared_head_head) {
            cb(previous_hidden_states, "mtp_missing_tensors", layer_idx);
            if (config.enable_performance_monitoring) {
                metrics.validation_failures++;
                metrics.fallback_calls++;
            }
            return previous_hidden_states;
        }

        try {
            // Step 1: Get token embedding for last token (vLLM L97-98)
            ggml_tensor * inp_token_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            ggml_set_i32(inp_token_id, last_token_id);
            
            ggml_tensor * inputs_embeds = ggml_get_rows(ctx0, nextn.embed_tokens, inp_token_id);
            if (!inputs_embeds) {
                cb(previous_hidden_states, "mtp_embed_failed", layer_idx);
                if (config.enable_performance_monitoring) {
                    metrics.fallback_calls++;
                }
                return previous_hidden_states;
            }
            cb(inputs_embeds, "mtp_inputs_embeds", layer_idx);

            // Step 2: Mask inputs at position 0 (vLLM L100)
            // Note: In single token prediction, we don't have position 0 masking
            // but we apply enorm to token embeddings

            // Step 3: Apply enorm to token embeddings (vLLM L101)
            ggml_tensor * inputs_embeds_norm = inputs_embeds;
            if (nextn.enorm) {
                inputs_embeds_norm = ggml_rms_norm(ctx0, inputs_embeds, config.rms_norm_eps);
                inputs_embeds_norm = ggml_mul(ctx0, inputs_embeds_norm, nextn.enorm);
                cb(inputs_embeds_norm, "mtp_inputs_embeds_norm", layer_idx);
            }

            // Step 4: Apply hnorm to previous hidden states (vLLM L102)
            ggml_tensor * previous_hidden_states_norm = previous_hidden_states;
            if (nextn.hnorm) {
                previous_hidden_states_norm = ggml_rms_norm(ctx0, previous_hidden_states, config.rms_norm_eps);
                previous_hidden_states_norm = ggml_mul(ctx0, previous_hidden_states_norm, nextn.hnorm);
                cb(previous_hidden_states_norm, "mtp_prev_hidden_norm", layer_idx);
            }

            // Step 5: Concatenate embeddings (vLLM L104)
            // torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
            ggml_tensor * combined = ggml_concat(ctx0, inputs_embeds_norm, previous_hidden_states_norm, 0);
            if (!combined) {
                cb(previous_hidden_states, "mtp_concat_failed", layer_idx);
                if (config.enable_performance_monitoring) {
                    metrics.fallback_calls++;
                }
                return previous_hidden_states;
            }
            cb(combined, "mtp_combined", layer_idx);

            // Step 6: Apply eh_proj projection (vLLM L105)
            ggml_tensor * hidden_states = ggml_mul_mat(ctx0, nextn.eh_proj, combined);
            if (!hidden_states) {
                cb(previous_hidden_states, "mtp_eh_proj_failed", layer_idx);
                if (config.enable_performance_monitoring) {
                    metrics.fallback_calls++;
                }
                return previous_hidden_states;
            }
            cb(hidden_states, "mtp_eh_proj", layer_idx);

            // Step 7: Apply full transformer layer (mtp_block in vLLM)
            // This processes through attention and FFN of the MTP layer
            ggml_tensor * mtp_block_output = process_mtp_transformer_block(
                ctx0, hidden_states, n_past, layer_idx, cb
            );
            
            if (!mtp_block_output) {
                if (config.enable_performance_monitoring) {
                    metrics.fallback_calls++;
                }
                return previous_hidden_states;
            }

            // Step 8: Apply shared head for final token prediction (vLLM shared_head)
            ggml_tensor * final_output = apply_shared_head(
                ctx0, nextn, mtp_block_output, layer_idx, cb
            );

            if (final_output && config.enable_performance_monitoring) {
                metrics.successful_calls++;
                metrics.total_throughput_tokens += 1.0;
            }

            return final_output ? final_output : previous_hidden_states;

        } catch (...) {
            cb(previous_hidden_states, "mtp_exception", layer_idx);
            if (config.enable_performance_monitoring) {
                metrics.fallback_calls++;
            }
            return previous_hidden_states;
        }
    }

private:
    // Process the MTP transformer block (attention + FFN)
    ggml_tensor * process_mtp_transformer_block(
        ggml_context * ctx0,
        ggml_tensor * hidden_states,
        int n_past,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        const auto & mtp_layer = model.layers[layer_idx];
        ggml_tensor * residual = hidden_states;

        // Self-attention for MTP block
        ggml_tensor * attn_output = process_mtp_attention(
            ctx0, hidden_states, n_past, layer_idx, cb
        );
        
        if (!attn_output) {
            return nullptr;
        }

        // Residual connection
        ggml_tensor * attn_residual = ggml_add(ctx0, attn_output, residual);
        cb(attn_residual, "mtp_attn_residual", layer_idx);

        // FFN processing
        ggml_tensor * ffn_output = process_mtp_ffn(
            ctx0, attn_residual, layer_idx, cb
        );

        if (!ffn_output) {
            return attn_residual;
        }

        // Final residual connection
        ggml_tensor * final_output = ggml_add(ctx0, ffn_output, attn_residual);
        cb(final_output, "mtp_block_output", layer_idx);

        return final_output;
    }

    // MTP attention processing
    ggml_tensor * process_mtp_attention(
        ggml_context * ctx0,
        ggml_tensor * hidden_states,
        int n_past,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        const auto & mtp_layer = model.layers[layer_idx];
        const int64_t n_embd = model.hparams.n_embd;
        const int64_t n_head = model.hparams.n_head;
        const int64_t n_head_kv = model.hparams.n_head_kv;
        const int64_t n_embd_head = model.hparams.n_embd_head_v;

        // Pre-attention norm
        ggml_tensor * attn_inp = hidden_states;
        if (mtp_layer.attn_norm) {
            attn_inp = ggml_rms_norm(ctx0, hidden_states, config.rms_norm_eps);
            attn_inp = ggml_mul(ctx0, attn_inp, mtp_layer.attn_norm);
            cb(attn_inp, "mtp_attn_norm", layer_idx);
        }

        // Q, K, V projections
        ggml_tensor * Qcur = ggml_mul_mat(ctx0, mtp_layer.wq, attn_inp);
        if (mtp_layer.bq) {
            Qcur = ggml_add(ctx0, Qcur, mtp_layer.bq);
        }
        cb(Qcur, "mtp_Qcur", layer_idx);

        ggml_tensor * Kcur = ggml_mul_mat(ctx0, mtp_layer.wk, attn_inp);
        if (mtp_layer.bk) {
            Kcur = ggml_add(ctx0, Kcur, mtp_layer.bk);
        }
        cb(Kcur, "mtp_Kcur", layer_idx);

        ggml_tensor * Vcur = ggml_mul_mat(ctx0, mtp_layer.wv, attn_inp);
        if (mtp_layer.bv) {
            Vcur = ggml_add(ctx0, Vcur, mtp_layer.bv);
        }
        cb(Vcur, "mtp_Vcur", layer_idx);

        // Reshape for attention
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, 1);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, 1);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, 1);

        // Apply Q/K norm if available
        if (mtp_layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, config.rms_norm_eps);
            Qcur = ggml_mul(ctx0, Qcur, mtp_layer.attn_q_norm);
            cb(Qcur, "mtp_Qcur_normed", layer_idx);
        }
        if (mtp_layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, config.rms_norm_eps);
            Kcur = ggml_mul(ctx0, Kcur, mtp_layer.attn_k_norm);
            cb(Kcur, "mtp_Kcur_normed", layer_idx);
        }

        // Apply RoPE
        ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_i32(inp_pos, n_past);

        const auto & hparams = model.hparams;
        Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            hparams.n_rot, hparams.rope_type, hparams.n_ctx_orig, hparams.rope_freq_base, hparams.rope_freq_scale,
            hparams.rope_ext_factor, hparams.rope_attn_factor, hparams.rope_beta_fast, hparams.rope_beta_slow
        );

        Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr,
            hparams.n_rot, hparams.rope_type, hparams.n_ctx_orig, hparams.rope_freq_base, hparams.rope_freq_scale,
            hparams.rope_ext_factor, hparams.rope_attn_factor, hparams.rope_beta_fast, hparams.rope_beta_slow
        );

        cb(Qcur, "mtp_Qcur_rope", layer_idx);
        cb(Kcur, "mtp_Kcur_rope", layer_idx);

        // Attention computation (simplified for single token)
        ggml_tensor * kq = ggml_mul_mat(ctx0, Kcur, Qcur);
        kq = ggml_scale(ctx0, kq, 1.0f / sqrtf(float(n_embd_head)));
        kq = ggml_soft_max(ctx0, kq);
        cb(kq, "mtp_kq", layer_idx);

        ggml_tensor * attn_out = ggml_mul_mat(ctx0, Vcur, kq);
        cb(attn_out, "mtp_attn_out", layer_idx);

        // Output projection
        if (mtp_layer.wo) {
            attn_out = ggml_mul_mat(ctx0, mtp_layer.wo, attn_out);
            cb(attn_out, "mtp_wo", layer_idx);
        }

        return attn_out;
    }

    // MTP FFN processing (handles MoE if available)
    ggml_tensor * process_mtp_ffn(
        ggml_context * ctx0,
        ggml_tensor * hidden_states,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        const auto & mtp_layer = model.layers[layer_idx];

        // Post-attention norm
        ggml_tensor * ffn_inp = hidden_states;
        if (mtp_layer.attn_post_norm) {
            ffn_inp = ggml_rms_norm(ctx0, hidden_states, config.rms_norm_eps);
            ffn_inp = ggml_mul(ctx0, ffn_inp, mtp_layer.attn_post_norm);
            cb(ffn_inp, "mtp_post_attn_norm", layer_idx);
        }

        // Check if this is a MoE layer
        if (mtp_layer.ffn_gate_inp) {
            return process_mtp_moe_ffn(ctx0, ffn_inp, layer_idx, cb);
        } else {
            return process_mtp_dense_ffn(ctx0, ffn_inp, layer_idx, cb);
        }
    }

    // MTP MoE FFN processing
    ggml_tensor * process_mtp_moe_ffn(
        ggml_context * ctx0,
        ggml_tensor * ffn_inp,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        const auto & mtp_layer = model.layers[layer_idx];
        const auto & hparams = model.hparams;

        // Process routed experts
        ggml_tensor * routed_out = build_moe_ffn(ctx0, ffn_inp,
            mtp_layer.ffn_gate_inp,
            mtp_layer.ffn_up_exps,
            mtp_layer.ffn_gate_exps,
            mtp_layer.ffn_down_exps,
            mtp_layer.ffn_exp_probs_b,
            hparams.n_expert, hparams.n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            true, hparams.expert_weights_scale,
            static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
            layer_idx);
        cb(routed_out, "mtp_ffn_moe_out", layer_idx);

        // Process shared expert if available
        ggml_tensor * shared_out = nullptr;
        if (mtp_layer.ffn_up_shexp) {
            shared_out = build_ffn(ctx0, ffn_inp,
                mtp_layer.ffn_up_shexp, nullptr, nullptr,
                mtp_layer.ffn_gate_shexp, nullptr, nullptr,
                mtp_layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, layer_idx);
            cb(shared_out, "mtp_ffn_shexp_out", layer_idx);

            // Combine routed and shared outputs
            routed_out = ggml_add(ctx0, routed_out, shared_out);
            cb(routed_out, "mtp_ffn_combined", layer_idx);
        }

        return routed_out;
    }

    // MTP dense FFN processing
    ggml_tensor * process_mtp_dense_ffn(
        ggml_context * ctx0,
        ggml_tensor * ffn_inp,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        const auto & mtp_layer = model.layers[layer_idx];

        return build_ffn(ctx0, ffn_inp,
            mtp_layer.ffn_up, nullptr, nullptr,
            mtp_layer.ffn_gate, nullptr, nullptr,
            mtp_layer.ffn_down, nullptr, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, layer_idx);
    }

    // Apply shared head for final prediction
    ggml_tensor * apply_shared_head(
        ggml_context * ctx0,
        const llama_layer_nextn & nextn,
        ggml_tensor * hidden_states,
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        ggml_tensor * output = hidden_states;

        // Apply shared head normalization if available
        if (nextn.shared_head_norm) {
            output = ggml_rms_norm(ctx0, hidden_states, config.rms_norm_eps);
            output = ggml_mul(ctx0, output, nextn.shared_head_norm);
            cb(output, "mtp_shared_head_norm", layer_idx);
        }

        // Apply shared head projection
        if (nextn.shared_head_head) {
            output = ggml_mul_mat(ctx0, nextn.shared_head_head, output);
            cb(output, "mtp_shared_head_out", layer_idx);
        }

        return output;
    }

    // Helper functions from llm_graph_context (to be implemented)
    ggml_tensor * build_moe_ffn(ggml_context * ctx0, ggml_tensor * cur, /* ... other params ... */) {
        // Implementation would call the actual MoE FFN builder
        // This is a placeholder - actual implementation depends on llm_graph_context
        return nullptr;
    }

    ggml_tensor * build_ffn(ggml_context * ctx0, ggml_tensor * cur, /* ... other params ... */) {
        // Implementation would call the actual FFN builder
        // This is a placeholder - actual implementation depends on llm_graph_context
        return nullptr;
    }

public:
    // Get current performance metrics
    const llama_mtp_perf_metrics & get_performance_metrics() const {
        return metrics;
    }

    // Reset performance metrics
    void reset_performance_metrics() {
        metrics = llama_mtp_perf_metrics{};
    }

    // Check if MTP is available
    static bool is_mtp_available(const llama_model & model) {
        return model.hparams.nextn_predict_layers > 0;
    }
};