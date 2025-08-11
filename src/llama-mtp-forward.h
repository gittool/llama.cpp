// MTP (Multi-Token Prediction) Forward Pass Implementation
// このファイルはGLM4のMTPレイヤーを実際に使用するための実装提案です

#pragma once
#include "llama-model.h"

// NextNレイヤーの処理関数
struct llm_nextn_processor {
    const llama_model & model;
    const llm_graph_params & params;
    ggml_context * ctx0;
    
    llm_nextn_processor(const llama_model & model_, const llm_graph_params & params_) 
        : model(model_), params(params_), ctx0(params_.ctx0) {}
    
    // NextNレイヤーでの多重トークン予測処理
    ggml_tensor * process_nextn_layer(ggml_tensor * cur, int layer_idx, int n_predict_ahead = 4) {
        const auto & nextn = model.layers[layer_idx].nextn;
        
        if (!nextn.eh_proj || !nextn.embed_tokens || !nextn.shared_head_head) {
            // NextNテンソルが存在しない場合は標準処理
            return cur;
        }
        
        // 1. 埋め込み投影 (eh_proj)
        ggml_tensor * projected = ggml_mul_mat(ctx0, nextn.eh_proj, cur);
        
        // 2. 入力正規化 (enorm)
        if (nextn.enorm) {
            projected = ggml_rms_norm(ctx0, projected, nextn.enorm);
        }
        
        // 3. 複数トークンの並列予測
        std::vector<ggml_tensor*> token_predictions;
        for (int i = 0; i < n_predict_ahead; ++i) {
            // 各予測位置での処理
            ggml_tensor * hidden_state = projected;
            
            // 隠れ状態正規化 (hnorm)
            if (nextn.hnorm) {
                hidden_state = ggml_rms_norm(ctx0, hidden_state, nextn.hnorm);
            }
            
            // 最終的なトークン予測 (shared_head_head)
            ggml_tensor * token_logits = ggml_mul_mat(ctx0, nextn.shared_head_head, hidden_state);
            
            // 共有ヘッド正規化 (shared_head_norm)  
            if (nextn.shared_head_norm) {
                token_logits = ggml_rms_norm(ctx0, token_logits, nextn.shared_head_norm);
            }
            
            token_predictions.push_back(token_logits);
            
            // 次の予測のために状態を更新
            // embed_tokensを使って予測トークンを次の入力に変換
            if (i < n_predict_ahead - 1 && nextn.embed_tokens) {
                // トークンを埋め込みに変換して次の予測に使用
                ggml_tensor * next_embed = ggml_get_rows(ctx0, nextn.embed_tokens, 
                    ggml_argmax(ctx0, token_logits));
                projected = ggml_add(ctx0, projected, next_embed);
            }
        }
        
        // 複数予測結果を結合
        if (token_predictions.size() == 1) {
            return token_predictions[0];
        }
        
        // 複数予測をconcatenateまたは平均化
        ggml_tensor * combined = token_predictions[0];
        for (size_t i = 1; i < token_predictions.size(); ++i) {
            combined = ggml_add(ctx0, combined, token_predictions[i]);
        }
        
        // 平均化
        combined = ggml_scale(ctx0, combined, 1.0f / token_predictions.size());
        
        return combined;
    }
};

// 改善されたGLM4 forward pass実装
struct llm_build_glm4_mtp : public llm_graph_context {
    llm_build_glm4_mtp(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        
        // MTPプロセッサーの初期化
        llm_nextn_processor nextn_proc(model, params);
        
        // 標準的なtransformerレイヤー数
        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);
        ggml_tensor * inp_pos = build_inp_pos();
        auto * inp_attn = build_attn_inp_kv_unified();
        ggml_tensor * inp_out_ids = build_inp_out_ids();

        // Phase 1: 標準Transformerレイヤー処理
        for (int il = 0; il < n_transformer_layers; ++il) {
            ggml_tensor * inpSA = inpL;

            // 標準のattention + FFN処理
            // ... (既存のGLM4処理と同じ)
            
            // Add residual connection
            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }
        
        // Phase 2: NextN/MTPレイヤー処理
        ggml_tensor * mtp_predictions = nullptr;
        if (hparams.nextn_predict_layers > 0) {
            for (int il = n_transformer_layers; il < n_layer; ++il) {
                // NextNレイヤーでの多重トークン予測
                ggml_tensor * nextn_output = nextn_proc.process_nextn_layer(inpL, il, 4);
                
                if (mtp_predictions == nullptr) {
                    mtp_predictions = nextn_output;
                } else {
                    // 複数NextNレイヤーの結果を統合
                    mtp_predictions = ggml_add(ctx0, mtp_predictions, nextn_output);
                }
                
                cb(nextn_output, "nextn_out", il);
            }
        }

        // Final norm
        cur = mtp_predictions ? mtp_predictions : inpL;
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // Output projection - MTPを考慮
        if (mtp_predictions) {
            // MTP予測がある場合は、それを優先使用
            cur = build_lora_mm(model.output, cur);
        } else {
            // 標準処理
            cur = build_lora_mm(model.output, cur);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// 投機的実行のための補助構造体
struct mtp_speculative_execution {
    static std::vector<llama_token> predict_multiple_tokens(
        struct llama_context * ctx,
        const llama_model & model,
        int n_predict,
        float confidence_threshold = 0.7f
    ) {
        std::vector<llama_token> predicted_tokens;
        
        if (model.hparams.nextn_predict_layers == 0) {
            return predicted_tokens; // MTP未対応
        }
        
        // NextNレイヤーからの多重予測を使用
        const float * logits = llama_get_logits(ctx);
        const int n_vocab = llama_n_vocab(llama_get_model(ctx));
        
        for (int i = 0; i < n_predict; ++i) {
            // 最も確信度の高いトークンを選択
            float max_prob = -1.0f;
            llama_token best_token = 0;
            
            for (int j = 0; j < n_vocab; ++j) {
                float prob = expf(logits[j]);
                if (prob > max_prob) {
                    max_prob = prob;
                    best_token = j;
                }
            }
            
            // 信頼度チェック
            if (max_prob < confidence_threshold) {
                break; // 低信頼度で予測中止
            }
            
            predicted_tokens.push_back(best_token);
            
            // 次の予測位置のlogitsに移動
            logits += n_vocab;
        }
        
        return predicted_tokens;
    }
};

// Complete GLM4_MOE MTP Builder with vLLM-style implementation
struct llm_build_glm4_moe_mtp : public llm_graph_context {
    llm_build_glm4_moe_mtp(const llama_model & model, const llm_graph_params & params,
        ggml_tensor * hidden_state_inp, llama_token last_token_id, int n_past
    ) : llm_graph_context(params) {
        
        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        // Assuming a single MTP layer at the end
        const int il = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il];

        ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_i32(inp_pos, n_past);

        ggml_tensor * cur;

        // 1. Get MTP embedding for last (conventionally sampled) token
        ggml_tensor * inp_token_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_i32(inp_token_id, last_token_id);
        
        ggml_tensor * token_emb = ggml_get_rows(ctx0, mtp_layer.nextn.embed_tokens, inp_token_id);
        ggml_tensor * token_emb_norm = ggml_rms_norm(ctx0, token_emb, hparams.f_norm_rms_eps);
        if (mtp_layer.nextn.enorm) {
            token_emb_norm = ggml_mul(ctx0, token_emb_norm, mtp_layer.nextn.enorm);
        }
        cb(token_emb_norm, "mtp_token_emb_norm", il);

        // 2. vLLM l99: previous_hidden_states = self.hnorm(previous_hidden_states)  
        ggml_tensor * hidden_state_norm = ggml_rms_norm(ctx0, hidden_state_inp, hparams.f_norm_rms_eps);
        if (mtp_layer.nextn.hnorm) {
            hidden_state_norm = ggml_mul(ctx0, hidden_state_norm, mtp_layer.nextn.hnorm);
        }
        cb(hidden_state_norm, "mtp_hidden_norm", il);

        // 3. Concatenate token embedding and hidden state (torch.cat)
        ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
        cb(combined, "mtp_combined", il);
        
        // 4. Apply eh_proj projection
        cur = ggml_mul_mat(ctx0, mtp_layer.nextn.eh_proj, combined);
        cb(cur, "mtp_eh_proj", il);

        // 5. Now proceed through complete last layer (previously skipped in main model)
        ggml_tensor * inpSA = cur;

        // Pre-attention norm for the MTP block
        ggml_tensor * attn_inp = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
        if (mtp_layer.attn_norm) {
            attn_inp = ggml_mul(ctx0, attn_inp, mtp_layer.attn_norm);
        }
        cb(attn_inp, "mtp_attn_norm", il);

        // 6. Self-attention for MTP layer
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, mtp_layer.wq, attn_inp);
            if (mtp_layer.bq) {
                Qcur = ggml_add(ctx0, Qcur, mtp_layer.bq);
            }
            cb(Qcur, "mtp_Qcur", il);

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, mtp_layer.wk, attn_inp);
            if (mtp_layer.bk) {
                Kcur = ggml_add(ctx0, Kcur, mtp_layer.bk);
            }
            cb(Kcur, "mtp_Kcur", il);

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, mtp_layer.wv, attn_inp);
            if (mtp_layer.bv) {
                Vcur = ggml_add(ctx0, Vcur, mtp_layer.bv);
            }
            cb(Vcur, "mtp_Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // Apply Q/K norm if available (GLM-4.5 355B variant)
            if (mtp_layer.attn_q_norm) {
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Qcur = ggml_mul(ctx0, Qcur, mtp_layer.attn_q_norm);
                cb(Qcur, "mtp_Qcur_normed", il);
            }
            if (mtp_layer.attn_k_norm) {
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                Kcur = ggml_mul(ctx0, Kcur, mtp_layer.attn_k_norm);
                cb(Kcur, "mtp_Kcur_normed", il);
            }

            // Apply RoPE
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

            cb(Qcur, "mtp_Qcur_rope", il);
            cb(Kcur, "mtp_Kcur_rope", il);
            cb(Vcur, "mtp_Vcur", il);

            // Attention computation (simplified for MTP single position)
            ggml_tensor * kq = ggml_mul_mat(ctx0, Kcur, Qcur);
            kq = ggml_scale(ctx0, kq, 1.0f/sqrtf(float(n_embd_head)));
            kq = ggml_soft_max(ctx0, kq);
            cb(kq, "mtp_kq", il);
            
            cur = ggml_mul_mat(ctx0, Vcur, kq);
            cb(cur, "mtp_attn_out", il);

            // Output projection
            if (mtp_layer.wo) {
                cur = ggml_mul_mat(ctx0, mtp_layer.wo, cur);
                cb(cur, "mtp_wo", il);
            }
        }

        // 7. Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "mtp_attn_residual", il);

        // 8. Post-attention norm
        cur = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);
        if (mtp_layer.attn_post_norm) {
            cur = ggml_mul(ctx0, cur, mtp_layer.attn_post_norm);
        }
        cb(cur, "mtp_post_attn_norm", il);

        // 9. MoE FFN for NextN block
        {
            // Process routed experts
            ggml_tensor * routed_out = build_moe_ffn(cur,
                    mtp_layer.ffn_gate_inp,
                    mtp_layer.ffn_up_exps,
                    mtp_layer.ffn_gate_exps,
                    mtp_layer.ffn_down_exps,
                    mtp_layer.ffn_exp_probs_b,
                    hparams.n_expert, hparams.n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(routed_out, "mtp_ffn_moe_out", il);

            // Process shared expert on original input  
            ggml_tensor * shared_out = build_ffn(cur,
                    mtp_layer.ffn_up_shexp,   NULL, NULL,
                    mtp_layer.ffn_gate_shexp, NULL, NULL,
                    mtp_layer.ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(shared_out, "mtp_ffn_shexp_out", il);

            // Final output: routed_output + shared_output
            cur = ggml_add(ctx0, routed_out, shared_out);
            cb(cur, "mtp_ffn_out", il);
        }

        // 10. FFN residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "mtp_ffn_residual", il);

        // 11. Final shared head processing
        if (mtp_layer.nextn.shared_head_norm) {
            cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
            cur = ggml_mul(ctx0, cur, mtp_layer.nextn.shared_head_norm);
            cb(cur, "mtp_shared_head_norm", il);
        }
        
        if (mtp_layer.nextn.shared_head_head) {
            cur = ggml_mul_mat(ctx0, mtp_layer.nextn.shared_head_head, cur);
            cb(cur, "mtp_shared_head_out", il);
        }

        res->t_logits = cur;
        ggml_build_forward_expand(gf, res->t_logits);
    }
};

// パフォーマンス監視構造体
struct mtp_performance_monitor {
    struct stats {
        int total_predictions = 0;
        int accepted_predictions = 0;
        float avg_confidence = 0.0f;
        double total_time = 0.0;
        int tokens_per_forward_pass = 1;
    } current_stats;
    
    void update_stats(int n_accepted, float confidence, double time_ms) {
        current_stats.total_predictions++;
        current_stats.accepted_predictions += n_accepted;
        current_stats.avg_confidence = 
            (current_stats.avg_confidence * (current_stats.total_predictions - 1) + confidence) / 
            current_stats.total_predictions;
        current_stats.total_time += time_ms;
        current_stats.tokens_per_forward_pass = 
            std::max(1, current_stats.accepted_predictions / current_stats.total_predictions);
    }
    
    float get_speedup() const {
        return static_cast<float>(current_stats.tokens_per_forward_pass);
    }
    
    void print_stats() const {
        printf("MTP Performance Stats:\n");
        printf("  Total predictions: %d\n", current_stats.total_predictions);
        printf("  Accepted: %d (%.1f%%)\n", current_stats.accepted_predictions,
               100.0f * current_stats.accepted_predictions / current_stats.total_predictions);
        printf("  Avg confidence: %.3f\n", current_stats.avg_confidence);
        printf("  Speedup: %.2fx\n", get_speedup());
        printf("  Avg time per prediction: %.2fms\n", 
               current_stats.total_time / current_stats.total_predictions);
    }
};
