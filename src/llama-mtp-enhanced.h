// Enhanced MTP Implementation - 既存コードとの整合性を重視
// 現在のllama.cppのMTP実装を拡張し、vLLMスタイルの処理を追加

#pragma once

#include "llama-model.h"
#include "llama-mtp-optimized.h"

// 既存のMTP実装を拡張するクラス
class llama_mtp_enhanced_processor {
private:
    const llama_model & model;
    llama_mtp_config config;
    mutable llama_mtp_perf_metrics metrics;

public:
    llama_mtp_enhanced_processor(const llama_model & model_, const llama_mtp_config & config_)
        : model(model_), config(config_) {}

    // 既存のMTP処理を拡張 - vLLMスタイルのtoken embedding処理を追加
    ggml_tensor * process_enhanced_mtp_layer(
        ggml_context * ctx0,
        ggml_tensor * hidden_states,
        llama_token last_token_id,  // 最後にサンプリングされたtoken
        int layer_idx,
        const std::function<void(ggml_tensor *, const char *, int)> & cb
    ) {
        const auto & nextn = model.layers[layer_idx].nextn;
        
        // 既存のテンソルチェック
        if (!nextn.eh_proj || !nextn.shared_head_head) {
            cb(hidden_states, "mtp_enhanced_skip", layer_idx);
            return hidden_states;
        }

        const int64_t n_embd = model.hparams.n_embd;
        ggml_tensor * cur = hidden_states;

        try {
            // Phase 1: vLLMスタイルのtoken embedding処理（新機能）
            ggml_tensor * token_contribution = nullptr;
            if (nextn.embed_tokens && last_token_id >= 0) {
                // 1. Get token embedding for last token (vLLM style)
                ggml_tensor * inp_token_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
                ggml_set_i32(inp_token_id, last_token_id);
                
                ggml_tensor * token_emb = ggml_get_rows(ctx0, nextn.embed_tokens, inp_token_id);
                if (token_emb) {
                    cb(token_emb, "mtp_token_emb", layer_idx);
                    
                    // 2. Apply enorm to token embeddings (vLLM L101)
                    if (nextn.enorm) {
                        token_emb = ggml_rms_norm(ctx0, token_emb, config.rms_norm_eps);
                        token_emb = ggml_mul(ctx0, token_emb, nextn.enorm);
                        cb(token_emb, "mtp_token_enorm", layer_idx);
                    }
                    
                    token_contribution = token_emb;
                }
            }

            // Phase 2: Hidden state normalization (vLLM L102)
            ggml_tensor * hidden_normalized = cur;
            if (nextn.hnorm) {
                hidden_normalized = ggml_rms_norm(ctx0, cur, config.rms_norm_eps);
                hidden_normalized = ggml_mul(ctx0, hidden_normalized, nextn.hnorm);
                cb(hidden_normalized, "mtp_hidden_hnorm", layer_idx);
            }

            // Phase 3: Combine token and hidden representations
            ggml_tensor * combined_input = hidden_normalized;
            if (token_contribution) {
                // vLLM style concatenation: torch.cat([token_emb, hidden_states], dim=-1)
                combined_input = ggml_concat(ctx0, token_contribution, hidden_normalized, 0);
                cb(combined_input, "mtp_combined", layer_idx);
            }

            // Phase 4: 既存のeh_proj処理を改良
            ggml_tensor * eh_proj_for_mul = nextn.eh_proj;
            
            // GLM4_MOE vs GLM4の次元処理 - 既存ロジックを維持
            if (nextn.eh_proj->ne[0] == 2 * n_embd && nextn.eh_proj->ne[1] == n_embd) {
                // GLM4_MOE case: transpose for correct multiplication
                eh_proj_for_mul = ggml_cont(ctx0, ggml_transpose(ctx0, nextn.eh_proj));
                cb(eh_proj_for_mul, "mtp_eh_proj_transpose_moe", layer_idx);
            } else if (nextn.eh_proj->ne[0] == n_embd && nextn.eh_proj->ne[1] == n_embd) {
                // Standard GLM4: transpose for correct dimensions
                eh_proj_for_mul = ggml_cont(ctx0, ggml_transpose(ctx0, nextn.eh_proj));
                cb(eh_proj_for_mul, "mtp_eh_proj_transpose_std", layer_idx);
            }

            // 既存の安全性チェックを維持
            if (eh_proj_for_mul->ne[0] == combined_input->ne[0]) {
                cur = ggml_mul_mat(ctx0, eh_proj_for_mul, combined_input);
                cb(cur, "mtp_eh_proj", layer_idx);

                // Phase 5: 既存の正規化処理を維持
                if (nextn.enorm && !token_contribution) { // token embeddingで既に適用していない場合
                    cur = ggml_rms_norm(ctx0, cur, config.rms_norm_eps);
                    cur = ggml_mul(ctx0, cur, nextn.enorm);
                    cb(cur, "mtp_enorm_fallback", layer_idx);
                }

                // Phase 6: Shared head processing - 既存ロジックを維持
                if (nextn.shared_head_norm) {
                    cur = ggml_rms_norm(ctx0, cur, config.rms_norm_eps);
                    cur = ggml_mul(ctx0, cur, nextn.shared_head_norm);
                    cb(cur, "mtp_shared_head_norm", layer_idx);
                }

                if (nextn.shared_head_head) {
                    // 既存の次元チェック
                    if (nextn.shared_head_head->ne[0] == cur->ne[0]) {
                        cur = ggml_mul_mat(ctx0, nextn.shared_head_head, cur);
                        cb(cur, "mtp_shared_head", layer_idx);
                        
                        if (config.enable_performance_monitoring) {
                            metrics.successful_calls++;
                            metrics.total_throughput_tokens += 1.0;
                        }
                        
                        return cur;
                    } else {
                        cb(cur, "mtp_shared_head_dim_mismatch", layer_idx);
                    }
                }
            } else {
                cb(combined_input, "mtp_eh_proj_dim_mismatch", layer_idx);
            }
            
        } catch (...) {
            cb(hidden_states, "mtp_enhanced_exception", layer_idx);
            if (config.enable_performance_monitoring) {
                metrics.fallback_calls++;
            }
        }

        // Fallback to input if processing fails
        if (config.enable_performance_monitoring) {
            metrics.fallback_calls++;
        }
        return hidden_states;
    }

    // 既存のprocess_mtp_layersと互換性のあるインターフェース
    ggml_tensor * process_enhanced_mtp_layers(
        ggml_context * ctx0,
        ggml_tensor * input_tensor,
        const std::vector<int> & layer_indices,
        llama_token last_token_id,  // 追加パラメータ
        const std::function<void(ggml_tensor *, const char *, int)> & callback
    ) {
        if (config.enable_performance_monitoring) {
            metrics.total_calls++;
        }

        if (!ctx0 || !input_tensor || layer_indices.empty()) {
            if (config.enable_performance_monitoring) {
                metrics.validation_failures++;
            }
            return input_tensor;
        }

        ggml_tensor * current = input_tensor;

        // 順次各MTPレイヤーを処理
        for (int layer_idx : layer_indices) {
            if (layer_idx < 0 || layer_idx >= static_cast<int>(model.layers.size())) {
                continue;
            }

            ggml_tensor * layer_result = process_enhanced_mtp_layer(
                ctx0, current, last_token_id, layer_idx, callback
            );
            
            if (layer_result && layer_result != current) {
                current = layer_result;
                callback(current, "mtp_enhanced_layer_success", layer_idx);
            } else {
                callback(current, "mtp_enhanced_layer_unchanged", layer_idx);
            }
        }

        if (config.enable_performance_monitoring) {
            metrics.successful_calls++;
        }

        return current;
    }

    // 既存のAPIとの互換性
    static bool is_enhanced_mtp_available(const llama_model & model) {
        if (model.hparams.nextn_predict_layers == 0) return false;
        
        // 少なくとも1つのMTPレイヤーがenhanced機能を持つかチェック
        const int n_transformer_layers = model.hparams.n_layer - model.hparams.nextn_predict_layers;
        for (int il = n_transformer_layers; il < model.hparams.n_layer; ++il) {
            const auto & nextn = model.layers[il].nextn;
            if (nextn.embed_tokens && nextn.eh_proj && nextn.shared_head_head) {
                return true;
            }
        }
        return false;
    }

    const llama_mtp_perf_metrics & get_metrics() const { return metrics; }
    void reset_metrics() { metrics = llama_mtp_perf_metrics{}; }
};

// 既存のコードとの統合用ヘルパー関数
namespace llama_mtp_integration {

// 既存のMTP処理をenhancedバージョンで置き換える
inline ggml_tensor * process_mtp_with_enhancements(
    const llama_model & model,
    ggml_context * ctx0,
    ggml_tensor * input_tensor,
    llama_token last_token_id,
    const std::function<void(ggml_tensor *, const char *, int)> & cb
) {
    if (!llama_mtp_enhanced_processor::is_enhanced_mtp_available(model)) {
        // フォールバック: 既存のMTP処理を使用
        llama_mtp_processor base_processor(model, llama_mtp_config_default());
        
        // MTPレイヤーのインデックスを収集
        std::vector<int> mtp_layers;
        const int n_transformer_layers = model.hparams.n_layer - model.hparams.nextn_predict_layers;
        for (int il = n_transformer_layers; il < model.hparams.n_layer; ++il) {
            if (model.layers[il].nextn.eh_proj && model.layers[il].nextn.shared_head_head) {
                mtp_layers.push_back(il);
            }
        }
        
        return base_processor.process_mtp_layers(ctx0, input_tensor, mtp_layers, cb);
    }

    // Enhanced MTP処理
    llama_mtp_config enhanced_config = llama_mtp_config_fast();
    enhanced_config.enable_speculative = true;
    enhanced_config.enable_performance_monitoring = true;
    
    llama_mtp_enhanced_processor enhanced_processor(model, enhanced_config);
    
    // MTPレイヤーのインデックスを収集
    std::vector<int> mtp_layers;
    const int n_transformer_layers = model.hparams.n_layer - model.hparams.nextn_predict_layers;
    for (int il = n_transformer_layers; il < model.hparams.n_layer; ++il) {
        mtp_layers.push_back(il);
    }
    
    return enhanced_processor.process_enhanced_mtp_layers(
        ctx0, input_tensor, mtp_layers, last_token_id, cb
    );
}

// 既存のコードパッチ用マクロ
#define LLAMA_MTP_ENHANCED_PROCESS(model, ctx0, input, last_token, cb) \
    llama_mtp_integration::process_mtp_with_enhancements(model, ctx0, input, last_token, cb)

} // namespace llama_mtp_integration