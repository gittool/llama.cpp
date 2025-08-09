#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if accept_grammar is true, the token is accepted both by the sampling chain and the grammar
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

//
// Multi-Token Prediction (MTP) functions
//

// Initialize MTP state for a model with NextN layers
void common_sampler_init_mtp(struct common_sampler * sampler, int n_predict_tokens);

// Sample multiple tokens using MTP (Multi-Token Prediction)
// acceptance_threshold:
//   - 通常: 0.0-1.0 の確率しきい値
//   - margin モード有効時 (mtp_use_margin=true): 無視され内部で負値に変換され logit margin 閾値として扱われる
std::vector<llama_token> common_sampler_sample_mtp(
        struct common_sampler * sampler,
        struct llama_context * ctx,
        int idx,
        int n_predict_tokens,
        float acceptance_threshold);

// Sample & 受理 (accept) まで行うユーティリティ
//  - 先頭トークン: 通常サンプリングチェーン + (必要なら) grammar
//  - 2個目以降: MTP 予測 (grammar が有効な場合は安全のため停止)
// 返り値: 実際に accept 済みのトークン列 (少なくとも1個)
std::vector<llama_token> common_sampler_sample_and_accept_mtp(
        struct common_sampler * sampler,
        struct llama_context * ctx,
        int idx,
        int n_predict_tokens,
        float acceptance_threshold,
        bool  grammar_first = false);

// Check if MTP should be enabled based on model architecture
// サンプラパラメータ (enabled / grammar 無効 等) も考慮した包括的判定
bool common_sampler_can_use_mtp(struct common_sampler * sampler, struct llama_context * ctx);

// MTP runtime metrics (軽量集計)
struct common_mtp_metrics {
        uint64_t calls              = 0;   // number of MTP attempts
        uint64_t tokens_first_only  = 0;   // only first token produced (fallback)
        uint64_t tokens_extra       = 0;   // extra tokens predicted (accepted)
        double   ema_accept_len     = 1.0; // EMA of accepted tokens per forward
};

// 収集された metrics を取得 (nullptr なら空)
const common_mtp_metrics * common_sampler_get_mtp_metrics(const struct common_sampler * sampler);

// MTP 動的調整: 内部で n_predict_tokens を最適化 (戻り値: 現在値)
int common_sampler_mtp_adapt(struct common_sampler * sampler);

// 既存互換: 古い呼び出し箇所向け (将来削除予定)
inline bool common_sampler_can_use_mtp(struct llama_context * ctx) { return common_sampler_can_use_mtp(nullptr, ctx); }
