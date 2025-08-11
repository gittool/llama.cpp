#include "sampling.h"

#include "common.h"
#include "log.h"

#include <cmath>
#include <unordered_map>
#include <algorithm>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // MTP metrics
    common_mtp_metrics mtp_metrics{}; // default-initialize to avoid -Wmissing-field-initializers

    void set_logits(struct llama_context * ctx, int idx) {
        const auto * logits = llama_get_logits_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        cur.resize(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    struct llama_sampler * grmr;
    if (params.grammar.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<std::string> patterns_anywhere;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    patterns_anywhere.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    patterns_anywhere.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        if (!patterns_anywhere.empty()) {
            trigger_patterns.push_back("^[\\s\\S]*?(" + string_join(patterns_anywhere, "|") + ")[\\s\\S]*");
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        grmr = params.grammar_lazy
             ? llama_sampler_init_grammar_lazy_patterns(vocab, params.grammar.c_str(), "root",
                                                        trigger_patterns_c.data(), trigger_patterns_c.size(),
                                                        trigger_tokens.data(), trigger_tokens.size())
             :      llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
        if (!grmr) {
            return nullptr;
        }
    }

    auto * result = new common_sampler {
        /* .params = */ params,
        /* .grmr   = */ grmr,
        /* .chain  = */ llama_sampler_chain_init(lparams),
        /* .prev   = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };

    llama_sampler_chain_add(result->chain,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                params.logit_bias.size(),
                params.logit_bias.data()));

    if (params.mirostat == 0) {
        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }

                        llama_sampler_chain_add(result->chain, llama_sampler_init_dry      (vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_top_k       (params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_top_p       (params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_top_n_sigma (params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_min_p       (params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_xtc         (params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_typical     (params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_temp_ext    (params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_infill      (vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_penalties   (params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        llama_sampler_chain_add(result->chain, llama_sampler_init_dist(params.seed));
    } else if (params.mirostat == 1) {
        llama_sampler_chain_add(result->chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        llama_sampler_chain_add(result->chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (gsmpl) {
        llama_sampler_free(gsmpl->grmr);

        llama_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    llama_sampler_reset(gsmpl->grmr);

    llama_sampler_reset(gsmpl->chain);
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params = */ gsmpl->params,
        /* .grmr   = */ llama_sampler_clone(gsmpl->grmr),
        /* .chain  = */ llama_sampler_clone(gsmpl->chain),
        /* .prev   = */ gsmpl->prev,
        /* .cur    = */ gsmpl->cur,
        /* .cur_p  = */ gsmpl->cur_p,
    };
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    if (gsmpl) {
        llama_perf_sampler_print(gsmpl->chain);
    }
    if (ctx) {
        llama_perf_context_print(ctx);
    }
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    if (grammar_first) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    const llama_token id = cur_p.data[cur_p.selected].id;

    if (grammar_first) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(grmr,  &cur_p);
    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during re-sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl) {
    return &gsmpl->cur_p;
}

// Initialize MTP state for a model with NextN layers
void common_sampler_init_mtp(struct common_sampler * sampler, int n_predict_tokens) {
    if (!sampler || n_predict_tokens <= 0) {
        return;
    }
    sampler->params.n_predict_tokens = std::max(1, n_predict_tokens);
    sampler->params.mtp_enabled = true;
    LOG_INF("%s: initialized MTP (n_predict_tokens=%d)\n", __func__, sampler->params.n_predict_tokens);
}

// Capability check (considers sampler params & model)
bool common_sampler_can_use_mtp(struct common_sampler * sampler, struct llama_context * ctx) {
    if (!ctx) return false;
    if (sampler && (!sampler->params.mtp_enabled || sampler->params.n_predict_tokens <= 0)) return false;
    if (!llama_context_can_use_mtp(ctx)) return false; // core API check
    const struct llama_model * model = llama_get_model(ctx);
    if (!model) return false;
    const int32_t n_mtp = llama_model_n_mtp_layers(model);
    if (n_mtp <= 0) return false;
    return true;
}

const common_mtp_metrics * common_sampler_get_mtp_metrics(const struct common_sampler * sampler) {
    if (!sampler || !sampler->params.mtp_enabled) return nullptr;
    return &sampler->mtp_metrics;
}

int common_sampler_mtp_adapt(struct common_sampler * sampler) {
    if (!sampler || !sampler->params.mtp_enabled) return 0;
    
    // STABLE MTP ADAPTATION - Less aggressive, more stable adjustments
    const double target = sampler->params.mtp_target_avg_len;
    const double cur    = sampler->mtp_metrics.ema_accept_len;
    const double delta = target - cur;
    int new_n = sampler->params.n_predict_tokens;
    
    // More conservative thresholds to prevent oscillation
    if (delta > 0.5) { // Only adjust when significantly below target
        new_n = std::min(new_n + 2, sampler->params.mtp_max_predict); // Increase by 2 for faster convergence
    } else if (delta < -0.8) { // Only adjust when significantly above target  
        new_n = std::max(new_n - 1, 4); // Decrease more slowly, minimum 4 tokens
    }
    
    // Keep within bounds but prefer higher values for speed
    new_n = std::max(4, std::min(sampler->params.mtp_max_predict, new_n));
    sampler->params.n_predict_tokens = new_n;
    
    return new_n;
}

// Low-level multi-token sampling (no accept) using core API if available
std::vector<llama_token> common_sampler_sample_mtp(
        struct common_sampler * sampler,
        struct llama_context * ctx,
        int idx,
        int n_predict_tokens,
        float acceptance_threshold) {
    std::vector<llama_token> out;
    if (!sampler || !ctx || n_predict_tokens <= 0) return out;

    // Always sample the first token via existing chain to keep behavior consistent
    llama_token first = common_sampler_sample(sampler, ctx, idx);
    if (first == LLAMA_TOKEN_NULL) return out;
    out.push_back(first);

    if (!common_sampler_can_use_mtp(sampler, ctx)) {
        LOG_DBG("%s: MTP not usable -> single token only\n", __func__);
        if (sampler->params.mtp_enabled) {
            sampler->mtp_metrics.calls++;
            sampler->mtp_metrics.tokens_first_only++;
            sampler->mtp_metrics.ema_accept_len = 0.9 * sampler->mtp_metrics.ema_accept_len + 0.1 * 1.0;
        }
        return out;
    }

    // If grammar is active AND not grammar_first, 2nd 以降の MTP 予測は grammar 未検証になるため安全のため今は停止 (TODO: grammar 適用)
    if (sampler->grmr && !sampler->params.grammar.empty()) {
        LOG_DBG("%s: grammar active -> disable multi-token extension (TODO)\n", __func__);
        return out;
    }

    const int want_extra = std::min({n_predict_tokens, sampler->params.n_predict_tokens, sampler->params.mtp_max_predict}) - 1;
    if (want_extra <= 0) return out;

    std::vector<llama_token> buf(want_extra);
    float thr = acceptance_threshold;
    if (sampler->params.mtp_use_margin) {
        // margin モード: 符号付きで API に渡す (負値 = margin threshold)
        thr = -sampler->params.mtp_margin_thresh;
    }
    int32_t predicted = llama_predict_mtp_tokens(ctx, idx, want_extra, thr, buf.data());
    if (predicted > 0) {
        out.insert(out.end(), buf.begin(), buf.begin() + predicted);
    }
    LOG_DBG("%s: produced %zu tokens (requested=%d, extra=%d, accepted_extra=%d)\n", __func__, out.size(), n_predict_tokens, want_extra, predicted);

    if (sampler->params.mtp_enabled) {
        sampler->mtp_metrics.calls++;
        if (predicted == 0) {
            sampler->mtp_metrics.tokens_first_only++;
        } else {
            sampler->mtp_metrics.tokens_extra += predicted;
        }
        const double accept_len = 1.0 + std::max(0, predicted);
        // STABLE EMA: Slower moving average for more stability (0.95 vs 0.9)
        sampler->mtp_metrics.ema_accept_len = 0.95 * sampler->mtp_metrics.ema_accept_len + 0.05 * accept_len;
        // STABLE ADAPTATION: Adapt less frequently for stability (every 64 calls instead of 32)
        if ((sampler->mtp_metrics.calls & 0x3F) == 0) {
            common_sampler_mtp_adapt(sampler);
        }
    }
    return out;
}

std::vector<llama_token> common_sampler_sample_and_accept_mtp(
        struct common_sampler * sampler,
        struct llama_context * ctx,
        int idx,
        int n_predict_tokens,
        float acceptance_threshold,
        bool  grammar_first) {
    auto tokens = common_sampler_sample_mtp(sampler, ctx, idx, n_predict_tokens, acceptance_threshold);
    if (tokens.empty()) return tokens;

    const bool grammar_active = sampler->grmr && !sampler->params.grammar.empty();

    // grammar_first は全候補が grammar 制約下であることを要求するため安全策として multi 生成を抑止
    if (grammar_first && grammar_active) {
        common_sampler_accept(sampler, tokens[0], true);
        tokens.resize(1);
        LOG_DBG("%s: grammar_first enabled -> MTP extra tokens disabled\n", __func__);
        return tokens;
    }

    // 先頭トークンは従来通り grammar 適用
    common_sampler_accept(sampler, tokens[0], true);

    if (!grammar_active) {
        // grammar 無し: 追加トークンをそのまま受理
        for (size_t i = 1; i < tokens.size(); ++i) {
            common_sampler_accept(sampler, tokens[i], false);
        }
        return tokens;
    }

    // grammar 有効: 追加トークンを逐次検証
    std::vector<llama_token> accepted;
    accepted.reserve(tokens.size());
    accepted.push_back(tokens[0]);

    for (size_t i = 1; i < tokens.size(); ++i) {
        llama_token tok = tokens[i];
        llama_token_data single { tok, 1.0f, 0.0f };
        llama_token_data_array arr { &single, 1, -1, false };
        llama_sampler_apply(sampler->grmr, &arr);
        const bool valid = arr.data[0].logit != -INFINITY;
        if (!valid) {
            LOG_DBG("%s: grammar rejected MTP token %d at pos %zu -> stop extension\n", __func__, tok, i);
            break; // 以降は破棄
        }
        // grammar state を前進
        common_sampler_accept(sampler, tok, true);
        accepted.push_back(tok);
    // TODO (DFA fast path): grammar sampler 内部 DFA 状態を複製しベクトル化検証することで
    // ここを O(k) -> O(1) バルク検証に最適化可能。現状は逐次。
    }
    return accepted;
}

// ---------------------------------------------------------------------------
// Sampler type helper implementations (previously missing -> linker errors)
// ---------------------------------------------------------------------------

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_PENALTIES:    return 'R'; // Repeat penalties
        case COMMON_SAMPLER_TYPE_DRY:          return 'D';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:  return 'S';
        case COMMON_SAMPLER_TYPE_TOP_K:        return 'K';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:    return 'Y';
        case COMMON_SAMPLER_TYPE_TOP_P:        return 'P';
        case COMMON_SAMPLER_TYPE_MIN_P:        return 'M';
        case COMMON_SAMPLER_TYPE_XTC:          return 'X';
        case COMMON_SAMPLER_TYPE_TEMPERATURE:  return 'T';
        default: return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_PENALTIES:    return "penalties";
        case COMMON_SAMPLER_TYPE_DRY:          return "dry";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:  return "top-n-sigma";
        case COMMON_SAMPLER_TYPE_TOP_K:        return "top-k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:    return "typical";
        case COMMON_SAMPLER_TYPE_TOP_P:        return "top-p";
        case COMMON_SAMPLER_TYPE_MIN_P:        return "min-p";
        case COMMON_SAMPLER_TYPE_XTC:          return "xtc";
        case COMMON_SAMPLER_TYPE_TEMPERATURE:  return "temperature";
        default: return "unknown";
    }
}

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::vector<enum common_sampler_type> out;
    out.reserve(names.size());
    for (const auto & n : names) {
        if (n == "penalties") out.push_back(COMMON_SAMPLER_TYPE_PENALTIES);
        else if (n == "dry") out.push_back(COMMON_SAMPLER_TYPE_DRY);
        else if (n == "top-n-sigma" || n == "top_n_sigma") out.push_back(COMMON_SAMPLER_TYPE_TOP_N_SIGMA);
        else if (n == "top-k" || n == "top_k") out.push_back(COMMON_SAMPLER_TYPE_TOP_K);
        else if (n == "typical" || n == "typical-p") out.push_back(COMMON_SAMPLER_TYPE_TYPICAL_P);
        else if (n == "top-p" || n == "top_p") out.push_back(COMMON_SAMPLER_TYPE_TOP_P);
        else if (n == "min-p" || n == "min_p") out.push_back(COMMON_SAMPLER_TYPE_MIN_P);
        else if (n == "xtc") out.push_back(COMMON_SAMPLER_TYPE_XTC);
        else if (n == "temperature" || n == "temp") out.push_back(COMMON_SAMPLER_TYPE_TEMPERATURE);
        else if (allow_alt_names) {
            // alternative short names
            if (n == "k") out.push_back(COMMON_SAMPLER_TYPE_TOP_K);
            else if (n == "p") out.push_back(COMMON_SAMPLER_TYPE_TOP_P);
            else if (n == "t") out.push_back(COMMON_SAMPLER_TYPE_TEMPERATURE);
            else if (n == "m") out.push_back(COMMON_SAMPLER_TYPE_MIN_P);
            else if (n == "s") out.push_back(COMMON_SAMPLER_TYPE_TOP_N_SIGMA);
        }
    }
    return out;
}

std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::vector<enum common_sampler_type> out;
    out.reserve(chars.size());
    for (char c : chars) {
        switch (c) {
            case 'R': out.push_back(COMMON_SAMPLER_TYPE_PENALTIES); break;
            case 'D': out.push_back(COMMON_SAMPLER_TYPE_DRY); break;
            case 'S': out.push_back(COMMON_SAMPLER_TYPE_TOP_N_SIGMA); break;
            case 'K': out.push_back(COMMON_SAMPLER_TYPE_TOP_K); break;
            case 'Y': out.push_back(COMMON_SAMPLER_TYPE_TYPICAL_P); break;
            case 'P': out.push_back(COMMON_SAMPLER_TYPE_TOP_P); break;
            case 'M': out.push_back(COMMON_SAMPLER_TYPE_MIN_P); break;
            case 'X': out.push_back(COMMON_SAMPLER_TYPE_XTC); break;
            case 'T': out.push_back(COMMON_SAMPLER_TYPE_TEMPERATURE); break;
            default: break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Additional helper implementations required by main.cpp (link errors)
// ---------------------------------------------------------------------------

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    if (!gsmpl || gsmpl->prev.empty()) return LLAMA_TOKEN_NULL;
    // Access via size()-1 using operator[] semantics emulation
    // Since ring_buffer is internal, we replicate minimal safe logic
    const size_t sz = gsmpl->prev.size();
    const size_t last_rel = sz - 1;
    const size_t real_index = (gsmpl->prev.first + last_rel) % gsmpl->prev.capacity;
    return gsmpl->prev.data[real_index];
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    if (!gsmpl) return {};
    std::string out;
    out.reserve(64);
    out += '[';
    bool first = true;
    for (auto t : gsmpl->params.samplers) {
        if (!first) out += ' ';
        out += common_sampler_type_to_chr(t);
        first = false;
    }
    out += ']';
    if (gsmpl->params.mtp_enabled) {
        char buf[64];
        snprintf(buf, sizeof(buf), " MTP(n=%d,avg=%.2f)", gsmpl->params.n_predict_tokens, (float)gsmpl->mtp_metrics.ema_accept_len);
        out += buf;
    }
    return out;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n) {
    if (!gsmpl || n <= 0 || gsmpl->prev.empty()) return {};
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string out;
    n = std::min<int>(n, (int)gsmpl->prev.size());
    out.reserve(n * 6);
    // iterate last n tokens
    for (int i = (int)gsmpl->prev.size() - n; i < (int)gsmpl->prev.size(); ++i) {
        size_t idx = (gsmpl->prev.first + i) % gsmpl->prev.capacity;
        llama_token tok = gsmpl->prev.data[idx];
        out += llama_vocab_get_text(vocab, tok);
    }
    return out;
}
