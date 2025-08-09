# Speculative MTP Design (Experimental)

Status: draft / stubs implemented

## Goals
- Reduce average forward passes per generated token below 1.0 using NextN (MTP) outputs
- Integrate with existing speculative decoding path as a draft provider
- Preserve sampling features (penalties, grammar) with minimal divergence

## Current Limitations
- NextN layers expose only concatenated logits, not intermediate hidden/KV states
- Accepting >1 predicted token still requires separate forward passes to build KV -> no real speedup

## Planned Extensions
1. Model output augmentation
   - Export per-predicted-step final hidden or per-layer projections
   - Layout: [n_pred][n_embd] contiguous buffer retrievable via `llama_get_mtp_hidden_ith`
2. Fast accept path
   - `llama_accept_predicted_tokens(ctx, idx, n, tokens)` writes K/V entries for positions idx+1..idx+n
   - Skips re-decoding those steps
3. Verification logic
   - Optional lightweight re-check: confidence margin, grammar DFA pre-check, logit agreement under penalties
   - Fallback decode of first failing token only
4. Adaptive window
   - Maintain EMA of accepted_len; adjust requested MTP n_predict toward target (2–4)
5. Grammar integration
   - Clone DFA state: bulk check predicted tokens; truncate at first reject
6. Metrics
   - tokens_per_forward, accepted_len_avg, margin_histogram

## API Additions (stubs)
- `llama_get_mtp_hidden_ith`
- `llama_accept_predicted_tokens`
- `llama_context_can_speculative_mtp`

All currently return nullptr / 0 / false.

## Integration Flow (future)
```
forward() -> logits + MTP hidden
pred = greedy/logit-margin decode (k tokens)
verify (grammar + penalties preview)
accepted = longest valid prefix
llama_accept_predicted_tokens(..., accepted-1)
if accepted < k: normal forward for next step; else skip accepted-1 forwards
```

## Risks
- Memory overhead for exporting hidden
- KV consistency bugs if partial accept overlaps with context shift
- Divergence under dynamic penalties (repeat, freq)

## Next Steps
- Prototype model side hidden export
- Implement accept path writing KV
- Add unit tests for:
  - Deterministic greedy equality (temp=0) with / without speculative path
  - Grammar rejection boundary
  - Repeat penalty divergence metrics

