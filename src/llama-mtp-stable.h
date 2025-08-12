// Stable MTP Configuration - 安定性を重視したMTPパラメータ設定
// Token生成速度の安定化とパフォーマンスの一貫性を実現

#pragma once

#include "llama-mtp-optimized.h"

// 普通の運用向けMTP設定 - 実用的なバランス
inline llama_mtp_config llama_mtp_config_stable() {
    llama_mtp_config config;
    
    // 基本パラメータ - 実用的な設定
    config.n_predict_ahead = 4;           // 適度な予測数で効率的
    config.confidence_threshold = 0.65f;  // 実用的な信頼度閾値
    
    // 機能制御 - 実用性重視
    config.enable_speculative = true;     // 投機実行を有効（パフォーマンス向上）
    config.enable_parallel = true;        // 並列処理を有効（効率性重視）
    config.enable_memory_optimization = true;   // メモリ最適化は保持
    config.enable_tensor_fusion = true;   // テンソル融合を有効（パフォーマンス向上）
    config.enable_performance_monitoring = true; // 監視は有効
    
    // 数値パラメータ - 標準的な値
    config.rms_norm_eps = 1e-6f;          // 標準的なイプシロン値
    
    return config;
}

// バランス重視のMTP設定 - 速度と安定性のバランス
inline llama_mtp_config llama_mtp_config_balanced() {
    llama_mtp_config config;
    
    // 基本パラメータ - 中程度の設定でバランス確保
    config.n_predict_ahead = 4;           // 適度な予測数
    config.confidence_threshold = 0.7f;   // バランスの取れた信頼度
    
    // 機能制御 - 安定性に影響の少ない機能のみ有効
    config.enable_speculative = true;     // 控えめな投機実行
    config.enable_parallel = false;       // 並列処理は無効（安定性のため）
    config.enable_memory_optimization = true;   // メモリ最適化は有効
    config.enable_tensor_fusion = true;   // テンソル融合は有効
    config.enable_performance_monitoring = true; // 監視は有効
    
    // 数値パラメータ
    config.rms_norm_eps = 1e-6f;          // 標準的なイプシロン値
    
    return config;
}

// 一貫性重視のMTP設定 - 予測可能なパフォーマンス
inline llama_mtp_config llama_mtp_config_consistent() {
    llama_mtp_config config;
    
    // 基本パラメータ - 一貫したパフォーマンス重視
    config.n_predict_ahead = 2;           // 最小限の予測数で一貫性確保
    config.confidence_threshold = 0.8f;   // 高い信頼度で確実性重視
    
    // 機能制御 - 最小限の機能で一貫性確保
    config.enable_speculative = false;    // 投機実行無効
    config.enable_parallel = false;       // 並列処理無効
    config.enable_memory_optimization = true;   // メモリ最適化のみ有効
    config.enable_tensor_fusion = false;  // テンソル融合無効
    config.enable_performance_monitoring = true; // 監視は有効
    
    // 数値パラメータ - 保守的な値
    config.rms_norm_eps = 1e-6f;          // 標準値
    
    return config;
}

// 動的調整機能付きMTP設定
class llama_mtp_adaptive_config {
private:
    llama_mtp_config base_config;
    mutable float recent_success_rate = 0.0f;
    mutable int adaptation_count = 0;
    
public:
    llama_mtp_adaptive_config() : base_config(llama_mtp_config_stable()) {}
    
    // パフォーマンス履歴に基づいて設定を動的調整
    llama_mtp_config get_adaptive_config(const llama_mtp_perf_metrics & metrics) const {
        llama_mtp_config config = base_config;
        
        // 成功率に基づく調整
        float success_rate = metrics.success_rate();
        
        if (success_rate > 0.9f) {
            // 高成功率: 少し積極的に
            config.n_predict_ahead = std::min(config.n_predict_ahead + 1, 6);
            config.confidence_threshold = std::max(config.confidence_threshold - 0.05f, 0.6f);
        } else if (success_rate < 0.6f) {
            // 低成功率: より保守的に
            config.n_predict_ahead = std::max(config.n_predict_ahead - 1, 1);
            config.confidence_threshold = std::min(config.confidence_threshold + 0.05f, 0.9f);
        }
        
        // レイテンシに基づく調整
        double avg_latency = metrics.avg_latency_ms();
        if (avg_latency > 100.0) {  // 100ms以上は遅すぎる
            config.enable_parallel = false;
            config.enable_tensor_fusion = false;
        } else if (avg_latency < 10.0) {  // 10ms未満は余裕がある
            config.enable_tensor_fusion = true;
        }
        
        return config;
    }
    
    // 基本設定をリセット
    void reset_to_stable() {
        base_config = llama_mtp_config_stable();
        recent_success_rate = 0.0f;
        adaptation_count = 0;
    }
};

// MTP設定の検証機能
namespace llama_mtp_validation {

// 設定の妥当性をチェック
inline bool validate_config(const llama_mtp_config & config) {
    // 基本パラメータの範囲チェック
    if (config.n_predict_ahead < 1 || config.n_predict_ahead > 32) {
        return false;
    }
    
    if (config.confidence_threshold < 0.1f || config.confidence_threshold > 1.0f) {
        return false;
    }
    
    if (config.rms_norm_eps <= 0.0f || config.rms_norm_eps > 1e-3f) {
        return false;
    }
    
    return true;
}

// 安定性のためのパラメータ調整
inline llama_mtp_config sanitize_config(const llama_mtp_config & config) {
    llama_mtp_config sanitized = config;
    
    // パラメータを安全な範囲にクランプ
    sanitized.n_predict_ahead = std::max(1, std::min(sanitized.n_predict_ahead, 8));
    sanitized.confidence_threshold = std::max(0.5f, std::min(sanitized.confidence_threshold, 0.95f));
    sanitized.rms_norm_eps = std::max(1e-8f, std::min(sanitized.rms_norm_eps, 1e-5f));
    
    // 不安定な組み合わせを回避
    if (sanitized.n_predict_ahead > 6) {
        sanitized.enable_parallel = false;  // 高予測数では並列処理を無効
        sanitized.enable_speculative = false; // 投機実行も無効
    }
    
    if (sanitized.confidence_threshold < 0.6f) {
        sanitized.enable_speculative = false; // 低信頼度では投機実行を無効
    }
    
    return sanitized;
}

// デバッグ用設定情報出力
inline void print_config_info(const llama_mtp_config & config) {
    printf("=== MTP Configuration ===\n");
    printf("Predict ahead: %d tokens\n", config.n_predict_ahead);
    printf("Confidence threshold: %.2f\n", config.confidence_threshold);
    printf("Speculative execution: %s\n", config.enable_speculative ? "enabled" : "disabled");
    printf("Parallel processing: %s\n", config.enable_parallel ? "enabled" : "disabled");
    printf("Memory optimization: %s\n", config.enable_memory_optimization ? "enabled" : "disabled");
    printf("Tensor fusion: %s\n", config.enable_tensor_fusion ? "enabled" : "disabled");
    printf("RMS norm epsilon: %.2e\n", config.rms_norm_eps);
    printf("=========================\n");
}

} // namespace llama_mtp_validation

// 推奨設定の選択ヘルパー
inline llama_mtp_config llama_mtp_get_recommended_config(const char* use_case = "default") {
    std::string case_str(use_case);
    
    if (case_str == "stable" || case_str == "production") {
        return llama_mtp_config_stable();
    } else if (case_str == "balanced" || case_str == "default") {
        return llama_mtp_config_balanced();
    } else if (case_str == "consistent" || case_str == "reliable") {
        return llama_mtp_config_consistent();
    } else {
        // 不明な場合は最も安定した設定を返す
        return llama_mtp_config_stable();
    }
}

// 既存コードとの互換性のためのマクロ
#define LLAMA_MTP_STABLE_CONFIG() llama_mtp_config_stable()
#define LLAMA_MTP_BALANCED_CONFIG() llama_mtp_config_balanced()
#define LLAMA_MTP_CONSISTENT_CONFIG() llama_mtp_config_consistent()