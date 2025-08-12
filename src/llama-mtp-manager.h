// MTP Performance Manager - パフォーマンス管理と安定化
// Token生成速度の安定化とパフォーマンス監視を行う

#pragma once

#include "llama-mtp-stable.h"
#include <chrono>
#include <deque>
#include <algorithm>

// MTPパフォーマンス管理クラス
class llama_mtp_performance_manager {
private:
    struct performance_record {
        std::chrono::steady_clock::time_point timestamp;
        float success_rate;
        double latency_ms;
        int tokens_generated;
        bool fallback_used;
    };
    
    mutable std::deque<performance_record> recent_records;
    mutable llama_mtp_adaptive_config adaptive_config;
    
    // 履歴サイズの制限
    static constexpr size_t MAX_HISTORY_SIZE = 100;
    static constexpr std::chrono::minutes HISTORY_DURATION{5}; // 5分間の履歴を保持
    
    // パフォーマンス閾値
    static constexpr float MIN_STABLE_SUCCESS_RATE = 0.7f;
    static constexpr double MAX_ACCEPTABLE_LATENCY_MS = 50.0;
    static constexpr double MIN_ACCEPTABLE_LATENCY_MS = 1.0;
    
public:
    llama_mtp_performance_manager() = default;
    
    // パフォーマンス記録を追加
    void record_performance(const llama_mtp_perf_metrics & metrics, bool fallback_used = false) {
        auto now = std::chrono::steady_clock::now();
        
        performance_record record;
        record.timestamp = now;
        record.success_rate = metrics.success_rate();
        record.latency_ms = metrics.avg_latency_ms();
        record.tokens_generated = static_cast<int>(metrics.total_throughput_tokens);
        record.fallback_used = fallback_used;
        
        recent_records.push_back(record);
        
        // 古い記録を削除
        cleanup_old_records();
    }
    
    // 現在の推奨設定を取得
    llama_mtp_config get_recommended_config() const {
        if (recent_records.empty()) {
            return llama_mtp_config_stable();
        }
        
        // 最近のパフォーマンスを分析
        auto analysis = analyze_recent_performance();
        
        // 不安定な場合は保守的な設定を使用
        if (analysis.is_unstable) {
            return llama_mtp_config_consistent();
        }
        
        // 安定している場合は動的調整設定を使用
        if (analysis.is_stable) {
            // 最新のメトリクスを使用して適応的設定を取得
            llama_mtp_perf_metrics latest_metrics;
            latest_metrics.total_calls = 1;
            latest_metrics.successful_calls = analysis.avg_success_rate >= MIN_STABLE_SUCCESS_RATE ? 1 : 0;
            latest_metrics.total_latency_ms = analysis.avg_latency_ms;
            
            return adaptive_config.get_adaptive_config(latest_metrics);
        }
        
        // デフォルトはバランス設定
        return llama_mtp_config_balanced();
    }
    
    // パフォーマンス状態を取得
    struct performance_status {
        bool is_stable = false;
        bool is_unstable = false;
        float avg_success_rate = 0.0f;
        double avg_latency_ms = 0.0;
        float latency_variance = 0.0f;
        int total_records = 0;
        bool requires_fallback = false;
        std::string recommendation;
    };
    
    performance_status get_status() const {
        return analyze_recent_performance();
    }
    
    // パフォーマンス統計をリセット
    void reset() {
        recent_records.clear();
        adaptive_config.reset_to_stable();
    }
    
    // デバッグ用パフォーマンス情報を出力
    void print_status() const {
        auto status = get_status();
        
        printf("=== MTP Performance Status ===\n");
        printf("Records: %d\n", status.total_records);
        printf("Average success rate: %.2f%%\n", status.avg_success_rate * 100.0f);
        printf("Average latency: %.2f ms\n", status.avg_latency_ms);
        printf("Latency variance: %.2f\n", status.latency_variance);
        printf("Stable: %s\n", status.is_stable ? "yes" : "no");
        printf("Unstable: %s\n", status.is_unstable ? "yes" : "no");
        printf("Requires fallback: %s\n", status.requires_fallback ? "yes" : "no");
        printf("Recommendation: %s\n", status.recommendation.c_str());
        printf("==============================\n");
    }
    
private:
    // 古い記録をクリーンアップ
    void cleanup_old_records() const {
        auto now = std::chrono::steady_clock::now();
        
        // 時間ベースのクリーンアップ
        recent_records.erase(
            std::remove_if(recent_records.begin(), recent_records.end(),
                [now](const performance_record& record) {
                    return (now - record.timestamp) > HISTORY_DURATION;
                }),
            recent_records.end()
        );
        
        // サイズベースのクリーンアップ
        while (recent_records.size() > MAX_HISTORY_SIZE) {
            recent_records.pop_front();
        }
    }
    
    // 最近のパフォーマンスを分析
    performance_status analyze_recent_performance() const {
        performance_status status;
        
        if (recent_records.empty()) {
            status.recommendation = "No data available - using stable config";
            return status;
        }
        
        status.total_records = static_cast<int>(recent_records.size());
        
        // 成功率の計算
        float total_success_rate = 0.0f;
        double total_latency = 0.0;
        int fallback_count = 0;
        
        for (const auto& record : recent_records) {
            total_success_rate += record.success_rate;
            total_latency += record.latency_ms;
            if (record.fallback_used) {
                fallback_count++;
            }
        }
        
        status.avg_success_rate = total_success_rate / recent_records.size();
        status.avg_latency_ms = total_latency / recent_records.size();
        status.requires_fallback = (static_cast<size_t>(fallback_count) > recent_records.size() / 2);
        
        // レイテンシの分散を計算
        double latency_variance = 0.0;
        for (const auto& record : recent_records) {
            double diff = record.latency_ms - status.avg_latency_ms;
            latency_variance += diff * diff;
        }
        status.latency_variance = static_cast<float>(latency_variance / recent_records.size());
        
        // 安定性の判定
        bool success_rate_stable = status.avg_success_rate >= MIN_STABLE_SUCCESS_RATE;
        bool latency_acceptable = (status.avg_latency_ms >= MIN_ACCEPTABLE_LATENCY_MS && 
                                 status.avg_latency_ms <= MAX_ACCEPTABLE_LATENCY_MS);
        bool low_variance = status.latency_variance < (status.avg_latency_ms * 0.5); // 50%以下の変動
        
        status.is_stable = success_rate_stable && latency_acceptable && low_variance && !status.requires_fallback;
        status.is_unstable = !success_rate_stable || status.latency_variance > (status.avg_latency_ms * 1.0); // 100%以上の変動
        
        // 推奨事項を設定
        if (status.is_unstable) {
            if (status.avg_success_rate < 0.5f) {
                status.recommendation = "Low success rate - use consistent config";
            } else if (status.latency_variance > status.avg_latency_ms) {
                status.recommendation = "High latency variance - use stable config";
            } else {
                status.recommendation = "Unstable performance - use conservative config";
            }
        } else if (status.is_stable) {
            status.recommendation = "Stable performance - adaptive config available";
        } else {
            status.recommendation = "Moderate performance - use balanced config";
        }
        
        return status;
    }
};

// グローバルMTPマネージャー（オプション）
extern llama_mtp_performance_manager* g_mtp_manager;

// 便利関数
inline llama_mtp_config llama_mtp_get_optimal_config() {
    if (g_mtp_manager) {
        return g_mtp_manager->get_recommended_config();
    }
    return llama_mtp_config_stable(); // デフォルトは安定設定
}

inline void llama_mtp_record_performance(const llama_mtp_perf_metrics & metrics, bool fallback_used = false) {
    if (g_mtp_manager) {
        g_mtp_manager->record_performance(metrics, fallback_used);
    }
}

// 自動パフォーマンス管理マクロ
#define LLAMA_MTP_AUTO_CONFIG() llama_mtp_get_optimal_config()
#define LLAMA_MTP_RECORD_PERF(metrics, fallback) llama_mtp_record_performance(metrics, fallback)