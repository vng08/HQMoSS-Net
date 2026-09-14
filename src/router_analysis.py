"""
Phân tích cân bằng định tuyến (routing load balance) của khối QuantumMoE.

Đo lường xem ba nhánh xử lý lượng tử (expert) có được mạng định tuyến (router)
phân bổ tải tương đối đồng đều hay không -- tức kiểm chứng định lượng cho giả
thuyết thiết kế "tránh sụp đổ nhánh xử lý" (Expert Collapse, xem Chương 3 khóa
luận) -- bằng cách forward-pass mô hình đã huấn luyện qua tập kiểm định và
phân tích phân bố trọng số softmax của router theo từng điểm ảnh.

Đây là phân tích chẩn đoán trên MỘT checkpoint đại diện (một fold cụ thể có
lưu trọng số), không phải trung bình nhiều fold; số liệu cần được diễn giải
kèm giới hạn này.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import mannwhitneyu

EPS = 1e-8


def get_router_probe(model, layer_name="quantum_moe_router_softmax"):
    """Trích mô hình con trả về đầu ra softmax của router QuantumMoE."""
    return tf.keras.Model(model.inputs, model.get_layer(layer_name).output)


def compute_router_balance(router_probe, X):
    """
    Forward-pass X qua router_probe, trả về (summary, per_image_df).

    summary: tỷ trọng trung bình mỗi nhánh, entropy trung bình (và % so với
        entropy tối đa log(M)), imbalance_range (chênh lệch nhánh cao nhất -
        thấp nhất) và hệ số biến thiên (CV) giữa các nhánh.
    per_image_df: tỷ trọng trung bình + entropy trung bình của từng ảnh, dùng
        để so sánh biến thiên giữa các ảnh hoặc giữa hai tập dữ liệu bằng
        kiểm định phi tham số (xem compare_router_balance).
    """
    router_w = router_probe.predict(X, verbose=0)  # (N, H, W, n_experts)
    n_experts = router_w.shape[-1]
    max_entropy = np.log(n_experts)

    expert_means = router_w.mean(axis=(0, 1, 2))
    entropy_map = -np.sum(router_w * np.log(router_w + EPS), axis=-1)  # (N, H, W)

    summary = {
        "expert_means": expert_means,
        "entropy_mean": float(entropy_map.mean()),
        "entropy_std": float(entropy_map.std()),
        "entropy_pct_of_max": float(100 * entropy_map.mean() / max_entropy),
        "imbalance_range": float(expert_means.max() - expert_means.min()),
        "cv_across_experts": float(expert_means.std() / (expert_means.mean() + EPS)),
        "max_entropy": float(max_entropy),
        "n_experts": int(n_experts),
    }

    per_image_expert_means = router_w.mean(axis=(1, 2))  # (N, n_experts)
    per_image_entropy_mean = entropy_map.mean(axis=(1, 2))  # (N,)
    per_image_df = pd.DataFrame({
        "image_idx": np.arange(len(X)),
        **{f"expert{i + 1}_mean_weight": per_image_expert_means[:, i] for i in range(n_experts)},
        "entropy_mean": per_image_entropy_mean,
        "entropy_pct_of_max": 100 * per_image_entropy_mean / max_entropy,
    })
    return summary, per_image_df


def print_router_summary(summary, per_image_df, tag="model"):
    """In tóm tắt cân bằng định tuyến theo đúng định dạng dùng trong phân tích."""
    n_images = len(per_image_df)
    print(f"\n=== Cân bằng định tuyến — {tag} ({n_images} ảnh) ===")
    print(f"Tỷ trọng trung bình mỗi nhánh : {np.round(summary['expert_means'], 4)} "
          f"(lý tưởng cân bằng: {1 / summary['n_experts']:.4f} mỗi nhánh)")
    print(f"Entropy định tuyến             : {summary['entropy_mean']:.4f} ± {summary['entropy_std']:.4f} "
          f"({summary['entropy_pct_of_max']:.1f}% của max={summary['max_entropy']:.4f})")
    print(f"imbalance_range (max-min nhánh): {summary['imbalance_range']:.4f}")
    print(f"cv_across_experts              : {summary['cv_across_experts']:.4f}")
    for i in range(summary["n_experts"]):
        col = f"expert{i + 1}_mean_weight"
        print(f"  {col:25s}: {per_image_df[col].mean():.4f} ± {per_image_df[col].std():.4f} (giữa các ảnh)")


def compare_router_balance(df_a, df_b, label_a="A", label_b="B"):
    """
    So sánh phân bố tỷ trọng mỗi nhánh giữa hai tập ảnh (ví dụ hai bộ dữ liệu)
    bằng kiểm định Mann-Whitney U (hai mẫu độc lập, không giả định phân phối
    chuẩn -- phù hợp vì hai tập ảnh validation là các mẫu độc lập, không ghép
    cặp), kèm hệ số tương quan rank-biserial làm cỡ hiệu ứng (effect size).

    Trả về DataFrame gồm trung bình mỗi nhóm, giá trị p và rank-biserial r
    cho từng nhánh xử lý.
    """
    expert_cols = [c for c in df_a.columns if c.startswith("expert") and c.endswith("_mean_weight")]
    n1, n2 = len(df_a), len(df_b)
    rows = []
    for col in expert_cols:
        stat, p = mannwhitneyu(df_a[col], df_b[col], alternative="two-sided")
        r = 1 - (2 * stat) / (n1 * n2)
        rows.append({
            "expert": col.replace("_mean_weight", ""),
            f"mean_{label_a}": df_a[col].mean(),
            f"mean_{label_b}": df_b[col].mean(),
            "p_value": p,
            "rank_biserial_r": r,
        })
    return pd.DataFrame(rows)
