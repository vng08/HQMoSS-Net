"""
Tính bảng chỉ số báo cáo từ các file dự đoán numpy đã lưu và vẽ biểu đồ so sánh.

Tương ứng với notebook Report gốc, gồm 3 nhóm phân tích:
  1. Ablation số lớp mạch lượng tử của QuMoE: (1,1,1) ... (5,5,5), kèm CMoE
     (đối chứng cổ điển của cấu hình (5,5,5))                       [PH2]
  2. Ablation đóng góp từng thành phần (QMoE / LSS / RDMS)          [PH2]
  3. So sánh HQMoSS-Net với các baseline U-Net                     [PH2 & GlaS]

Cách dùng:
    python evaluate.py                 # in + lưu toàn bộ bảng CSV vào ./reports
    python evaluate.py --plots         # kèm vẽ biểu đồ
    python evaluate.py --group models --dataset glas
"""

import os
import re
import argparse

import numpy as np
import pandas as pd

import config
from data import load_fold_data
from metrics import compute_metrics

METRIC_COLS = ["soft_iou", "hard_iou", "soft_dice", "hard_dice",
               "accuracy", "sensitivity", "specificity", "precision"]
RENAME_COLS = {
    "model": "Model", "soft_iou": "Soft IoU", "hard_iou": "Hard IoU",
    "soft_dice": "Soft Dice", "hard_dice": "Hard Dice", "accuracy": "Accuracy",
    "sensitivity": "Sensitivity", "specificity": "Specificity", "precision": "Precision",
}

# Số tham số của từng mô hình (dùng cho biểu đồ hiệu quả tham số).
MODELS_PARAM = {
    "UNet": 88209, "HQMoSS-Net": 153660, "UNet++": 637509,
    "Attention UNet": 644263, "VNet (2D)": 1110933, "R2U-Net": 1873581,
    "QUNet (12,1)": 84149,
    # Đối chứng cổ điển của QuMoE(5,5,5) (mục 5.6.2): cùng bộ khung + patch
    # size đa tỷ lệ + router + residual, nhưng expert là MLP cổ điển.
    "QMoE with (5,5,5) Layer": 86996,
    "CMoE with 5 Experts": 1693908,
}


def find_prediction_files(result_dir):
    """Tìm tất cả file predictions_fold*.npy trong một thư mục kết quả."""
    pred_files = {}
    for root, _, files in os.walk(result_dir):
        for file in files:
            m = re.match(r"predictions_fold(\d+)\.npy", file)
            if m:
                pred_files[int(m.group(1))] = os.path.join(root, file)
    return dict(sorted(pred_files.items()))


def evaluate_results(model_dirs, dataset, image_size=(192, 256), threshold=0.5, output_path=None):
    """
    Tính mean ± std các chỉ số qua các fold cho từng mô hình.
    model_dirs: dict {tên hiển thị -> thư mục kết quả}. Trả về DataFrame báo cáo.
    """
    image_dir, mask_dir, fold_dir, kind = config.dataset_paths(dataset)
    all_reports = []

    for model_name, result_dir in model_dirs.items():
        pred_files = find_prediction_files(result_dir)
        results = []
        for fold, pred_path in pred_files.items():
            y_pred = np.load(pred_path)
            _, _, _, val_masks = load_fold_data(fold, image_dir, mask_dir, fold_dir, image_size, kind=kind)
            if len(y_pred) != len(val_masks):
                print(f"Bỏ qua {model_name} fold {fold}: pred={len(y_pred)}, mask={len(val_masks)}")
                continue
            results.append(compute_metrics(val_masks, y_pred, threshold=threshold))

        if not results:
            print(f"Bỏ qua {model_name}: không có kết quả hợp lệ (kiểm tra đường dẫn {result_dir})")
            continue

        df = pd.DataFrame(results)[METRIC_COLS]
        mean_vals, std_vals = df.mean(), df.std(ddof=1).fillna(0.0)
        row = {"model": model_name}
        for col in METRIC_COLS:
            row[col] = f"{mean_vals[col]:.4f} ± {std_vals[col]:.4f}"
        all_reports.append(row)

    df_report = pd.DataFrame(all_reports, columns=["model", *METRIC_COLS]).rename(columns=RENAME_COLS)
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_report.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu: {output_path}")
    return df_report


# ---------------------------------------------------------------------------
# Các nhóm mô hình để báo cáo (xây từ config.result_dir)
# ---------------------------------------------------------------------------
def _rd(dataset, name):
    return config.result_dir(dataset, name)


def qmoe_layer_dirs():
    dirs = {f"QMoE with ({i},{i},{i}) Layer": _rd("ph2", f"QuMoE{i}{i}{i}") for i in range(1, 6)}
    # Đối chứng cổ điển của QMoE(5,5,5): kiến trúc mirror y hệt (cùng patch
    # size đa tỷ lệ + router + residual), chỉ thay PQC lượng tử bằng MLP cổ
    # điển -- cô lập xem lợi ích có đến từ biểu diễn lượng tử hay không.
    dirs["CMoE with 5 Experts"] = _rd("ph2", "CMoE555")
    return dirs


def components_dirs():
    return {
        "UNet": _rd("ph2", "UNet"),
        "UNet + LSS": _rd("ph2", "UNet_LSS"),
        "UNet + RDMS": _rd("ph2", "UNet_RDMS"),
        "UNet + LSS + RDMS": _rd("ph2", "UNet_RDMS_LSS"),
        "QuMoE": _rd("ph2", "QuMoE555"),
        "QuMoE + LSS": _rd("ph2", "QMoE_LSS"),
        "QuMoE + RDMS": _rd("ph2", "QMoE_RDMS"),
        "HQMoSS-Net": _rd("ph2", "HQMoSS_Net"),
        # QuFeX gốc (mô hình tác giả paper ta lấy ý tưởng) làm mốc so sánh.
        "QUNet (12,1)": _rd("ph2", "Qunet_12_1"),
    }


def models_dirs(dataset):
    return {
        "UNet": _rd(dataset, "UNet"),
        "HQMoSS-Net": _rd(dataset, "HQMoSS_Net"),
        "UNet++": _rd(dataset, "UNetPlusPlus"),
        "Attention UNet": _rd(dataset, "AttentionUNet"),
        "VNet (2D)": _rd(dataset, "VNet"),
        "R2U-Net": _rd(dataset, "R2UNet"),
        # QuFeX gốc (baseline lượng tử tham chiếu).
        "QUNet (12,1)": _rd(dataset, "Qunet_12_1"),
    }


# ---------------------------------------------------------------------------
# Vẽ biểu đồ (tuỳ chọn)
# ---------------------------------------------------------------------------
def _mean_soft_iou(df):
    return df["Soft IoU"].str.split("±").str[0].str.strip().astype(float)


def plot_bar(df, labels, title, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    d = df.copy()
    d["Mean Soft IoU"] = _mean_soft_iou(d)
    d["Label"] = labels
    y_min, y_max = d["Mean Soft IoU"].min() - 0.02, d["Mean Soft IoU"].max() + 0.02

    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=d, x="Label", y="Mean Soft IoU", color="skyblue")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=10)
    plt.xlabel("Model"); plt.ylabel("Mean Soft IoU"); plt.title(title)
    plt.ylim(y_min, y_max); plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.show()


def delta_table(df, baseline="UNet", output_path=None):
    """Bảng mức cải thiện Soft IoU so với baseline (mặc định UNet): Delta tuyệt đối
    và phần trăm — dùng cho nhóm ablation thành phần (ứng với bảng delta trong report)."""
    d = df.copy()
    d["Mean Soft IoU"] = _mean_soft_iou(d).round(4)
    base_val = d.loc[d["Model"] == baseline, "Mean Soft IoU"]
    if base_val.empty:
        print(f"Không tìm thấy baseline '{baseline}' trong bảng -> bỏ qua delta table.")
        return None
    base = base_val.iloc[0]
    d["Delta"] = (d["Mean Soft IoU"] - base).round(4)
    d["Percent"] = (d["Delta"] / base * 100).round(2)
    d = d[["Model", "Mean Soft IoU", "Delta", "Percent"]]
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        d.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu: {output_path}")
    return d


def plot_efficiency(df, title, save_path=None):
    """Biểu đồ hiệu quả tham số: số tham số (trục log) vs Soft IoU trung bình,
    mỗi mô hình một điểm. Số tham số lấy từ MODELS_PARAM."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    d = df.copy()
    d["Mean Soft IoU"] = _mean_soft_iou(d)
    d["Params_num"] = d["Model"].map(MODELS_PARAM)
    d = d.dropna(subset=["Params_num"])
    if d.empty:
        print("Không có mô hình nào có số tham số trong MODELS_PARAM -> bỏ qua biểu đồ hiệu quả.")
        return

    markers = ["o", "^", "s", "D", "P", "X", "v", "*"]
    plt.figure(figsize=(12, 5))
    for i, (_, row) in enumerate(d.iterrows()):
        plt.scatter(row["Params_num"], row["Mean Soft IoU"], s=140,
                    marker=markers[i % len(markers)], label=row["Model"])
        plt.text(row["Params_num"], row["Mean Soft IoU"] + 0.0015, row["Model"],
                 fontsize=9, ha="center")
    plt.xscale("log")
    plt.xlabel("Number of Parameters"); plt.ylabel("Mean Soft IoU"); plt.title(title)
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Tính bảng chỉ số báo cáo từ kết quả numpy.")
    p.add_argument("--group", choices=["all", "qmoe", "components", "models"], default="all")
    p.add_argument("--dataset", choices=["glas", "ph2"], default="ph2",
                   help="Dùng cho --group models.")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports"))
    p.add_argument("--plots", action="store_true", help="Vẽ kèm biểu đồ so sánh.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)

    def report(name, model_dirs, dataset):
        print("\n" + "=" * 100 + f"\n{name}\n" + "=" * 100)
        df = evaluate_results(model_dirs, dataset, image_size=config.IMAGE_SIZE,
                              output_path=os.path.join(out_dir, f"{name}.csv"))
        print(df.to_string(index=False))
        return df

    if args.group in ("all", "qmoe"):
        df = report("qmoe_layer_ablation_ph2", qmoe_layer_dirs(), "ph2")
        if args.plots and not df.empty:
            plot_bar(df, [f"{i},{i},{i}" for i in range(1, 6)] + ["CMoE (đối chứng)"],
                     "Soft IoU vs số lớp mạch QMoE (PH2)",
                     os.path.join(out_dir, "qmoe_layer_ablation_ph2.png"))

    if args.group in ("all", "components"):
        df = report("components_ablation_ph2", components_dirs(), "ph2")
        if not df.empty:
            dt = delta_table(df, baseline="UNet",
                             output_path=os.path.join(out_dir, "components_delta_ph2.csv"))
            if dt is not None:
                print("\n-- Mức cải thiện Soft IoU so với UNet --")
                print(dt.to_string(index=False))
        if args.plots and not df.empty:
            plot_bar(df, ["UNet", "U+LSS", "U+RDMS", "U+LSS+RDMS", "QuMoE", "Q+LSS", "Q+RDMS", "HQMoSS-Net", "QUNet (12,1)"],
                     "So sánh đóng góp các thành phần (PH2)",
                     os.path.join(out_dir, "components_ablation_ph2.png"))

    if args.group in ("all", "models"):
        for ds in (["ph2", "glas"] if args.group == "all" else [args.dataset]):
            df = report(f"models_comparison_{ds}", models_dirs(ds), ds)
            if args.plots and not df.empty:
                plot_bar(df, list(models_dirs(ds).keys()),
                         f"So sánh các mô hình ({ds.upper()})",
                         os.path.join(out_dir, f"models_comparison_{ds}.png"))
                plot_efficiency(df, f"Hiệu quả tham số: Params vs Soft IoU ({ds.upper()})",
                                os.path.join(out_dir, f"models_efficiency_{ds}.png"))


if __name__ == "__main__":
    main()
