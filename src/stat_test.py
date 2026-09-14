"""
Kiểm định thống kê chính thức cho khóa luận: HQMoSS-Net vs. mọi baseline/ablation.

KHÔNG huấn luyện lại -- chỉ dùng lại các file ``predictions_fold{1..5}.npy`` đã
lưu trong ``Reuslt/<PH2|GlaS>/<model>/`` (giống evaluate.py), rồi tính chỉ số
Soft IoU *pooled* (gộp toàn bộ ảnh của một fold thành một giá trị -- đúng công
thức Soft IoU của bảng HEADLINE ở evaluate.py/metrics.py) làm đơn vị so sánh
mức fold (n = 5).

Pipeline 7 thành phần (khớp notebook thống kê đã chạy trên Kaggle):
  1. Bootstrap 95% CI (percentile, 10000 resample)  -- BẰNG CHỨNG CHÍNH
  2. Shapiro-Wilk (kiểm tra chuẩn của hiệu số, minh bạch quy trình)
  3. Paired one-sided t-test (giả thuyết a priori: HQMoSS > baseline)
  4. Wilcoxon signed-rank (tham khảo, phi tham số, không dùng để kết luận)
  5. Cohen's d (cỡ hiệu ứng)
  6. Win-rate mức ảnh (mô tả, KHÔNG kèm p-value)
  7. Meta-analysis Stouffer's Z (gộp p one-sided mức fold PH2 + GlaS)

Module này KHÔNG cần TensorFlow (chỉ numpy/scipy/pandas/Pillow): nó đọc lại
predictions đã lưu và tự nạp mask bằng Pillow, nên chạy được ngay tại máy trên
các file .npy đã tải về. Công thức Soft IoU pooled và cách nạp/nhị phân mask
được giữ TRÙNG KHỚP với ``metrics.compute_metrics`` và ``data.load_fold_data``
để số liệu khớp bảng báo cáo HEADLINE (evaluate.py).

Cách dùng:
    python stat_test.py                    # chạy cả PH2 + GlaS, lưu CSV vào ../reports/stat_tests
    python stat_test.py --dataset ph2      # chỉ một bộ dữ liệu
    python stat_test.py --out-dir ./out --n-bootstrap 10000 --seed 42
"""

import os
import re
import sys
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image

import config

EPS = 1e-7
RNG_SEED = 42
N_BOOTSTRAP = 10000
MAIN_MODEL = "HQMoSS_Net"          # tên thư mục của mô hình đề xuất

# ---------------------------------------------------------------------------
# 1. Danh mục mô hình theo TÊN THƯ MỤC trong Reuslt/<PH2|GlaS>/ (khớp evaluate.py)
# ---------------------------------------------------------------------------
DATASET_MODELS = {
    "ph2": ["AttentionUNet", "HQMoSS_Net", "QMoE_LSS", "QMoE_RDMS",
            "QuMoE111", "QuMoE222", "QuMoE333", "QuMoE444", "QuMoE555",
            "Qunet_12_1", "R2UNet", "UNet", "UNetPlusPlus",
            "UNet_LSS", "UNet_RDMS", "UNet_RDMS_LSS", "VNet", "CMoE555"],
    "glas": ["AttentionUNet", "HQMoSS_Net", "Qunet_12_1", "R2UNet",
             "UNet", "UNetPlusPlus", "VNet"],
}

# Baseline thuộc nhóm "ablation" (bóc tách thành phần) -- phần còn lại là
# "architecture" (đối chứng với các kiến trúc U-Net khác).
ABLATION_MODELS = {
    "ph2": {"QMoE_LSS", "QMoE_RDMS", "QuMoE111", "QuMoE222", "QuMoE333",
            "QuMoE444", "QuMoE555", "UNet_LSS", "UNet_RDMS", "UNet_RDMS_LSS",
            "CMoE555"},
    "glas": set(),
}

# Tên hiển thị cho bảng thesis (thiếu -> dùng luôn tên thư mục).
DISPLAY_NAMES = {
    "UNet": "U-Net", "HQMoSS_Net": "HQMoSS-Net", "UNetPlusPlus": "UNet++",
    "AttentionUNet": "Attention U-Net", "VNet": "V-Net (2D)", "R2UNet": "R2U-Net",
    "Qunet_12_1": "QUNet (12,1)", "CMoE555": "CMoE (5 experts)",
    "QuMoE111": "QMoE (1,1,1)", "QuMoE222": "QMoE (2,2,2)", "QuMoE333": "QMoE (3,3,3)",
    "QuMoE444": "QMoE (4,4,4)", "QuMoE555": "QMoE (5,5,5)",
    "QMoE_LSS": "QuMoE + LSS", "QMoE_RDMS": "QuMoE + RDMS",
    "UNet_LSS": "UNet + LSS", "UNet_RDMS": "UNet + RDMS",
    "UNet_RDMS_LSS": "UNet + LSS + RDMS",
}


# ---------------------------------------------------------------------------
# 2. Nạp mask (Pillow) -- TRÙNG quy ước của data.load_fold_data:
#    lặp theo TÊN ẢNH đã sort, lọc theo tập val, suy tên mask theo kind, resize
#    NEAREST về IMAGE_SIZE, nhị phân hoá (ph2: /255 > 0.5; glas: > 0).
# ---------------------------------------------------------------------------
def _mask_name(img_filename, kind):
    name, ext = os.path.splitext(img_filename)
    if kind == "glas":
        return name + "_anno" + ext
    if kind == "ph2":
        return img_filename
    raise ValueError(f"Unknown dataset kind: {kind}")


def _load_mask(path, kind, image_size):
    h, w = image_size
    img = Image.open(path).convert("L").resize((w, h), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32)[..., None]  # (H, W, 1), 0..255
    if kind == "ph2":
        return (arr / 255.0 > 0.5).astype(np.float32)
    return (arr > 0).astype(np.float32)  # glas


# Mask validation của mỗi (dataset, fold) giống nhau với mọi mô hình -> cache.
_MASK_CACHE = {}


def val_masks_for_fold(dataset, fold):
    """Mask val của một fold theo ĐÚNG thứ tự predictions được lưu (data.py):
    sort tên ảnh, lọc theo fold_{fold}_val.csv."""
    key = (dataset, fold)
    if key not in _MASK_CACHE:
        image_dir, mask_dir, fold_dir, kind = config.dataset_paths(dataset)
        val_df = pd.read_csv(os.path.join(fold_dir, f"fold_{fold}_val.csv"))
        val_img_files = set(val_df["image_filename"].values)
        image_filenames = sorted(f for f in os.listdir(image_dir) if not f.startswith("."))
        masks = [
            _load_mask(os.path.join(mask_dir, _mask_name(f, kind)), kind, config.IMAGE_SIZE)
            for f in image_filenames if f in val_img_files
        ]
        _MASK_CACHE[key] = np.array(masks)
    return _MASK_CACHE[key]


# ---------------------------------------------------------------------------
# 3. IoU: pooled (mức fold, cho kiểm định) và per-image (chỉ cho win-rate)
# ---------------------------------------------------------------------------
def pooled_soft_iou(y_true, y_pred):
    """Soft IoU gộp toàn bộ batch của một fold -- TRÙNG công thức soft_iou trong
    metrics.compute_metrics (pooled, trên xác suất, EPS=1e-7)."""
    yt = y_true.astype(np.float32).reshape(-1)
    yp = np.clip(y_pred.astype(np.float32), 0, 1).reshape(-1)
    inter = float(np.sum(yt * yp))
    union = float(np.sum(yt) + np.sum(yp) - inter)
    return (inter + EPS) / (union + EPS)


def soft_iou_per_image(y_true, y_pred):
    """Soft IoU riêng từng ảnh -> mảng (N,). Chỉ dùng cho win-rate mô tả."""
    yt = y_true.astype(np.float32).reshape(y_true.shape[0], -1)
    yp = np.clip(y_pred.astype(np.float32), 0, 1).reshape(y_pred.shape[0], -1)
    inter = (yt * yp).sum(axis=1)
    union = yt.sum(axis=1) + yp.sum(axis=1) - inter
    return (inter + EPS) / (union + EPS)


def find_prediction_files(result_dir):
    """Tìm mọi predictions_fold*.npy trong thư mục kết quả (giống evaluate.py,
    tự lặp thư mục con nếu có)."""
    pred_files = {}
    for root, _, files in os.walk(result_dir):
        for file in files:
            m = re.match(r"predictions_fold(\d+)\.npy", file)
            if m:
                pred_files[int(m.group(1))] = os.path.join(root, file)
    return dict(sorted(pred_files.items()))


def collect_iou(dataset, model_name, k_folds=config.K_FOLDS):
    """Trả về (pooled_by_fold, per_image_all) cho một mô hình.
       pooled_by_fold: list k giá trị pooled Soft IoU (1 số/fold).
       per_image_all : mảng gộp per-image Soft IoU của mọi fold."""
    pred_files = find_prediction_files(config.result_dir(dataset, model_name))
    if len(pred_files) < k_folds:
        raise FileNotFoundError(
            f"{dataset}/{model_name}: chỉ tìm thấy {len(pred_files)}/{k_folds} "
            f"file predictions_fold*.npy")
    pooled_by_fold, per_image_all = [], []
    for fold in range(1, k_folds + 1):
        y_pred = np.load(pred_files[fold])
        y_val = val_masks_for_fold(dataset, fold)
        if len(y_pred) != len(y_val):
            raise ValueError(f"{dataset}/{model_name} fold{fold}: số ảnh lệch -- "
                             f"pred {len(y_pred)} vs mask {len(y_val)}")
        pooled_by_fold.append(pooled_soft_iou(y_val, y_pred))
        per_image_all.append(soft_iou_per_image(y_val, y_pred))
    return pooled_by_fold, np.concatenate(per_image_all)


# ---------------------------------------------------------------------------
# 3. Các phép kiểm định thống kê
# ---------------------------------------------------------------------------
def bootstrap_ci_fold_diff(fold_a, fold_b, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """a = HQMoSS, b = baseline. Bootstrap CI 95% (percentile) cho mean(a - b)."""
    diff = np.asarray(fold_a) - np.asarray(fold_b)
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(diff, size=n, replace=True).mean()
                           for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return diff.mean(), lo, hi, bool(lo > 0 or hi < 0)


def paired_compare_full(fold_a, fold_b):
    """Shapiro-Wilk (trên hiệu số), t-test ghép cặp one-sided, Wilcoxon (tham
    khảo), Cohen's d."""
    diff = np.asarray(fold_a) - np.asarray(fold_b)

    if np.std(diff) > 0:
        _, sw_p = stats.shapiro(diff)
    else:
        sw_p = np.nan

    _, p_two = stats.ttest_rel(fold_a, fold_b)
    t_p_onesided = p_two / 2 if diff.mean() > 0 else 1 - p_two / 2

    sd = diff.std(ddof=1)
    cohens_d = diff.mean() / sd if sd > 0 else 0.0

    if np.any(diff != 0):
        _, w_p_twosided = stats.wilcoxon(fold_a, fold_b, zero_method="wilcox",
                                         alternative="two-sided")
    else:
        w_p_twosided = 1.0

    return {"shapiro_p": sw_p, "t_p_onesided": t_p_onesided,
            "cohens_d": cohens_d, "w_p_twosided_ref_only": w_p_twosided}


def stouffer_combine(pvalues_onesided, weights=None):
    """Gộp p-value one-sided ĐỘC LẬP, CÙNG HƯỚNG giả thuyết bằng Stouffer's Z."""
    p = np.clip(np.asarray(pvalues_onesided, dtype=float), 1e-300, 1 - 1e-16)
    z = stats.norm.isf(p)
    weights = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    z_combined = np.sum(weights * z) / np.sqrt(np.sum(weights ** 2))
    return z_combined, stats.norm.sf(z_combined)


# ---------------------------------------------------------------------------
# 4. Thu thập IoU cho toàn bộ mô hình của các bộ dữ liệu
# ---------------------------------------------------------------------------
def collect_all_iou(datasets):
    iou_data = {}
    for dataset in datasets:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset.upper()}")
        print("=" * 100)
        for m in DATASET_MODELS[dataset]:
            try:
                pooled, per_img = collect_iou(dataset, m)
                iou_data[(dataset, m)] = {"pooled": pooled, "per_image": per_img}
                print(f"  {m:16s}: {np.round(pooled, 4)}  (mean={np.mean(pooled):.4f})")
            except (FileNotFoundError, ValueError) as e:
                print(f"  !! Bỏ qua {m}: {e}")
    return iou_data


# ---------------------------------------------------------------------------
# 5. Kiểm định mức fold: HQMoSS vs mọi baseline
# ---------------------------------------------------------------------------
def build_fold_level_table(iou_data, datasets, n_boot, seed):
    rows = []
    for dataset in datasets:
        if (dataset, MAIN_MODEL) not in iou_data:
            print(f"!! Không có {MAIN_MODEL} trong {dataset}, bỏ qua so sánh.")
            continue
        main_pooled = np.array(iou_data[(dataset, MAIN_MODEL)]["pooled"])
        baselines = [m for (d, m) in iou_data if d == dataset and m != MAIN_MODEL]
        ablation_set = ABLATION_MODELS.get(dataset, set())

        for baseline in baselines:
            base_pooled = np.array(iou_data[(dataset, baseline)]["pooled"])
            mean_diff, ci_lo, ci_hi, ci_excl0 = bootstrap_ci_fold_diff(
                main_pooled, base_pooled, n_boot=n_boot, seed=seed)
            extra = paired_compare_full(main_pooled, base_pooled)
            rows.append({
                "dataset": dataset.upper(),
                "group": "ablation" if baseline in ablation_set else "architecture",
                "baseline": DISPLAY_NAMES.get(baseline, baseline),
                "mean_HQMoSS": main_pooled.mean(), "mean_baseline": base_pooled.mean(),
                "mean_diff": mean_diff, "mean_diff_pct_points": mean_diff * 100,
                "ci_lower": ci_lo, "ci_upper": ci_hi, "ci_excludes_zero": ci_excl0,
                "shapiro_p": extra["shapiro_p"], "t_p_onesided": extra["t_p_onesided"],
                "cohens_d": extra["cohens_d"],
                "w_p_twosided_ref_only": extra["w_p_twosided_ref_only"],
            })
    return pd.DataFrame(rows).sort_values(["dataset", "group", "t_p_onesided"])


# ---------------------------------------------------------------------------
# 6. Meta-analysis gộp PH2 + GlaS (chỉ baseline chung cả hai bộ)
# ---------------------------------------------------------------------------
def build_meta_table(df_fold, k_folds=config.K_FOLDS):
    ph2 = set(df_fold.query("dataset=='PH2'").baseline)
    glas = set(df_fold.query("dataset=='GLAS'").baseline)
    common = ph2 & glas
    rows = []
    for baseline in sorted(common):
        r1 = df_fold.query("dataset=='PH2' and baseline==@baseline").iloc[0]
        r2 = df_fold.query("dataset=='GLAS' and baseline==@baseline").iloc[0]
        z_c, p_c = stouffer_combine([r1.t_p_onesided, r2.t_p_onesided],
                                    weights=[np.sqrt(k_folds), np.sqrt(k_folds)])
        rows.append({
            "baseline": baseline,
            "diff_PH2_pct_points": r1.mean_diff * 100, "p_PH2": r1.t_p_onesided,
            "diff_GLAS_pct_points": r2.mean_diff * 100, "p_GLAS": r2.t_p_onesided,
            "Z_combined": z_c, "p_combined": p_c,
        })
    return pd.DataFrame(rows).sort_values("p_combined") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 7. Win-rate mức ảnh (mô tả)
# ---------------------------------------------------------------------------
def build_winrate_table(iou_data, datasets):
    rows = []
    for dataset in datasets:
        if (dataset, MAIN_MODEL) not in iou_data:
            continue
        main_img = iou_data[(dataset, MAIN_MODEL)]["per_image"]
        baselines = [m for (d, m) in iou_data if d == dataset and m != MAIN_MODEL]
        for baseline in baselines:
            base_img = iou_data[(dataset, baseline)]["per_image"]
            n_images = len(main_img)
            n_win = int((main_img > base_img).sum())
            rows.append({
                "dataset": dataset.upper(),
                "baseline": DISPLAY_NAMES.get(baseline, baseline),
                "n_images": n_images, "n_win": n_win,
                "win_rate_pct": 100.0 * n_win / n_images,
            })
    return pd.DataFrame(rows).sort_values(["dataset", "win_rate_pct"],
                                          ascending=[True, False])


# ---------------------------------------------------------------------------
# 8. Bảng rút gọn cho khóa luận
# ---------------------------------------------------------------------------
def _fmt_p(p):
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def make_concise_fold_table(df):
    c = df[["dataset", "group", "baseline", "mean_HQMoSS", "mean_baseline",
            "mean_diff_pct_points", "ci_lower", "ci_upper", "ci_excludes_zero",
            "cohens_d", "t_p_onesided"]].copy()
    c["mean_HQMoSS"] = c["mean_HQMoSS"].round(3)
    c["mean_baseline"] = c["mean_baseline"].round(3)
    c["mean_diff_pct_points"] = c["mean_diff_pct_points"].round(2)
    c["95%_CI"] = c.apply(lambda r: f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]", axis=1)
    c["cohens_d"] = c["cohens_d"].round(2)
    c["t_p_onesided"] = c["t_p_onesided"].apply(_fmt_p)
    c = c.drop(columns=["ci_lower", "ci_upper"]).rename(columns={
        "mean_HQMoSS": "IoU_HQMoSS", "mean_baseline": "IoU_baseline",
        "mean_diff_pct_points": "Delta (diem %)", "ci_excludes_zero": "CI loai tru 0",
        "cohens_d": "Cohen's d", "t_p_onesided": "p (1-sided)"})
    return c[["dataset", "group", "baseline", "IoU_HQMoSS", "IoU_baseline",
              "Delta (diem %)", "95%_CI", "CI loai tru 0", "Cohen's d", "p (1-sided)"]]


def make_concise_meta_table(df):
    if df.empty:
        return df
    c = df.copy()
    c["diff_PH2_pct_points"] = c["diff_PH2_pct_points"].round(2)
    c["diff_GLAS_pct_points"] = c["diff_GLAS_pct_points"].round(2)
    for col in ("p_PH2", "p_GLAS", "p_combined"):
        c[col] = c[col].apply(_fmt_p)
    return c.drop(columns=["Z_combined"]).rename(columns={
        "diff_PH2_pct_points": "Delta_PH2 (diem %)", "diff_GLAS_pct_points": "Delta_GLAS (diem %)",
        "p_PH2": "p_PH2 (1-sided)", "p_GLAS": "p_GLAS (1-sided)",
        "p_combined": "p_combined (Stouffer)"})


# ---------------------------------------------------------------------------
# 9. Chạy toàn bộ pipeline
# ---------------------------------------------------------------------------
def _save(df, out_dir, name):
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {path}")


def run(datasets, out_dir, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    os.makedirs(out_dir, exist_ok=True)
    iou_data = collect_all_iou(datasets)

    df_fold = build_fold_level_table(iou_data, datasets, n_boot, seed)
    _save(df_fold, out_dir, "stat_test_fold_level_pooled.csv")
    print("\n" + "=" * 100)
    print("KẾT QUẢ ĐẦY ĐỦ (MỨC FOLD, n=5, POOLED Soft IoU) -- LƯU CHO PHỤ LỤC")
    print("=" * 100)
    cols = ["baseline", "mean_HQMoSS", "mean_baseline", "mean_diff_pct_points",
            "ci_lower", "ci_upper", "ci_excludes_zero", "shapiro_p", "t_p_onesided",
            "cohens_d", "w_p_twosided_ref_only"]
    for (dataset, group), sub in df_fold.groupby(["dataset", "group"]):
        print(f"\n--- {dataset} / {group} (n={len(sub)} baseline) ---")
        print(sub[cols].to_string(index=False))

    df_meta = build_meta_table(df_fold)
    _save(df_meta, out_dir, "meta_analysis_fold_level_pooled.csv")
    print("\n" + "=" * 100)
    print("META-ANALYSIS (gộp PH2 + GlaS, mức fold, pooled Soft IoU, Stouffer's Z)")
    print("=" * 100)
    print(df_meta.to_string(index=False) if not df_meta.empty else "(không có baseline chung)")

    df_winrate = build_winrate_table(iou_data, datasets)
    _save(df_winrate, out_dir, "descriptive_winrate_per_image.csv")
    print("\n" + "=" * 100)
    print("THỐNG KÊ MÔ TẢ MỨC ẢNH (win-rate) -- chỉ minh họa, KHÔNG suy diễn thống kê")
    print("=" * 100)
    print(df_winrate.to_string(index=False))

    df_concise = make_concise_fold_table(df_fold).sort_values(
        ["dataset", "group", "p (1-sided)"])
    _save(df_concise, out_dir, "stat_test_CONCISE_for_thesis.csv")
    print("\n" + "=" * 100)
    print("BẢNG RÚT GỌN CHO KHÓA LUẬN (Chương 5)")
    print("=" * 100)
    for (dataset, group), sub in df_concise.groupby(["dataset", "group"]):
        print(f"\n--- {dataset} / {group} ---")
        print(sub.drop(columns=["dataset", "group"]).to_string(index=False))

    df_meta_concise = make_concise_meta_table(df_meta)
    _save(df_meta_concise, out_dir, "meta_analysis_CONCISE_for_thesis.csv")
    print("\n" + "=" * 100)
    print("META-ANALYSIS RÚT GỌN CHO KHÓA LUẬN")
    print("=" * 100)
    print(df_meta_concise.to_string(index=False) if not df_meta_concise.empty else "(không có baseline chung)")

    return df_fold, df_meta, df_winrate


def parse_args():
    p = argparse.ArgumentParser(
        description="Kiểm định thống kê HQMoSS-Net vs baseline/ablation từ predictions đã lưu.")
    p.add_argument("--dataset", choices=["ph2", "glas", "both"], default="both")
    p.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "stat_tests"))
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    return p.parse_args()


def main():
    # Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = parse_args()
    datasets = ["ph2", "glas"] if args.dataset == "both" else [args.dataset]
    run(datasets, os.path.abspath(args.out_dir),
        n_boot=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
