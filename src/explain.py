"""
Giải thích mô hình phân vùng bằng Seg-Grad-CAM + các độ đo interpretability.

Module hỗ trợ 4 chế độ (``--mode``):

  cam      : vẽ Image / Ground Truth / Prediction / Seg-Grad-CAM cho vài ảnh
             (mặc định) và lưu cams.npy.
  metrics  : tính đầy đủ độ đo định lượng chất lượng CAM trên toàn bộ val set
             (localization, boundary, faithfulness — xem xai_metrics.py).
  compare  : so sánh Seg-Grad-CAM giữa HQMoSS-Net và UNet, chọn các ảnh mà
             HQMoSS định vị tốt hơn UNet nhiều nhất để minh hoạ.
  router   : phân tích cân bằng định tuyến (routing load balance) của khối
             QuantumMoE trên một checkpoint — xem router_analysis.py.

Seg-Grad-CAM tính gradient của tổng dự đoán trong vùng quan tâm đối với một
feature map trung gian rồi lấy trung bình có trọng số các kênh -> heatmap cho
biết mô hình "nhìn" vào đâu. Lớp lượng tử (qml.qnn.KerasLayer) khả vi nên
gradient vẫn chảy qua bình thường.

Cách dùng (chạy trong thư mục src/, cần train trước với --save-weights):
    python explain.py --dataset ph2 --fold 1 --num-samples 5
    python explain.py --dataset ph2 --mode metrics --weights <hq_weights>
    python explain.py --dataset glas --mode compare \
        --weights <hq_weights> --unet-weights <unet_weights>

Layer đích mặc định được chọn theo *vị trí cấu trúc* (output của decoder stage
áp chót, cùng resolution 96x128 giữa mọi kiến trúc) nên tiêu chí so sánh XAI nhất
quán giữa HQMoSS-Net và UNet.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import tensorflow.keras.layers as layers

import pandas as pd

import config
from data import load_fold_data
from metrics import compute_iou
from models.segmentation import build_variant
import xai_metrics as xm
import router_analysis as ra


# ---------------------------------------------------------------------------
# 1. Chọn layer đích cho Grad-CAM
# ---------------------------------------------------------------------------
def list_conv_layers(model):
    """In danh sách layer conv/deconv (index, tên, output shape) để chọn layer đích."""
    candidates = [
        (i, l.name, l.output_shape)
        for i, l in enumerate(model.layers)
        if isinstance(l, (layers.Conv2D, layers.Conv2DTranspose, layers.DepthwiseConv2D))
    ]
    for i, name, shape in candidates:
        print(f"[{i:3d}] {name:35s} -> {shape}")
    return candidates


def find_layers_by_name(model, keyword):
    """Liệt kê layer có từ khoá trong tên (vd 'quantum_moe', 'mamba_bottleneck')."""
    matches = [l.name for l in model.layers if keyword.lower() in l.name.lower()]
    for name in matches:
        print(name)
    return matches


def get_default_target_layer(model):
    """Chọn layer conv áp chót (ngay trước Conv2D(1,1) sigmoid) làm layer đích."""
    conv_layers = [
        l.name for l in model.layers
        if isinstance(l, (layers.Conv2D, layers.Conv2DTranspose, layers.DepthwiseConv2D))
    ]
    return conv_layers[-2] if len(conv_layers) >= 2 else conv_layers[-1]


def get_true_stage_output_layer(model, stages_from_end=2, verbose=True):
    """
    Lấy layer là OUTPUT THẬT SỰ của một decoder stage — bất kể kiểu layer
    (Conv2D, Add, Activation...) — bằng cách lấy layer NGAY TRƯỚC Conv2DTranspose
    kế tiếp trong thứ tự topo của model.layers.

    Không hardcode tên/index tuyệt đối -> ổn định qua nhiều lần chạy và DÙNG CHUNG
    cho cả UNet lẫn HQMoSS-Net, đảm bảo tiêu chí chọn layer nhất quán khi so sánh.
    Với stages_from_end=2 -> output decoder stage áp chót (resolution 96x128).
    """
    transpose_idx = [i for i, l in enumerate(model.layers)
                     if isinstance(l, layers.Conv2DTranspose)]
    if len(transpose_idx) < stages_from_end:
        raise ValueError("Không đủ Conv2DTranspose để xác định stage yêu cầu.")

    end_idx = transpose_idx[-stages_from_end + 1] if stages_from_end > 1 else len(model.layers)
    true_output_idx = end_idx - 1
    layer = model.layers[true_output_idx]
    if verbose:
        print(f"Layer stage output: [{true_output_idx}] {layer.name} "
              f"({layer.__class__.__name__}) -> {layer.output_shape}")
    return layer.name


# ---------------------------------------------------------------------------
# 2. Lõi Seg-Grad-CAM
# ---------------------------------------------------------------------------
def seg_grad_cam(model, image, layer_name, roi_mask=None):
    """
    Tính Seg-Grad-CAM cho 1 ảnh.
    roi_mask=None -> mặc định dùng vùng foreground mô hình dự đoán (pred > 0.5).
    Trả về (cam [H,W] chuẩn hoá [0,1], pred_mask [H,W]).
    """
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    img_batch = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_batch)
        preds = preds[0, ..., 0]
        roi = (tf.cast(preds > 0.5, tf.float32) if roi_mask is None
               else tf.convert_to_tensor(np.squeeze(roi_mask), dtype=tf.float32))
        loss = tf.reduce_sum(preds * roi)

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        raise ValueError(f"Không tính được gradient tại layer '{layer_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_output[0] * pooled_grads, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2], method="bilinear")[..., 0]
    return cam.numpy(), preds.numpy()


def overlay_cam(image, cam, alpha=0.45, cmap="jet"):
    """Trộn ảnh gốc và heatmap thành 1 ảnh RGB (alpha compositing) để tránh nhiễu moiré."""
    colormap = cm.get_cmap(cmap)
    heatmap_rgb = colormap(cam)[..., :3]
    base = image if image.max() <= 1.0 else image / 255.0
    return np.clip((1 - alpha) * base + alpha * heatmap_rgb, 0, 1)


# ---------------------------------------------------------------------------
# 3. Tiện ích: dựng mô hình + chọn postprocess theo dataset
# ---------------------------------------------------------------------------
def build_xai_model(model_name, input_shape):
    """Dựng mô hình (HQMoSS_Net / UNet / biến thể khác) để nạp weights và giải thích."""
    return build_variant(model_name, input_shape, config.NUM_FILTERS, iou_fn=compute_iou)


def dataset_postprocess(dataset):
    """
    Chọn hàm lọc connected-component cho CAM theo đặc tính dataset:
      ph2  -> keep_largest_component  (single-instance: 1 tổn thương/ảnh),
      glas -> remove_small_components (multi-instance: nhiều tuyến/ảnh).
    """
    if dataset.lower() == "glas":
        return lambda cam_bin: xm.remove_small_components(cam_bin, min_area_ratio=0.001)
    return xm.keep_largest_component


# ---------------------------------------------------------------------------
# 4. Chế độ CAM: vẽ + lưu ảnh cho 1 fold
# ---------------------------------------------------------------------------
def run_for_fold(model, dataset, fold, layer_name, num_samples, out_dir):
    image_dir, mask_dir, fold_dir, kind = config.dataset_paths(dataset)
    _, X_val, _, y_val = load_fold_data(fold, image_dir, mask_dir, fold_dir, config.IMAGE_SIZE, kind=kind)

    os.makedirs(out_dir, exist_ok=True)
    n = min(num_samples, len(X_val))
    cams = []

    for i in range(n):
        cam, pred_mask = seg_grad_cam(model, X_val[i], layer_name)
        cams.append(cam)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(X_val[i]); axes[0].set_title("Image"); axes[0].axis("off")
        axes[1].imshow(np.squeeze(y_val[i]), cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Prediction"); axes[2].axis("off")
        axes[3].imshow(overlay_cam(X_val[i], cam), interpolation="bilinear")
        axes[3].set_title(f"Seg-Grad-CAM\n@ {layer_name}"); axes[3].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{i}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    np.save(os.path.join(out_dir, "cams.npy"), np.array(cams))
    print(f"Đã lưu {n} ảnh Seg-Grad-CAM + cams.npy vào {out_dir}")
    return np.array(cams)


# ---------------------------------------------------------------------------
# 5. Chế độ metrics: tính đầy đủ độ đo interpretability trên 1 fold
# ---------------------------------------------------------------------------
def evaluate_interpretability(model, dataset, fold, layer_name, postprocess_fn,
                              num_samples=None, n_steps=15, out_dir=None, tag="model"):
    """Tính bộ độ đo định lượng chất lượng CAM trên (một phần) val set của 1 fold."""
    image_dir, mask_dir, fold_dir, kind = config.dataset_paths(dataset)
    _, X_val, _, y_val = load_fold_data(fold, image_dir, mask_dir, fold_dir, config.IMAGE_SIZE, kind=kind)
    n = len(X_val) if num_samples is None else min(num_samples, len(X_val))

    results = {
        "iou_otsu": [], "dice_otsu": [],
        "iou_matched": [], "dice_matched": [],
        "boundary_iou": [], "hausdorff95": [],
        "energy_pg": [], "deletion_auc": [], "insertion_auc": [],
    }

    for i in range(n):
        cam, _ = seg_grad_cam(model, X_val[i], layer_name)
        gt = np.squeeze(y_val[i])

        iou_o, dice_o = xm.cam_gt_overlap_otsu(cam, gt)
        iou_m, dice_m, cam_bin_m = xm.cam_gt_overlap_matched(cam, gt, postprocess_fn=postprocess_fn)
        b_iou = xm.boundary_iou(cam_bin_m, gt)
        hd95 = xm.hausdorff_distance_95(cam_bin_m, gt)
        epg = xm.energy_pointing_game(cam, gt)
        del_auc, ins_auc = xm.deletion_insertion_auc(model, X_val[i], cam, gt_mask=gt, n_steps=n_steps)

        results["iou_otsu"].append(iou_o)
        results["dice_otsu"].append(dice_o)
        results["iou_matched"].append(iou_m)
        results["dice_matched"].append(dice_m)
        results["boundary_iou"].append(b_iou)
        results["hausdorff95"].append(hd95)
        results["energy_pg"].append(epg)
        results["deletion_auc"].append(del_auc)
        results["insertion_auc"].append(ins_auc)

    summary = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in results.items()}

    print(f"\n=== {tag} — Interpretability metrics @ {layer_name} ({dataset.upper()}, fold {fold}) ===")
    for k, (mean, std) in summary.items():
        print(f"{k:15s}: {mean:.4f} ± {std:.4f}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"interp_{tag}_summary.npy"), summary)
        np.save(os.path.join(out_dir, f"interp_{tag}_raw.npy"), results)
        print(f"Đã lưu độ đo interpretability vào {out_dir}")
    return summary, results


# ---------------------------------------------------------------------------
# 6. Chế độ compare: so sánh Seg-Grad-CAM HQMoSS-Net vs UNet
# ---------------------------------------------------------------------------
def compare_models(dataset, fold, hq_weights, unet_weights, out_dir,
                   top_k=5, quality_percentile=50, stages_from_end=2):
    """
    Dựng HQMoSS-Net và UNet (kiến trúc dùng chung IMAGE_SIZE/NUM_FILTERS), nạp
    weights tương ứng, tính IoU (area-matched) của CAM với GT cho từng ảnh val,
    rồi chọn top-k ảnh mà HQMoSS vượt UNet nhiều nhất (trong nhóm HQMoSS đạt sàn
    chất lượng theo chính phân phối của dataset) để minh hoạ.
    """
    input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
    hq_model = build_xai_model("HQMoSS_Net", input_shape)
    unet_model = build_xai_model("UNet", input_shape)
    hq_model.load_weights(hq_weights)
    unet_model.load_weights(unet_weights)

    hq_layer = get_true_stage_output_layer(hq_model, stages_from_end)
    unet_layer = get_true_stage_output_layer(unet_model, stages_from_end)
    postprocess_fn = dataset_postprocess(dataset)

    image_dir, mask_dir, fold_dir, kind = config.dataset_paths(dataset)
    _, X_val, _, y_val = load_fold_data(fold, image_dir, mask_dir, fold_dir, config.IMAGE_SIZE, kind=kind)
    n_samples = len(X_val)
    print(f"Số ảnh val ({dataset.upper()}, fold {fold}): {n_samples}")

    hq_ious, unet_ious, hq_cams, unet_cams = [], [], [], []
    for i in range(n_samples):
        gt = np.squeeze(y_val[i])
        cam_hq, _ = seg_grad_cam(hq_model, X_val[i], hq_layer)
        iou_hq, _, _ = xm.cam_gt_overlap_matched(cam_hq, gt, postprocess_fn=postprocess_fn)
        cam_unet, _ = seg_grad_cam(unet_model, X_val[i], unet_layer)
        iou_unet, _, _ = xm.cam_gt_overlap_matched(cam_unet, gt, postprocess_fn=postprocess_fn)
        hq_ious.append(iou_hq); unet_ious.append(iou_unet)
        hq_cams.append(cam_hq); unet_cams.append(cam_unet)

    hq_ious = np.array(hq_ious)
    unet_ious = np.array(unet_ious)
    diff = hq_ious - unet_ious
    print(f"HQMoSS mean iou_matched: {hq_ious.mean():.4f}  (min={hq_ious.min():.3f}, max={hq_ious.max():.3f})")
    print(f"UNet   mean iou_matched: {unet_ious.mean():.4f}  (min={unet_ious.min():.3f}, max={unet_ious.max():.3f})")

    quality_floor = np.percentile(hq_ious, quality_percentile)
    candidate_idx = [i for i in range(n_samples) if hq_ious[i] >= quality_floor]
    candidate_idx.sort(key=lambda i: diff[i], reverse=True)
    print(f"Sàn chất lượng HQMoSS (percentile {quality_percentile}): {quality_floor:.3f} "
          f"-> {len(candidate_idx)} ảnh thoả")
    if not candidate_idx:
        print("Không có ảnh nào thoả sàn — giảm --quality-percentile rồi chạy lại.")
        return

    os.makedirs(out_dir, exist_ok=True)
    for rank, i in enumerate(candidate_idx[:top_k]):
        gt = np.squeeze(y_val[i])
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(X_val[i]); axes[0].set_title("Image"); axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(overlay_cam(X_val[i], unet_cams[i]), interpolation="bilinear")
        axes[2].set_title(f"UNet\nSeg-Grad-CAM (IoU={unet_ious[i]:.2f})"); axes[2].axis("off")
        axes[3].imshow(overlay_cam(X_val[i], hq_cams[i]), interpolation="bilinear")
        axes[3].set_title(f"HQMoSS-Net\nSeg-Grad-CAM (IoU={hq_ious[i]:.2f})"); axes[3].axis("off")
        fig.suptitle(f"{dataset.upper()} — rank {rank + 1}", y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rank{rank + 1}_idx{i}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Đã lưu {min(top_k, len(candidate_idx))} ảnh so sánh vào {out_dir}")


# ---------------------------------------------------------------------------
# 7. Chế độ router: phân tích cân bằng định tuyến của QuantumMoE
# ---------------------------------------------------------------------------
def run_router_analysis(model_name, dataset, fold, out_dir, weights=None, compare_csv=None, compare_label=None):
    """
    Forward-pass val set qua router QuantumMoE, in + lưu thống kê cân bằng
    định tuyến (tỷ trọng mỗi nhánh, entropy, imbalance range, CV). Nếu
    ``compare_csv`` được cung cấp (per_image CSV từ một lần chạy khác, ví dụ
    dataset còn lại), so sánh bằng kiểm định Mann-Whitney U + rank-biserial r.

    LƯU Ý: đây là phân tích trên một checkpoint đại diện (một fold cụ thể),
    không phải trung bình nhiều fold.
    """
    input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
    model = build_xai_model(model_name, input_shape)
    model.load_weights(_resolve_weights(weights))

    router_probe = ra.get_router_probe(model)
    print(f"Router probe output shape: {router_probe.output_shape} (batch, H, W, n_experts)")

    image_dir, mask_dir, fold_dir, kind = config.dataset_paths(dataset)
    _, X_val, _, _ = load_fold_data(fold, image_dir, mask_dir, fold_dir, config.IMAGE_SIZE, kind=kind)
    print(f"Số ảnh val ({dataset.upper()}, fold {fold}): {len(X_val)}")

    summary, per_image_df = ra.compute_router_balance(router_probe, X_val)
    tag = f"{dataset}_fold{fold}"
    ra.print_router_summary(summary, per_image_df, tag=tag)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"router_balance_{tag}_per_image.csv")
    per_image_df.to_csv(csv_path, index=False)
    print(f"Đã lưu bảng theo ảnh: {csv_path}")

    if compare_csv is not None:
        other_df = pd.read_csv(compare_csv)
        label_a = dataset.upper()
        label_b = compare_label or "khác"
        cmp_df = ra.compare_router_balance(per_image_df, other_df, label_a=label_a, label_b=label_b)
        print(f"\n=== So sánh cân bằng định tuyến: {label_a} vs {label_b} (Mann-Whitney U) ===")
        print(cmp_df.to_string(index=False))
        cmp_path = os.path.join(out_dir, f"router_compare_{label_a}_vs_{label_b}.csv")
        cmp_df.to_csv(cmp_path, index=False)
        print(f"Đã lưu bảng so sánh: {cmp_path}")

    return summary, per_image_df


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Giải thích mô hình bằng Seg-Grad-CAM + độ đo interpretability.")
    p.add_argument("--mode", choices=["cam", "metrics", "compare", "router"], default="cam",
                   help="cam: vẽ heatmap; metrics: độ đo định lượng; compare: HQMoSS vs UNet; "
                        "router: cân bằng định tuyến QuantumMoE.")
    p.add_argument("--dataset", choices=["glas", "ph2"], default="ph2")
    p.add_argument("--model", default="HQMoSS_Net",
                   help="Mô hình để giải thích ở chế độ cam/metrics (vd HQMoSS_Net, UNet).")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=None,
                   help="Số ảnh xử lý (mặc định: cam=5, metrics=toàn bộ val set).")
    p.add_argument("--layer", default=None,
                   help="Tên layer đích (mặc định: output decoder stage áp chót theo cấu trúc).")
    p.add_argument("--stages-from-end", type=int, default=2,
                   help="Chọn layer đích = output decoder stage thứ N tính từ cuối (mặc định 2).")
    p.add_argument("--weights", default=None,
                   help="Weights của mô hình (mặc định: XAI-result/XAI_Ph2_HQMoSS/HPweights_fold1.weights.h5).")
    p.add_argument("--unet-weights", default=None, help="Weights UNet (chỉ dùng cho --mode compare).")
    p.add_argument("--n-steps", type=int, default=15, help="Số bước deletion/insertion (metrics).")
    p.add_argument("--top-k", type=int, default=5, help="Số ảnh minh hoạ (compare).")
    p.add_argument("--quality-percentile", type=float, default=50,
                   help="Sàn chất lượng HQMoSS theo percentile của chính dataset (compare).")
    p.add_argument("--out-dir", default=None, help="Thư mục lưu kết quả (mặc định trong XAI-result).")
    p.add_argument("--compare-csv", default=None,
                   help="(chỉ --mode router) CSV per_image từ một lần chạy khác để so sánh Mann-Whitney U.")
    p.add_argument("--compare-label", default=None,
                   help="(chỉ --mode router) Nhãn của tập so sánh trong --compare-csv (vd GlaS).")
    return p.parse_args()


def _resolve_weights(path):
    weights = path or os.path.join(config.XAI_RESULT_DIR, "HPweights_fold1.weights.h5")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Không tìm thấy weights: {weights}\n"
            f"Hãy train trước với: python train.py --model <MODEL> --dataset <ds> --save-weights\n"
            f"rồi trỏ --weights tới file weights tương ứng."
        )
    return weights


def main():
    args = parse_args()
    input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)

    if args.mode == "compare":
        hq_weights = _resolve_weights(args.weights)
        if not args.unet_weights:
            raise ValueError("--mode compare cần cả --weights (HQMoSS) và --unet-weights (UNet).")
        unet_weights = _resolve_weights(args.unet_weights)
        out_dir = args.out_dir or os.path.join(config.XAI_ROOT, f"comparison_{args.dataset}")
        compare_models(
            args.dataset, args.fold, hq_weights, unet_weights, out_dir,
            top_k=args.top_k, quality_percentile=args.quality_percentile,
            stages_from_end=args.stages_from_end,
        )
        return

    if args.mode == "router":
        out_dir = args.out_dir or os.path.join(config.XAI_RESULT_DIR, "router")
        run_router_analysis(
            args.model, args.dataset, args.fold, out_dir, weights=args.weights,
            compare_csv=args.compare_csv, compare_label=args.compare_label,
        )
        return

    # cam / metrics: một mô hình đơn
    model = build_xai_model(args.model, input_shape)
    weights = _resolve_weights(args.weights)
    model.load_weights(weights)

    layer_name = args.layer or get_true_stage_output_layer(model, args.stages_from_end)
    print(f"Layer đích: {layer_name}")

    if args.mode == "metrics":
        out_dir = args.out_dir or os.path.join(config.XAI_RESULT_DIR, "metrics")
        evaluate_interpretability(
            model, args.dataset, args.fold, layer_name, dataset_postprocess(args.dataset),
            num_samples=args.num_samples, n_steps=args.n_steps, out_dir=out_dir, tag=args.model,
        )
    else:  # cam
        out_dir = args.out_dir or os.path.join(config.XAI_RESULT_DIR, "gradcam")
        run_for_fold(model, args.dataset, args.fold, layer_name,
                     args.num_samples or 5, out_dir)


if __name__ == "__main__":
    main()
