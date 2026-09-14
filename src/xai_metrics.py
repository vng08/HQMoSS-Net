"""
Các độ đo *định lượng* cho chất lượng giải thích Seg-Grad-CAM (interpretability).

Ứng với section 10 trong các notebook XAI. Ba nhóm chỉ số:

  1. Localization  : IoU/Dice giữa CAM (đã nhị phân hoá) và ground-truth
       - cam_gt_overlap_otsu    : ngưỡng Otsu (đối chiếu độ tin cậy ngưỡng).
       - cam_gt_overlap_matched : ngưỡng theo đúng tỉ lệ diện tích GT của từng ảnh
                                  (chỉ số chính) + lọc connected-component.
       - energy_pointing_game   : % năng lượng CAM rơi vào vùng GT (không cần ngưỡng).
  2. Boundary      : boundary_iou, hausdorff_distance_95 (độ khớp đường biên).
  3. Faithfulness  : deletion_insertion_auc (RISE, Petsiuk et al. 2018) — xoá/thêm
                     dần pixel theo thứ tự CAM giảm dần rồi đo IoU dự đoán.

Hàm lọc nhiễu connected-component (dùng làm postprocess cho CAM đã nhị phân):
  - keep_largest_component  : giữ 1 vùng lớn nhất — dataset single-instance (PH2:
                              100% ảnh chỉ có 1 vùng tổn thương liên thông).
  - remove_small_components : giữ mọi vùng đủ lớn, chỉ loại nhiễu nhỏ — dataset
                              multi-instance (GlaS: ~8.6 tuyến/ảnh).
"""

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. Lọc nhiễu connected-component sau khi nhị phân hoá CAM
# ---------------------------------------------------------------------------
def keep_largest_component(cam_bin):
    """Chỉ giữ connected component lớn nhất (dataset single-instance như PH2)."""
    labeled, num_features = ndimage.label(cam_bin)
    if num_features <= 1:
        return cam_bin
    sizes = ndimage.sum(cam_bin, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def remove_small_components(cam_bin, min_area_ratio=0.001):
    """Giữ mọi component đủ lớn, chỉ loại nhiễu nhỏ (dataset multi-instance như GlaS)."""
    labeled, num_features = ndimage.label(cam_bin)
    if num_features <= 1:
        return cam_bin
    total_pixels = cam_bin.size
    min_area = max(1, int(min_area_ratio * total_pixels))
    sizes = ndimage.sum(cam_bin, labeled, range(1, num_features + 1))
    valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if not valid_labels:
        largest_label = np.argmax(sizes) + 1
        return labeled == largest_label
    return np.isin(labeled, valid_labels)


def count_gt_components(y_val):
    """Đếm số connected component trong mỗi mask GT — dùng để xác nhận đặc tính
    single/multi-instance của dataset trước khi chọn phương pháp hậu xử lý CAM."""
    comps = []
    for i in range(len(y_val)):
        gt = np.squeeze(y_val[i]) > 0.5
        _, num = ndimage.label(gt)
        comps.append(num)
    return np.array(comps)


# ---------------------------------------------------------------------------
# 2. Localization: IoU/Dice giữa CAM và GT
# ---------------------------------------------------------------------------
def cam_gt_overlap_otsu(cam, gt_mask):
    """IoU/Dice với ngưỡng Otsu — dùng để đối chiếu độ tin cậy của ngưỡng
    (Otsu có thể lệch khi CAM phân phối lệch mạnh)."""
    gt = np.squeeze(gt_mask) > 0.5
    hist, bin_edges = np.histogram(cam, bins=256, range=(0, 1))
    hist = hist.astype(float)
    total = hist.sum()
    sum_all = np.sum(hist * bin_edges[:-1])
    sum_bg, w_bg, max_var, best_t = 0.0, 0.0, 0.0, 0.5
    for i in range(256):
        w_bg += hist[i]
        if w_bg == 0 or w_bg == total:
            continue
        w_fg = total - w_bg
        sum_bg += hist[i] * bin_edges[i]
        m_bg = sum_bg / w_bg
        m_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var > max_var:
            max_var, best_t = var, bin_edges[i]
    cam_bin = cam > best_t
    inter = np.logical_and(cam_bin, gt).sum()
    union = np.logical_or(cam_bin, gt).sum()
    iou = inter / (union + 1e-8)
    dice = 2 * inter / (cam_bin.sum() + gt.sum() + 1e-8)
    return iou, dice


def cam_gt_overlap_matched(cam, gt_mask, postprocess_fn=keep_largest_component):
    """
    IoU/Dice với ngưỡng area-matched — mỗi ảnh tự ngưỡng theo đúng tỉ lệ diện
    tích GT của chính nó (chỉ số localization chính).

    postprocess_fn: hàm lọc connected-component áp lên CAM đã nhị phân
        (keep_largest_component cho PH2, remove_small_components cho GlaS).
        Truyền None để bỏ lọc.
    Trả về (iou, dice, cam_bin).
    """
    gt = np.squeeze(gt_mask) > 0.5
    gt_ratio = gt.mean()
    percentile = 100 * (1 - gt_ratio)
    threshold = np.percentile(cam, percentile)

    cam_bin = cam > threshold
    if postprocess_fn is not None:
        cam_bin = postprocess_fn(cam_bin)

    inter = np.logical_and(cam_bin, gt).sum()
    union = np.logical_or(cam_bin, gt).sum()
    iou = inter / (union + 1e-8)
    dice = 2 * inter / (cam_bin.sum() + gt.sum() + 1e-8)
    return iou, dice, cam_bin


def energy_pointing_game(cam, gt_mask):
    """% năng lượng CAM rơi vào vùng GT — không phụ thuộc ngưỡng."""
    gt = np.squeeze(gt_mask) > 0.5
    total_energy = cam.sum() + 1e-8
    energy_in_gt = cam[gt].sum()
    return energy_in_gt / total_energy


# ---------------------------------------------------------------------------
# 3. Boundary: Boundary IoU + Hausdorff Distance (95%)
# ---------------------------------------------------------------------------
def _mask_boundary(mask, dilation_ratio=0.02):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag))) if dilation_ratio > 0 else 1
    eroded = ndimage.binary_erosion(mask, iterations=dilation, border_value=0)
    return mask.astype(bool) & (~eroded)


def boundary_iou(cam_bin, gt_mask, dilation_ratio=0.02):
    gt = np.squeeze(gt_mask) > 0.5
    cam_bin = np.squeeze(cam_bin).astype(bool)
    gt_boundary = _mask_boundary(gt, dilation_ratio)
    cam_boundary = _mask_boundary(cam_bin, dilation_ratio)
    inter = np.logical_and(gt_boundary, cam_boundary).sum()
    union = np.logical_or(gt_boundary, cam_boundary).sum()
    return inter / (union + 1e-8)


def hausdorff_distance_95(cam_bin, gt_mask):
    gt = np.squeeze(gt_mask) > 0.5
    cam_bin = np.squeeze(cam_bin).astype(bool)
    gt_boundary = _mask_boundary(gt, dilation_ratio=0.0)
    cam_boundary = _mask_boundary(cam_bin, dilation_ratio=0.0)
    if gt_boundary.sum() == 0 or cam_boundary.sum() == 0:
        return np.nan
    dt_gt = ndimage.distance_transform_edt(~gt_boundary)
    dt_cam = ndimage.distance_transform_edt(~cam_boundary)
    d_cam_to_gt = dt_gt[cam_boundary]
    d_gt_to_cam = dt_cam[gt_boundary]
    return max(np.percentile(d_cam_to_gt, 95), np.percentile(d_gt_to_cam, 95))


# ---------------------------------------------------------------------------
# 4. Faithfulness: Deletion / Insertion AUC (RISE, Petsiuk et al. 2018)
# ---------------------------------------------------------------------------
def _perturb_and_score(model, image, order, roi_ref, n_steps=15, mode="delete"):
    h, w, c = image.shape
    n_pixels = h * w
    step_size = max(n_pixels // n_steps, 1)
    baseline = np.full_like(image, image.mean())
    scores = []
    for step in range(n_steps + 1):
        k = min(step * step_size, n_pixels)
        idx = order[:k]
        rr, cc = np.unravel_index(idx, (h, w))
        if mode == "delete":
            canvas_step = image.copy()
            canvas_step[rr, cc, :] = baseline[rr, cc, :]
        else:
            canvas_step = baseline.copy()
            canvas_step[rr, cc, :] = image[rr, cc, :]
        pred = model.predict(canvas_step[None, ...], verbose=0)[0, ..., 0]
        inter = np.logical_and(pred > 0.5, roi_ref > 0.5).sum()
        union = np.logical_or(pred > 0.5, roi_ref > 0.5).sum()
        scores.append(inter / (union + 1e-8))
    return np.array(scores)


def deletion_insertion_auc(model, image, cam, gt_mask=None, n_steps=15):
    """
    Deletion AUC (thấp là tốt: xoá pixel quan trọng làm dự đoán sụp nhanh) và
    Insertion AUC (cao là tốt: thêm pixel quan trọng phục hồi dự đoán nhanh),
    xếp hạng pixel theo giá trị CAM giảm dần.
    """
    order = np.argsort(-cam.flatten())
    roi_ref = gt_mask if gt_mask is not None else (
        model.predict(image[None, ...], verbose=0)[0, ..., 0] > 0.5
    )
    roi_ref = np.squeeze(roi_ref)
    del_scores = _perturb_and_score(model, image, order, roi_ref, n_steps, mode="delete")
    ins_scores = _perturb_and_score(model, image, order, roi_ref, n_steps, mode="insert")
    del_auc = np.trapz(del_scores, dx=1.0 / n_steps)
    ins_auc = np.trapz(ins_scores, dx=1.0 / n_steps)
    return del_auc, ins_auc
