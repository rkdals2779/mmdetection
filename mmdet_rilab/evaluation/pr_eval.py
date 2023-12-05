import numpy as np
import cv2
from mmdet_rilab.visualization import Visualizer
from mmdet_rilab.logger import HistoryLogger


class PREvaluator:
    def __init__(self, iou_thresh, num_category):
        self.iou_thresh = iou_thresh
        self.num_category = num_category
        self.tpfpfn = {'tp': [0., 0., 0., 0.], 'fp': [0., 0., 0., 0.], 'fn': [0., 0., 0., 0.]}
        self.height_errs = []
        self.visualization = Visualizer()

    def count_tpfpfn(self, grtr, pred):
        if pred["bboxes"].size == 0:
            for label in grtr["labels"]:
                self.tpfpfn["fn"][label[0]] += 1
            return
        img_id_name = grtr["img_path"]
        grtr = {k: v for k, v in grtr.items() if k != "img_path"}
        splits = self.split_tpfpfn(grtr, pred)
        self.visualization(img_id_name, splits)
        self.accumulate_splits(splits)

    def get_metrics(self):
        print("[get_metrics] recall/precision per class\n")
        for metric, value in self.tpfpfn.items():
            print(metric, value)

        for key in self.tpfpfn:
            self.tpfpfn[key] = np.array(self.tpfpfn[key])
        total_recall = sum(self.tpfpfn["tp"])/(sum(self.tpfpfn["tp"]) + sum(self.tpfpfn["fn"]) + 1e-7)
        total_precision = sum(self.tpfpfn["tp"])/(sum(self.tpfpfn["tp"]) + sum(self.tpfpfn["fp"]) + 1e-7)
        recall = self.tpfpfn["tp"]/(self.tpfpfn["tp"] + self.tpfpfn["fn"] + 1e-7)
        precision = self.tpfpfn["tp"]/(self.tpfpfn["tp"] + self.tpfpfn["fp"] + 1e-7)
        if len(self.height_errs) != 0:
            height_err = np.mean(np.concatenate(self.height_errs))
        else:
            height_err = None
        self.tpfpfn = {'tp': [0., 0., 0., 0.], 'fp': [0., 0., 0., 0.], 'fn': [0., 0., 0., 0.]}
        return {"TOTAL_recall": total_recall, "TOTAL_precision": total_precision,
                "recall": recall, "precision": precision, "height_err": height_err}

    def split_tpfpfn(self, grtr, pred):
        """
        :param grtr: {'bbox': (N, 4), 'category': [N, 1] ...}
        :param pred: {'bbox': (M, 4), 'category': [M, 1] ...}
        :return:
        """
        pred = self.insert_batch(pred)
        grtr = self.insert_batch(grtr)
        batch, M, _ = pred["labels"].shape
        valid_mask = grtr["object"]
        ctgr_match = np.isclose(grtr["labels"], np.swapaxes(pred["labels"], 1, 2))  # (batch, N, M)
        ctgr_match = ctgr_match.astype(np.float32)
        iou = self.compute_iou_general(grtr["bboxes"], pred["bboxes"])  # (batch, N, M)
        iou *= ctgr_match
        best_iou = np.max(iou, axis=-1, keepdims=True)  # (batch, N, 1)
        best_idx = np.argmax(iou, axis=-1, keepdims=True)  # (batch, N, 1)
        grtr_tp_mask = best_iou > self.iou_thresh  # (batch, N, 1)
        grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
        grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
        grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
        grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
        grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]
        # last dimension rows where grtr_tp_mask == 0 are all-zero
        pred_tp_mask = self.indices_to_binary_mask(best_idx, grtr_tp_mask, M)
        pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
        pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
        pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}
        # (batch, M, D)[(batch, N)] = (batch, N, D)
        pred_aligned = {key: np.take(val, best_idx) for key, val in pred_tp.items()}
        return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn,
                "pred_tp_align": pred_aligned}

    def insert_batch(self, data):
        for key in data.keys():
            data[key] = np.expand_dims(data[key], axis=0)
        return data

    def compute_iou_general(self, grtr, pred):
        grtr = np.expand_dims(grtr, axis=-2)  # (batch, N1, 1, D1)
        pred = np.expand_dims(pred, axis=-3)  # (batch, 1, N2, D2)
        inter_tl = np.maximum(grtr[..., :2], pred[..., :2])  # (batch, N1, N2, 2)
        inter_br = np.minimum(grtr[..., 2:4], pred[..., 2:4])  # (batch, N1, N2, 2)
        inter_hw = inter_br - inter_tl  # (batch, N1, N2, 2)
        inter_hw = np.maximum(inter_hw, 0)
        inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (batch, N1, N2)

        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1]) # (batch, 1, N2)
        grtr_area = (grtr[..., 2] - grtr[..., 0]) * (grtr[..., 3] - grtr[..., 1])  # (batch, N1, 1)
        iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)  # (batch, N1, N2)
        return iou

    def create_categoriezed_mask(self, pred, grtr):
        max_length = max(len(pred), len(grtr))
        padded_pred = np.pad(pred, (0, max_length - len(pred)), mode="constant", constant_values=-1)
        padded_grtr = np.pad(grtr, (0, max_length - len(grtr)), mode="constant", constant_values=-1)
        categoriezed_mask = padded_pred == padded_grtr
        return categoriezed_mask

    def indices_to_binary_mask(self, best_idx, valid_mask, depth):
        best_idx_onehot = self.one_hot(best_idx[..., 0], depth) * valid_mask
        binary_mask = np.expand_dims(np.max(best_idx_onehot, axis=1), axis=-1) # (batch, M, 1)
        return binary_mask.astype(np.float32)

    def count_per_class(self, boxes):
        boxes_ctgr = boxes["labels"][..., 0].astype(np.int32)  # (batch, N)
        mask = boxes["scores"] if "scores" in boxes else boxes["object"]
        mask = mask > 0
        boxes_onehot = self.one_hot(boxes_ctgr, self.num_category) * mask  # (batch, N, K)
        boxes_count = np.sum(boxes_onehot, axis=(0, 1))     # (K)
        return boxes_count

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape)[grtr_category.astype(np.int32)]
        return one_hot_data

    def accumulate_splits(self, split):
        counts = {}
        for atr in split:
            if atr == "pred_tp_align":
                continue
            counts[atr.split('_')[1]] = self.count_per_class(split[atr])
        if len(self.tpfpfn) == 0:
            for key in counts:
                self.tpfpfn[key] = counts[key]
        else:
            for key in counts:
                self.tpfpfn[key] += counts[key]

        height_err = np.abs(split['pred_tp_align']['heights'] - split['grtr_tp']['heights']).reshape(-1)
        self.height_errs.append(height_err)
