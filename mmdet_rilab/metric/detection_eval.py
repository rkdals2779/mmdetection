from mmdet.registry import METRICS
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Union
from collections import defaultdict
import numpy as np
import torch
from pycocotools import mask as maskUtils
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results

@METRICS.register_module()
class DetectionEval(BaseMetric):
    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False) -> None:
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.maxDets = [100, 300, 1000]
        self.areaRng = [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
        print("detection eval created:")
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # parse gt
            gt = dict()
            gt['img_id'] = data_sample['img_id']
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['anns'] = data_sample['gt_instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        for gt in gts:
            image_id = gt["img_id"]
            for gt_label, gt_bbox in zip(gt["anns"]["labels"], gt["anns"]["bboxes"]):
                self._gts[image_id, gt_label].append(
                    {"image_id": image_id,
                     "category_id": gt_label,
                     "bbox": gt_bbox,
                     "area": gt_bbox[2] * gt_bbox[3]}
                )

        for dt in preds:
            image_id = dt["img_id"]
            for dt_label, dt_bbox, dt_score in zip(dt["labels"], dt["bboxes"], dt["scores"]):
                self._dts[image_id, dt_label].append(
                    {"image_id": image_id,
                     "category_id": dt_label,
                     "bbox": dt_bbox,
                     "score": dt_score,
                     "area": dt_bbox[2] * dt_bbox[3]}
                )


        self.cat_ids = [0, 1, 2, 3]
        self.img_ids = [gt['img_id'] for gt in gts]

        ious = self.calculate_iou()


        results = {'coco/bbox_mAP': 0.0,
                   'coco/bbox_mAP_50': 0.0,
                   'coco/bbox_mAP_75': 0.0,
                   'coco/bbox_mAP_l': 0.0,
                   'coco/bbox_mAP_m': 0.0,
                   'coco/bbox_mAP_s': 0.0}
        return results



    def calculate_iou(self):
        catIds = self.cat_ids
        imgIds = self.img_ids

        self.ious = {(imgId, catId): self.computeIoU(imgId, catId) \
                     for imgId in imgIds
                     for catId in catIds}
        maxDet = self.maxDets[-1]
        self.evalImgs = [self.evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in self.areaRng
                         for imgId in imgIds
                         ]

        print("")


    def computeIoU(self, imgId, catId):
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > self.maxDets[-1]:
            dt = dt[0:self.maxDets[-1]]

        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]

        iscrowd = [0 for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [0 for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))

