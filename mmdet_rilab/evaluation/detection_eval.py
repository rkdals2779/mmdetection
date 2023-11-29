from mmdet.registry import METRICS
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
from mmengine.evaluator import BaseMetric
from .pr_eval import PREvaluator


@METRICS.register_module()
class DetectionEval(BaseMetric):
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: str = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pr_eval = PREvaluator()

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()[:, np.newaxis]
            result['labels'] = pred['labels'].cpu().numpy()[:, np.newaxis]
            # parse gt
            gt = dict()
            gt['img_id'] = data_sample['img_id']
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['bboxes'] = data_sample['gt_instances']['bboxes'].cpu().numpy()
            gt['labels'] = data_sample['gt_instances']['labels'].cpu().numpy()[:, np.newaxis]
            gt['object'] = np.ones_like(gt['labels'])
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        for grtr, pred in results:
            self.pr_eval.count_tpfpfn(grtr, pred)
        return self.pr_eval.get_recall_precision()

