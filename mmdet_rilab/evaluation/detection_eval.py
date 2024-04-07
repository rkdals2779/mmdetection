from mmdet.registry import METRICS
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
from mmengine.evaluator import BaseMetric
from .pr_eval import PREvaluator
from mmdet_rilab.logger import HistoryLogger


@METRICS.register_module()
class DetectionEval(BaseMetric):
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: str = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pr_eval = PREvaluator(iou_thresh=kwargs['iou_thresh'], num_category=kwargs['num_category'])

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()[:, np.newaxis]
            result['labels'] = pred['labels'].cpu().numpy()[:, np.newaxis]
            result['heights'] = pred['heights'].cpu().numpy()[:, np.newaxis]
            # parse gt
            gt = dict()
            gt['img_path'] = data_sample['img_path']
            gt['img_id'] = data_sample['img_id']
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['bboxes'] = data_sample['gt_instances']['bboxes'].cpu().numpy()
            gt['labels'] = data_sample['gt_instances']['labels'].cpu().numpy()[:, np.newaxis]
            gt['heights'] = data_sample['gt_instances']['heights'].cpu().numpy()[:, np.newaxis]
            gt['object'] = np.ones_like(gt['labels'])
            # add converted result to the results list
            self.results.append((gt, result))

    @HistoryLogger(count_epoch_and_save=True)
    def compute_metrics(self, results: list) -> Dict[str, float]:
        class_num = [0, 0, 0, 0]
        for grtr, pred in results:
            ##
            pred['labels'] = np.where(pred['labels']==4, 0, pred['labels'])
            ##
            for i in grtr["labels"]:
                class_num[i[0]] += 1
            self.pr_eval.count_tpfpfn(grtr, pred)
        result_metric = self.pr_eval.get_metrics()

        return result_metric


