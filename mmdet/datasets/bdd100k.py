import os.path as osp
import json
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

from mmdet.utils.rilab.io_logger import IOLogger

@DATASETS.register_module()
class BDD100KDataset(BaseDetDataset):
    METAINFO = {
        'classes':
            ('person', 'car', 'rider', 'bus', 'truck', 'bike', 'motor',
             'traffic light', 'traffic sign', 'train', ),  # Define your classes
        'palette': [(220, 20, 60), (0, 0, 142), (197, 226, 255), (0, 60, 100),
                    (0, 0, 70), (119, 11, 32), (0, 0, 230),
                    (255, 179, 240), (0, 125, 92), (0, 80, 100), ]  # Define your palette
    }

    def load_data_list(self) -> List[dict]:

        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.bdd100k = self.load_bdd100k_raw_data(local_path)

        data_list = []
        asd = True
        for img_id, raw_ann_info in enumerate(self.bdd100k, start=1):

            parsed_data_info = self.parse_data_info({
                "img_id":
                img_id,
                "raw_ann_info":
                raw_ann_info
            })
            if asd:
                IOLogger("BDD100KDataset.load_data_list").log_var("raw_ann_info", raw_ann_info, )
                IOLogger("BDD100KDataset.load_data_list").log_var("parsed_data_info", parsed_data_info)
                asd = False
            data_list.append(parsed_data_info)

        del self.bdd100k
        IOLogger("BDD100KDataset.load_data_list").log_var("len(data_list)", len(data_list))
        IOLogger("BDD100KDataset.load_data_list").log_var("data_list[0]", data_list)
        with open(f'/home/falcon/shin_workspace/Datacleaning/log/bdd100k{len(data_list)}.json', 'w') as file:
            json.dump(data_list, file)
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_id = raw_data_info['img_id']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        img_path = osp.join(self.data_prefix['img'], ann_info['name'])

        data_info['img_path'] = img_path
        data_info['img_id'] = img_id
        data_info['seg_map_path'] = None
        data_info['height'] = 720
        data_info['width'] = 1280

        instances = []
        data_info['lane'] = []
        data_info['drivable area'] = []
        for i, ann in enumerate(ann_info['labels']):
            instance = {}
            if ann['category'] in ['lane', 'drivable area']:
                data_info[ann['category']].append(ann['poly2d'][0]['vertices'])
            else:
                x1, y1, w, h = self.xxyy2xywh(ann['box2d'])
                inter_w = max(0, min(x1 + w, 1280) - max(x1, 0))
                inter_h = max(0, min(y1 + h, 720) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if w * h <= 0 or w < 1 or h < 1:
                    continue
                bbox = [x1, y1, x1 + w, y1 + h]

                instance['ignore_flag'] = 0
                instance['bbox'] = bbox
                instance['bbox_label'] = self.cat2label(ann['category'])
                instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def load_bdd100k_raw_data(self, local_path):
        with open(local_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def xxyy2xywh(self, xxyy: dict):
        x1, y1, x2, y2 = xxyy['x1'], xxyy['y1'], xxyy['x2'], xxyy['y2']
        return x1, y1, x2 - x1, y2 - y1

    def cat2label(self, category):
        classes = ('person', 'car', 'rider', 'bus', 'truck', 'bike', 'motor',
             'traffic light', 'traffic sign', 'train', )
        return classes.index(category)
