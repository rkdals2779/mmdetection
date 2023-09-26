import numpy as np
import cv2

from mmdet.utils.rilab.dataloader.readers.mmdet_reader_base import DataFrameReaderBase


class MMDetCocoShapeReader(DataFrameReaderBase):
    def __init__(self, frame_data, classes):
        super().__init__(frame_data, classes)

    def get_image(self, index):
        image = self.frame_data['inputs'][index].cpu().numpy().transpose(1, 2, 0)
        # image = cv2.imread(self.frame_data['data_samples'][index].img_path)
        # print(self.frame_data['data_samples'][index].img_path)
        return image

    def get_box2d(self, index):
        bboxes = self.frame_data['data_samples'][index].gt_instances.bboxes.numpy()
        x1, y1, x2, y2 = np.split(bboxes, 4, axis=-1)
        bboxes = np.round(np.array([(y1 + y2) / 2, (x1 + x2) / 2, y2 - y1, x2 - x1], dtype=np.int32)).T
        return bboxes.reshape(-1, 4)

    def get_class(self, index):
        labels = self.frame_data['data_samples'][index].gt_instances.labels
        classes = []
        for label in labels:
            classes.append(self.classes[label])
        return classes

    def get_lane(self, index):
        lane = self.frame_data['data_samples'][index].lane
        return lane



# ====================================

