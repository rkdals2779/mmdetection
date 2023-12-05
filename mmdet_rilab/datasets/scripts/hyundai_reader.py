import os.path as op
import numpy as np
from glob import glob
import cv2


class HyundaiReader:
    def __init__(self, data_path, split):
        self.frame_names = self.init_drive(data_path)
        self.split = split

    def init_drive(self, data_path):
        drive_paths = glob(op.join(data_path, self.split, 'image'), recursive=True)
        frame_names = []
        for drive_path in drive_paths:
            frame_names += glob(op.join(drive_path, "*.png"))
        frame_names.sort()
        print("[HyundaiReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def num_frames(self):
        return len(self.frame_names)

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index):
        """
        :return: bounding boxes in 'yxhw' format
        """
        image_file = self.frame_names[index]
        label_file = image_file.replace("image", "label").replace(".png", ".txt")
        bboxes = []
        categories = []
        with open(label_file, 'r') as f:
            bbox_lines = f.readlines()
            for line in bbox_lines:
                bbox, category = self.extract_box(line)
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)

        bboxes = np.array(bboxes)
        return bboxes, categories

    def extract_box(self, line):
        raw_label = line.strip("\n").split(",")
        category_name, y1, x1, h, w, height = raw_label
        y = int(float(h)) / 2 + int(float(y1))
        x = int(float(w)) / 2 + int(float(x1))
        h = int(float(h))
        w = int(float(w))
        bbox = np.array([y, x, h, w, height], dtype=np.float32)
        return bbox, category_name
