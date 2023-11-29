import os.path as op
import os
import numpy as np
from glob import glob
import cv2


class HyundaiDriveManager():
    def __init__(self, datapath, split):
        self.datapath = datapath
        self.split = split
        self.drive_paths = self.list_drive_paths()
        self.split = "val" if self.split == "test" else self.split

    def list_drive_paths(self):
        dirlist = glob(op.join(self.datapath, '*'))
        dirlist = [dir_ for dirs in dirlist for dir_ in glob(op.join(dirs, '*'))]
        dirlist = [directory for directory in dirlist if op.isdir(op.join(directory, "image"))]
        testset_file = op.join(self.datapath, 'test_set.txt')

        if self.split == "train":
            dirlist = self.pop_list(testset_file, dirlist)
        else:
            dirlist = self.push_list(testset_file)

        return dirlist

    def get_drive_paths(self):
        return self.drive_paths

    def get_drive_name(self, drive_index):
        drive_path = self.drive_paths[drive_index]
        print("drive_path", drive_path)
        drive_name = op.basename(drive_path)
        return drive_name

    def pop_list(self, testset_file, dirlist):
        with open(testset_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                target_file = op.join(self.datapath, line).strip('\n')
                if target_file in dirlist:
                    index = dirlist.index(target_file)
                    dirlist.pop(index)
        return dirlist

    def push_list(self, testset_file):
        test_list = []
        with open(testset_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                target_file = op.join(self.datapath, line).strip('\n')
                test_list.append(target_file)
        return test_list


class HyundaiReader():
    def __init__(self, drive_path, split, dataset_cfg):
        self.dataset_cfg = dataset_cfg
        self.frame_names = self.init_drive(drive_path)
        self.split = split


    """
    Public methods used outside this class
    """

    def init_drive(self, drive_path):
        frame_names = glob(op.join(drive_path, "image", "*.png"))
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

        # if not bboxes:
        #     raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        bboxes = np.array(bboxes)
        return bboxes, categories

    def extract_box(self, line):
        raw_label = line.strip("\n").split(",")
        category_name, y1, x1, h, w, height = raw_label
        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        y = int(float(h)) / 2 + int(float(y1))
        x = int(float(w)) / 2 + int(float(x1))
        h = int(float(h))
        w = int(float(w))
        bbox = np.array([y, x, h, w, height], dtype=np.float32)
        return bbox, category_name


