import cv2
import numpy as np
import config as cfg


class Visualization:
    def __init__(self):
        self.classes = ['bump', 'manhole', 'steel', 'pothole']
        self.save_path = cfg.VIS_SAVE_PATH

    def __call__(self, img_or_path, pred):
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        self.draw_bboxes(img, pred)
        img = cv2.resize(img, dsize=None, fx=0.7, fy=0.7)
        cv2.imshow("pred", img)
        cv2.waitKey(1)

    def draw_bboxes(self, image, pred):
        color = (0, 255, 0)
        for bbox, label, height, score in zip(pred.bboxes, pred.labels, pred.heights, pred.scores):
            bbox = list(map(round, bbox.tolist()))
            ctgr = self.classes[int(label)]
            height = float(height)
            score = float(score)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            image = cv2.putText(image, ctgr + "_" + str(round(height, 3)) + "_" + str(round(score, 3)),
                                (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print()


