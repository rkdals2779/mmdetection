import numpy as np
import cv2
import os.path as op
import mmdet_rilab.config as cfg


class Visualizer:
    def __init__(self):
        self.classes = ['bump', 'manhole', 'steel', 'pothole']
        self.save_path = cfg.PROJECT_ROOT + "/visual_log"

    def __call__(self, img_name, splits):
        gt_image = cv2.imread(img_name)
        pred_image = cv2.imread(img_name)
        pr_tp = splits["pred_tp"]
        pr_fp = splits["pred_fp"]
        gt_tp = splits["grtr_tp"]
        gt_fn = splits["grtr_fn"]
        colors = ((0, 255, 0), (0, 0, 255))

        for i, color in zip([gt_tp, gt_fn], colors):
            gt_image = self.draw_bboxes(gt_image, i, color, gtpr="gt")

        for i, color in zip([pr_tp, pr_fp], colors):
            pred_image = self.draw_bboxes(pred_image, i, color, gtpr="pred")

        gt_image = gt_image[600:, ...]
        pred_image = pred_image[600:, ...]

        image = np.vstack([gt_image, pred_image])
        cv2.imshow("image", image)
        cv2.imwrite(op.join(self.save_path, (img_name.split('/')[-3] + "_" + img_name.split('/')[-1])), image)
        cv2.waitKey(0)

    def draw_bboxes(self, image, tpfpfn, color, gtpr):
        if gtpr == "gt":
            for bbox, labels, heights in zip(tpfpfn["bboxes"][0].astype(np.int16), tpfpfn["labels"][0], tpfpfn["heights"][0]):
                ctgr = self.classes[int(labels[0])]
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                image = cv2.putText(image, ctgr + "_" + str(round(heights[0], 3)),
                                    (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        elif gtpr == "pred":
            for bbox, labels, heights, score in zip(tpfpfn["bboxes"][0].astype(np.int16),
                                                    tpfpfn["labels"][0], tpfpfn["heights"][0], tpfpfn["scores"][0]):
                ctgr = self.classes[int(labels[0])]
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                image = cv2.putText(image, ctgr + "_" + str(round(heights[0], 3)) + "_" + str(round(score[0], 3)),
                                    (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image
