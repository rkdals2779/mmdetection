import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATASET_PATH = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/ActiveDrive/HYD_2023/hyundai_v3_FINAL/val/image"

VIS_SAVE_PATH = "/home/falcon/shin_work/Inference_hyundai/logs/visual_log"

MMDET_CONFIG_PATH = PROJECT_ROOT + "/mmdet_files/yolox_s_8xb8-300e_hyundai/yolox_s_8xb8-300e_hyundai.py"

MMDET_CHECKPOINT_PATH = "/home/falcon/shin_work/MMdetectionHyundai/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_hyundai/hyundai_v3_240312_pth_before_head/epoch_300.pth"


BATCH_SIZE = 1
