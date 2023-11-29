import numpy as np


class Paths:
    RESULT_ROOT = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/dataloader"
    DATAPATH = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/dataloader/tfrecord"
    CHECK_POINT = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/dataloader/ckpt"
    CONFIG_FILENAME = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/config.py"
    META_CFG_FILENAME = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/config_dir/meta_config.py"


class Datasets:
    class Hyundai:
        NAME = "hyundai"
        PATH = "/media/falcon/IanBook8T/datasets/hyundai_sample"
        CATEGORIES_TO_USE = ['bump', 'manhole', 'steel', 'pothole']

        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [300, 0, 0, 0]
        INCLUDE_LANE = True
        ORI_SHAPE = (1080, 1920)
