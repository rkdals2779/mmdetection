import numpy as np


class Datasets:
    class Hyundai:
        NAME = "hyundai"
        PATH = "/media/falcon/IanBook8T/datasets/hyundai_sample"
        CATEGORIES_TO_USE = ['bump', 'manhole', 'steel', 'pothole']

        ORI_SHAPE = (1080, 1920)
