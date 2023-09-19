import numpy as np


class Paths:
    RESULT_ROOT = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/dataloader"
    DATAPATH = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/dataloader/tfrecord"
    CHECK_POINT = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/dataloader/ckpt"
    CONFIG_FILENAME = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/config.py"
    META_CFG_FILENAME = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/BackUp/uplus_ws/RILabDetector/config_dir/meta_config.py"


class Datasets:
    class Uplus:
        NAME = "uplus"
        PATH = "/media/falcon/IanBook8T/datasets/uplus22_sample"
        CATEGORIES_TO_USE = ['보행자', '승용차', '트럭', '버스', '이륜차', '신호등', '자전거', '삼각콘', '차선규제봉', '과속방지턱', '포트홀', 'TS이륜차금지',
                             'TS우회전금지', 'TS좌회전금지', 'TS유턴금지', 'TS주정차금지', 'TS유턴', 'TS어린이보호', 'TS횡단보도', 'TS좌회전',
                             'TS속도제한_기타', 'TS속도제한_30', 'TS속도제한_50', 'TS속도제한_80', 'RM우회전금지', 'RM좌회전금지', 'RM직진금지',
                             'RM우회전', 'RM좌회전', 'RM직진', 'RM유턴', 'RM횡단예고', 'RM횡단보도', 'RM속도제한_기타', 'RM속도제한_30',
                             'RM속도제한_50', 'RM속도제한_80', "don't care"]

        CATEGORIES_TO_ENG = {"major": ["Pedestrian", "Car", "Truck", "Bus",
                                       "Motorcycle", "Traffic light", "Bicycle", "Cone",
                                       "Lane_stick", "Bump", "Pothole",
                                       "Don't Care", "Lane Don't Care", ],

                             "sign": ["TS_NO_TW", "TS_NO_RIGHT", "TS_NO_LEFT",
                                      "TS_NO_TURN", "TS_NO_STOP",
                                      "TS_U_TURN", "TS_CHILDREN", "TS_CROSSWK",
                                      "TS_GO_LEFT",
                                      "TS_SPEED_LIMIT_ETC", "TS_SPEED_LIMIT_30",
                                      "TS_SPEED_LIMIT_50", "TS_SPEED_LIMIT_80", ],

                             "mark": ["RM_NO_RIGHT", "RM_NO_LEFT", "RM_NO_STR",
                                      "RM_GO_RIGHT", "RM_GO_LEFT", "RM_GO_STR", "RM_U_TURN",
                                      "RM_SPEED_LIMIT_ETC", "RM_SPEED_LIMIT_30",
                                      "RM_SPEED_LIMIT_50", "RM_SPEED_LIMIT_80"], }

        CATEGORY_REMAP = {"보행자": "Pedestrian", "승용차": "Car", "트럭": "Truck",
                          "버스": "Bus", "이륜차": "Motorcycle", "신호등": "Traffic light",
                          "자전거": "Bicycle", "삼각콘": "Cone", "차선규제봉": "Lane_stick",
                          "과속방지턱": "Bump", "포트홀": "Pothole", "don't care": "Don't Care",
                          "lane don't care": "Lane Don't Care", "TS이륜차금지": "TS_NO_TW", "TS우회전금지": "TS_NO_RIGHT",
                          "TS좌회전금지": "TS_NO_LEFT", "TS유턴금지": "TS_NO_TURN", "TS주정차금지": "TS_NO_STOP",
                          "TS유턴": "TS_U_TURN", "TS어린이보호": "TS_CHILDREN", "TS횡단보도": "TS_CROSSWK",
                          "TS좌회전": "TS_GO_LEFT", "TS속도제한_기타": "TS_SPEED_LIMIT_ETC", "TS속도제한_30": "TS_SPEED_LIMIT_30",
                          "TS속도제한_50": "TS_SPEED_LIMIT_50", "TS속도제한_80": "TS_SPEED_LIMIT_80", "RM우회전금지": "RM_NO_RIGHT",
                          "RM좌회전금지": "RM_NO_LEFT", "RM직진금지": "RM_NO_STR", "RM우회전": "RM_GO_RIGHT",
                          "RM좌회전": "RM_GO_LEFT", "RM직진": "RM_GO_STR", "RM유턴": "RM_U_TURN",
                          "RM횡단예고": "RM_ANN_CWK", "RM횡단보도": "RM_CROSSWK", "RM속도제한_기타": "RM_SPEED_LIMIT_ETC",
                          "RM속도제한_30": "RM_SPEED_LIMIT_30", "RM속도제한_50": "RM_SPEED_LIMIT_50",
                          "RM속도제한_80": "RM_SPEED_LIMIT_80",

                          }
        LANE_TYPES = ['차선1', '차선2', '차선3', '차선4', 'RM정지선']
        LANE_REMAP = {"차선1": "Lane", "차선2": "Lane", "차선3": "Lane",
                      "차선4": "Lane", "RM정지선": "Stop_Line",
                      }
        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [300, 0, 0, 0]
        INCLUDE_LANE = True


class Dataloader:
    DATASETS_FOR_TFRECORD = {"kitti": ('train', 'val'),
                             }
    MAX_BBOX_PER_IMAGE = 100
    MAX_DONT_PER_IMAGE = 100
    MAX_LANE_PER_IMAGE = 30
    MAX_POINTS_PER_LANE = 50
    CATEGORY_NAMES = {
        "major": ['Bgd', 'Pedestrian', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Traffic light', 'Traffic sign',
                  'Road mark', 'Road mark dont', 'Road mark do', 'Bicycle', 'Cone', 'Lane_stick', 'Bump', 'Pothole'],
        "sign": ['TS_NO_TW', 'TS_NO_RIGHT', 'TS_NO_LEFT', 'TS_NO_TURN', 'TS_NO_STOP', 'TS_U_TURN', 'TS_CHILDREN',
                 'TS_CROSSWK', 'TS_GO_LEFT', 'TS_SPEED_LIMIT'],
        "mark": ['RM_U_TURN', 'RM_ANN_CWK', 'RM_CROSSWK', 'RM_SPEED_LIMIT'],
        "mark_dont": ['RM_NO_RIGHT', 'RM_NO_LEFT', 'RM_NO_STR'], "mark_do": ['RM_GO_RIGHT', 'RM_GO_LEFT', 'RM_GO_STR'],
        "sign_speed": ['TS_SPEED_LIMIT_30', 'TS_SPEED_LIMIT_50', 'TS_SPEED_LIMIT_80', 'TS_SPEED_LIMIT_ETC'],
        "mark_speed": ['RM_SPEED_LIMIT_30', 'RM_SPEED_LIMIT_50', 'RM_SPEED_LIMIT_80', 'RM_SPEED_LIMIT_ETC'],
        "dont": ["Don't Care"], "lane": ['Bgd', 'Lane', 'Stop_Line'],
        "dont_lane": ["Lane Don't Care"],
        }
    SHARD_SIZE = 2000
    MIN_PIX = {
        "train": {'Bgd': 0, 'Pedestrian': 68, 'Car': 87, 'Truck': 98, 'Bus': 150, 'Motorcycle': 38, 'Traffic light': 41,
                  'Traffic sign': 26, 'Road mark': 20, 'Road mark dont': 20, 'Road mark do': 20, 'Bicycle': 38,
                  'Cone': 34, 'Lane_stick': 30, 'Bump': 75, 'Pothole': 0},
        "val": {'Bgd': 0, 'Pedestrian': 76, 'Car': 98, 'Truck': 110, 'Bus': 168, 'Motorcycle': 42, 'Traffic light': 46,
                'Traffic sign': 29, 'Road mark': 20, 'Road mark dont': 20, 'Road mark do': 20, 'Bicycle': 42,
                'Cone': 38, 'Lane_stick': 34, 'Bump': 85, 'Pothole': 0},
        }
    LANE_MIN_PIX = {"train": {'Bgd': 0, 'Lane': 55, 'Stop_Line': 55}, "val": {'Bgd': 0, 'Lane': 50, 'Stop_Line': 50},
                    }


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    LANE_DET = True
    MINOR_CTGR = True
    SPEED_LIMIT = False
    FEAT_RAW = False
    IOU_AWARE = False
    NUM_ANCHORS_PER_SCALE = 3
    GRTR_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1,
                             "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1,
                             "anchor_ind": 1,
                             }
    PRED_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "distance": 1,
                             "category": 16, "sign_ctgr": 10, "mark_ctgr": 4,
                             "mark_dont": 3, "mark_do": 3,
                             }
    HEAD_COMPOSITION = {"reg": 6, "cls": 36,
                        }
    GRTR_INST_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1,
                             "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1,

                             }
    PRED_INST_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1,
                             "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1,
                             "ctgr_prob": 1, "score": 1, "anchor_ind": 1,

                             }
    NUM_MAIN_CHANNELS = 42
    NUM_LANE_ANCHORS_PER_SCALE = 1
    GRTR_FMAP_LANE_COMPOSITION = {"laneness": 1, "lane_fpoints": 10, "lane_centerness": 1,
                                  "lane_category": 1,
                                  }
    PRED_FMAP_LANE_COMPOSITION = {"laneness": 1, "lane_fpoints": 10, "lane_centerness": 1,
                                  "lane_category": 3,
                                  }
    GRTR_INST_LANE_COMPOSITION = {"lane_fpoints": 10, "lane_centerness": 1, "lane_category": 1,

                                  }
    PRED_INST_LANE_COMPOSITION = {"lane_fpoints": 10, "lane_centerness": 1, "lane_category": 1,

                                  }
    NUM_LANE_CHANNELS = 15

