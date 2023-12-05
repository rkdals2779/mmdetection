import pandas as pd
import os
import statistics
import torch

from collections import defaultdict


class HistoryLogger:
    STEP_INDEX = 0
    EPOCH_INDEX = 0
    DATA_DICT = defaultdict(list)
    DATA_DICT["EPOCH"] = 0

    def __init__(self, count_step=False, count_epoch_and_save=False, save_csv=False):
        self.count_step = count_step
        self.count_epoch_and_save = count_epoch_and_save
        self.save_csv = save_csv
        self.steps_per_epoch = 405
        self.file = '/home/falcon/shin_workspace/Datacleaning/history_log/history_log.csv'

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.count_step:
                HistoryLogger.STEP_INDEX += 1
            if self.count_epoch_and_save:
                HistoryLogger.EPOCH_INDEX += 1

            ret = func(*args, **kwargs)
            self.data_log(ret)
            if self.count_epoch_and_save:
                self.save_structure()
            return ret

        return wrapper

    def data_log(self, data_dict):
        if "recall" in data_dict:
            for key, value in data_dict.items():
                if key in ["TOTAL_recall", "TOTAL_precision"]:
                    HistoryLogger.DATA_DICT[key] = value
                if key in ["recall", "precision"]:
                    for val, category in zip(value, ['bump', 'manhole', 'steel', 'pothole']):
                        HistoryLogger.DATA_DICT[f"{key}_{category}"] = val
                else:
                    HistoryLogger.DATA_DICT[key] = value

        if "loss_cls" in data_dict:
            loss = sum(val.item() for val in data_dict.values())
            HistoryLogger.DATA_DICT["loss"].append(loss)
            for key, val in data_dict.items():
                if torch.is_tensor(val):
                    val = val.item()
                HistoryLogger.DATA_DICT[key].append(val)

    def save_structure(self):
        for key in HistoryLogger.DATA_DICT:
            if "loss" in key:
                HistoryLogger.DATA_DICT[key] = statistics.mean(HistoryLogger.DATA_DICT[key])

        HistoryLogger.DATA_DICT["EPOCH"] = HistoryLogger.EPOCH_INDEX
        df_new = pd.DataFrame(HistoryLogger.DATA_DICT, index=[HistoryLogger.EPOCH_INDEX])

        if os.path.isfile(self.file):
            df_old = pd.read_csv(self.file)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        print("saving history log...")
        df.to_csv(self.file, index=False)

        print(f"STEP: {HistoryLogger.STEP_INDEX}, EPOCH: {HistoryLogger.EPOCH_INDEX}\n"
              f"DATA: {HistoryLogger.DATA_DICT}")
        HistoryLogger.DATA_DICT = defaultdict(list)
        HistoryLogger.DATA_DICT["EPOCH"] = 0



