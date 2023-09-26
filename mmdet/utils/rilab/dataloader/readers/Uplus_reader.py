import numpy as np
import cv2

from mmdet.utils.rilab.dataloader.readers.mmdet_reader_base import DataFrameReaderBase
from mmdet.utils.rilab.dataloader.readers.dataset_visualizer import DatasetVisualizer


from mmdet.datasets.uplus import UplusDataset
def swap_and_concat(array1, array2):
    """
    Swap the positions of elements in two arrays and concatenate them.

    Args:
        array1 (ndarray): The first array.
        array2 (ndarray): The second array.

    Returns:
        ndarray: A new array containing the swapped and concatenated elements.
    """
    array1, array2 = array2, array1  # 위치를 바꾸기
    return np.concatenate((array1, array2))

# 두 배열 정의
array1 = np.array([0, 1, 2])
array2 = np.array([3, 4, 5])

# 위치를 바꾸고 하나로 합치기
result = swap_and_concat(array1, array2)

print(result)

class UplusReader(DataFrameReaderBase):
    def get_image(self, index):
        return cv2.imread(self.frame_data[index]['img_path'])

    def get_lane(self, index):
        return self.frame_data[index]['lane']



def uplus_vis():
    a = UplusDataset()

    asd = a.load_data_list()

    uplus_reader = UplusReader(asd, ['1'])
    vis_cfg = {'image': {},
               'lane': {}
               }
    vs = DatasetVisualizer(uplus_reader, vis_cfg)
    for i, frame in enumerate(asd):
        vs.visualize(i, wait=0)

if __name__ == "__main__":
    uplus_vis()
