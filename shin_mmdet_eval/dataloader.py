import os
import glob
import math


class Dataloader:
    def __init__(self, image_folder_path, batch_size=1):
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size

    def imgs_list_load(self):
        img_paths = glob.glob(os.path.join(self.image_folder_path, "*.png"))
        img_paths.sort()
        imgs_list = []
        num_batch = math.ceil(len(img_paths)/self.batch_size)

        for i in range(num_batch):
            if i == num_batch-1:
                imgs_list.append(img_paths[i * self.batch_size:])
            else:
                imgs_list.append(img_paths[i*self.batch_size:(i+1)*self.batch_size])
        return imgs_list



