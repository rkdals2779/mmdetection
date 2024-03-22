import glob
import os.path as op

datapath = '/media/falcon/IanBook8T/datasets/hyundai_sample'
dirlist = glob.glob(op.join(datapath, '**', 'image'), recursive=True)
print(op.join(datapath, '**', 'image'))
print(dirlist)
