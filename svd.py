# %load svd.py
import numpy as np
from PIL import Image
import os
from glob import glob
from torchvision import transforms

join = os.path.join
listdir = os.listdir
splitext = os.path.splitext
imgs_dir = "/home/sj/workspace/m/MA_NET/LITS/train/raw/"
ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
img_m = np.zeros([512, 512])

# img_m.shape
postfix = ids[0]
img_file = glob(imgs_dir + postfix + '.*')
img = Image.open(img_file[0])
# img.show()
img_nd = np.array(img)
img_trans = img_nd/255
transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-1.5,), (1.0,))
            transforms.Normalize((0.5,), (0.5,))
        ])
# transform(img_trans).float()
# print(img_trans)
U, s, V = np.linalg.svd(img_trans)
# print('U: {}'.format(U))
print('s: {}'.format(s))
# print('V: {}'.format(V))
