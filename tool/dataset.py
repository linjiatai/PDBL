import os
import sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    
    def __init__(self, data_path, n_class):
        self._data_path = data_path
        self.n_class = n_class
        self._normalize = True
        self._resize224 = transforms.Resize((224,224))
        self._resize160 = transforms.Resize((160,160))
        self._resize112 = transforms.Resize((112,112))
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._pre_process()

    def _pre_process(self):
        # find classes
        if sys.version_info >= (3, 5):
            # Faster and available in python 3.5 and above
            classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        # make dataset
        self._items = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                order = sorted(fnames)
                for fname in order:
                    if (fname.split('.')[-1] == 'tif') or (fname.split('.')[-1] == 'jpeg')or (fname.split('.')[-1] == 'png')or (fname.split('.')[-1] == 'jpg'):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        self._items.append(item)
        self._num_images = len(self._items)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path, label = self._items[idx]
        label = np.array(label, dtype=float)
        img = Image.open(path).convert('RGB')

        img1 = self._resize224(img)
        img2 = self._resize160(img)
        img3 = self._resize112(img)

        img1 = np.array(img1, dtype=np.float32).transpose((2, 0, 1))
        if self._normalize:
            img1 = (img1 - 128.0) / 128.0
            
        img2 = np.array(img2, dtype=np.float32).transpose((2, 0, 1))
        if self._normalize:
            img2 = (img2 - 128.0) / 128.0
            
        img3 = np.array(img3, dtype=np.float32).transpose((2, 0, 1))
        if self._normalize:
            img3 = (img3 - 128.0) / 128.0            
        label_onehot = np.zeros((self.n_class))

        label_onehot[int(label)] = 1
        return img1, img2, img3, label_onehot