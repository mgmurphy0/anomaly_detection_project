import os
from torch.utils.data import Dataset
from utils import read_image

from skimage.transform import resize
from skimage.feature import canny
from skimage.color import rgb2gray

class ImageDataset(Dataset):

    def __init__(self, img_dir, with_canny = True, transform=None, target_transform=None):
        
        self.img_dir = img_dir

        self.labels = []

        u_label_count = 1

        # Locate all images to include in the data loader
        self.full_image_paths = []
        for f in os.scandir(img_dir):
            if f.is_dir():
                for ff in os.scandir(f.path):
                    self.full_image_paths.append(ff.path)

                    # Assign the correct label to each image
                    if 'good' in f.name:
                        self.labels.append(0)
                    else:
                        self.labels.append(u_label_count)
                
                if 'good' not in f.name:
                    u_label_count += 1
        
        self.with_canny = with_canny
        self.transform = transform

    def __len__(self):
        return len(self.full_image_paths)

    def __getitem__(self, idx):

        '''Preprocessing steps includes:
         - changing from rgb to gray scale
         - reducing image size from 1024 to 256
         - applying canny edge detection (optional)'''

        img = read_image(self.full_image_paths[idx])
        label = self.labels[idx]
        
        # turn to gray
        img = rgb2gray(img)
        
        # resize image
        image = resize(img,(256,256))
        
        # apply  transform on image
        if self.with_canny:
            image = canny(image,sigma=1).astype(float)

        if self.transform:
            image = self.transform(image)

        return image, label