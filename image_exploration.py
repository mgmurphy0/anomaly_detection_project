
import os
from utils import read_image, clean_plot

from skimage.transform import resize
from skimage.filters import laplace, sobel
from skimage.feature import canny
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

import numpy as np

'''This file is a scratchpad to become familiar with the dataset and investigate image preprocessing techniques'''

main_dir = os.path.join('anomaly_dataset','metal_plate')

# Augment and display a selected image
if False:
    image_path = os.path.join(main_dir,'train','good','000.png')
    img = read_image(image_path)
    img = rgb2gray(img)
    clean_plot(img)
    
    # resize image
    img = resize(img,(256,256))
    clean_plot(img)
    
    # apply laplace transform on image
    limg = canny(img,sigma=0.5)
    clean_plot(limg)

    limg = sobel(img)
    clean_plot(limg)

# Save augmented image for each image within a "train" directory
# Or plot the number of pixels identified with Canny
if True:    

    train_dir = os.path.join(main_dir,'train','good')
    train_fnames = os.listdir(train_dir)

    main_save_path = os.path.join(main_dir + '_extra2','train')

    #edge_pixel_count = np.zeros(len(train_fnames))
    edge_pixel_count = []
    truth = []
    for fidx in range(len(train_fnames)):
        
        f = train_fnames[fidx]

        print(f)

        full_path = os.path.join(train_dir,f)
        
        img = read_image(full_path)
        img = rgb2gray(img)
        #clean_plot(img)
        
        # resize image
        img = resize(img,(256,256))
        #clean_plot(img)
        
        # apply  transform on image
        limg = canny(img,sigma=0.5)
        
        #edge_pixel_count[fidx] = sum(sum(limg))
        edge_pixel_count.append(sum(sum(limg)))
        truth.append(0)

        #clean_plot(limg,save_path = os.path.join(main_save_path,f))
    plt.scatter(range(len(train_fnames)),edge_pixel_count,c=truth)
    plt.show()


# Save augmented image for each image within a "test" directory
# Or plot the number of pixels identified with Canny
if True:
    test_dir = os.path.join(main_dir,'test')
    subdir_names = [ f.path for f in os.scandir(test_dir) if f.is_dir() ]
    #output = [ ff.path for f in os.scandir(test_dir) if f.is_dir() for ff in os.scandir(f.path) ]

    # output = []
    # for f in os.scandir(test_dir):
    #     if f.is_dir():
    #         for ff in os.scandir(f.path):
    #             output.append(ff.path)

    main_save_path = os.path.join(main_dir + '_extra2','test')

    test_edge_pixel_count = []
    test_truth = []

    sd_count = 0
    for sd in subdir_names:
        sd_count += 1
        for f in os.listdir(sd):
            
            print(f)

            full_path = os.path.join(sd,f)
            
            img = read_image(full_path)
            img = rgb2gray(img)
            #clean_plot(img)
            
            # resize image
            img = resize(img,(256,256))
            #clean_plot(img)
            
            # apply  transform on image
            limg = canny(img,sigma=0.5)
            #clean_plot(limg,save_path = os.path.join(main_save_path,str(sd_count)+'_'+f))

            test_edge_pixel_count.append(sum(sum(limg)))
            test_truth.append(sd_count)

    plt.scatter(range(len(test_edge_pixel_count)),test_edge_pixel_count,c=test_truth)
    plt.show()
