import os
import h5py
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def create_hdf5(image_folder, output_file):
    dataset = ImageFolder(image_folder, transform=transforms.ToTensor())
    num_images = len(dataset)
    
    with h5py.File(output_file, 'w') as h5f:
        # Create datasets for images and labels
        images = h5f.create_dataset('images', (num_images, 3, 64, 64), dtype='f')
        labels = h5f.create_dataset('labels', (num_images,), dtype='i')
        
        for i, (img, label) in enumerate(dataset):
            images[i] = img.numpy()
            labels[i] = label
            if i % 100 == 0:
                print(f'Processed {i} images')

create_hdf5('/home/gregor/Documents/imagenet/tiny-imagenet-200/train', 'train_images.hdf5')
create_hdf5('/home/gregor/Documents/imagenet/tiny-imagenet-200/val', 'val_images.hdf5')
