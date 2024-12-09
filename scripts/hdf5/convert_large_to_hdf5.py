import os, sys, io
import time
import h5py
import numpy as np
import torchvision.transforms as transforms

sys.path.append('scripts/')
from generics import time
from generics import ZipFolder


def create_hdf5(image_folder, output_file):
    dataset = ZipFolder(image_folder, transform=transforms.ToTensor())
    num_images = len(dataset)
    
    with h5py.File(output_file, 'w') as h5f:
        # Create datasets for images and labels
        images = h5f.create_dataset('images', (num_images, 3, 64, 64), dtype='f', compression="gzip")
        labels = h5f.create_dataset('labels', (num_images,), dtype='i')
        
        for i, (img, label) in enumerate(dataset):
            images[i] = img.numpy()
            labels[i] = label
            if i % 1000 == 0:
                print(f'Processed {i} images')


@time('convert_to_hdf5')
def main():
    folder_in = '/project/project_462000002/LUMI-AI-example/imagenet-object-localization-challenge.zip'
    folder_out = '/scratch/project_462000002/joachimsode/file-format-ai-benchmark/imagenet-object-localization-challenge.hdf5'
    
    #folder_in = 'data-formats/raw/tiny-imagenet-200.zip'
    #folder_out = 'data-formats/images.hdf5'
    
    create_hdf5(folder_in, folder_out)
    
if __name__ == '__main__':
    main()
    # image_folder = 'data-formats/raw/tiny-imagenet-200.zip'
    # image_folder = '/project/project_462000002/LUMI-AI-example/imagenet-object-localization-challenge.zip'
    # dataset = ZipFolder(image_folder, transform=transforms.ToTensor())
