import os, sys
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from pathlib import Path
import tensorflow as tf

sys.path.append('scripts/')
from generics import time

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(image, label):
    feature = {
        "image": image_feature(image),
        "label": int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tfrecords(image_folder, output_file, name=''):
    # num_samples: the number of data samples on each TFRecord file
    
    dataset = ImageFolder(image_folder + name + '/')
    num_images = len(dataset)
        
    with tf.io.TFRecordWriter(
            output_file + f"/{name}.tfrecord"
    ) as writer:
        for i in range(num_images):
            image, label = dataset[i]
            example = create_example(image, label)
            writer.write(example.SerializeToString())
            if i % 100 == 0:
                print(f'Processed {i} images')
                

# def create_tfrecords(image_folder, output_file, name='', num_samples=4096):
#     # num_samples: the number of data samples on each TFRecord file
    
#     dataset = ImageFolder(image_folder + name + '/')
#     num_images = len(dataset)
#     num_tfrecords = num_images // num_samples
        
#     for tfrec_num in range(num_tfrecords):
#         with tf.io.TFRecordWriter(
#                 output_file + f"/file_{tfrec_num}.tfrecord"
#         ) as writer:
#             for sample_num in range(num_samples):
#                 i = tfrec_num * num_samples + sample_num
#                 image, label = dataset[i]
#                 example = create_example(image, label)
#                 writer.write(example.SerializeToString())
#                 if i % 100 == 0:
#                     print(f'Processed {i} images')

#     remainder = num_images % num_samples
#     if remainder > 0:
#         with tf.io.TFRecordWriter(
#                 output_file + f"/file_{tfrec_num}.tfrecord"
#         ) as writer:
#             for remainder_num in range(remainder):
#                 i = num_tfrecords * num_samples + remainder_num
#                 image, label = dataset[i]
#                 example = create_example(image, label)
#                 writer.write(example.SerializeToString())
#                 if i % 100 == 0:
#                     print(f'Processed {i} images')

@time('convert_to_tfrecords')                    
def main():
    folder_in = 'data-formats/raw/tiny-imagenet-200/'
    folder_out = 'data-formats/tfrecords/'
    train_out = folder_out + 'train/'
    val_out = folder_out + 'val/'
    
    if not os.path.exists(train_out):
        os.makedirs(train_out)  # creating TFRecords output folder
    if not os.path.exists(val_out):
        os.makedirs(val_out)  # creating TFRecords output folder

    
    create_tfrecords(folder_in, train_out, 'train')
    create_tfrecords(folder_in, val_out, 'val')

    
if __name__ == '__main__':
    
    main()
