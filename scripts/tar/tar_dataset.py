import tarfile
import six
from pathlib import Path
from typing import Union
from numpy import zeros

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TarDataset(Dataset):
    def __init__(self, file_path: Union[str, Path], transform=None):
        self.file_path = Path(file_path)
        with open(self.file_path.with_suffix('.meta'), 'r') as fd:
            self.image_names = [line.strip('\n') for line in fd.readlines()]
        self.tar_files = {}
        self.transform = transform

    def __enter__(self):
        # self.open_tar_file = tarfile.open(self.file_path)
        self.open_tar_file = tarfile.open(self.file_path)
        self.tar_files[self.file_path] = self.open_tar_file

        return self

    def __exit__(self, type, value, traceback):
        self.open_tar_file.close()
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # The complicated code to extract a png file from a tar file
        if self.file_path in self.tar_files:  # If file has been opened before then use it
            self.open_tar_file = self.tar_files[self.file_path]
        else:  # If not then open the tar file and store the TarFile object for future use
            self.open_tar_file = tarfile.open(self.file_path)
            self.tar_files[self.file_path] = self.open_tar_file
        
        byte_image = self.open_tar_file.extractfile(self.image_names[index]).read()
        buf = six.BytesIO()
        buf.write(byte_image)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        label = torch.tensor(0, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    import torchvision.transforms as transforms
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    folder = 'data-formats/tar/'
    data = 'val'
    file_path = folder + f'{data}.tar'
    # with TarDataset(file_path, transform=transform) as dataset:
    #     test = [dataset[i] for i in range(10)]
    #     print(test)
        
    with tarfile.open(file_path) as open_tar_file:
        image_names = [name for name in open_tar_file.getnames()
                       if Path(name).suffix == '.JPEG']
        with open(folder + f'{data}.meta', 'w') as fd:
            for image_name in image_names:
                fd.write(str(image_name)  + '\n')


    with open(folder + f'{data}.meta', 'r') as fd:
        lines = fd.readlines()
        image_names = [i.strip('\n') for i in lines]
        print(image_names[:10])
