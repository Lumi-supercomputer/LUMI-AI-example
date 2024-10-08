import torch
import os
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from hdf5_dataset import HDF5Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import psutil


def set_cpu_affinity(local_rank):
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)


dist.init_process_group(backend='nccl')

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
rank = int(os.environ["RANK"])
set_cpu_affinity(local_rank)

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


model = vit_b_16(weights=None).to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if rank == 0:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if rank == 0:
            print(f'Accuracy: {100 * correct / total}%')
        scheduler.step()


with HDF5Dataset('train_images.hdf5', transform=transform) as train_dataset, \
     HDF5Dataset('val_images.hdf5', transform=transform) as val_dataset:

    from torch.utils.data import Subset
    fraction = 1.0
    indices = torch.randperm(len(train_dataset))[:int(len(train_dataset) * fraction)]
    reduced_train_dataset = Subset(train_dataset, indices)
    train_sampler = DistributedSampler(reduced_train_dataset)
    train_loader = DataLoader(reduced_train_dataset, sampler=train_sampler, batch_size=8, num_workers=7)

    indices = torch.randperm(len(val_dataset))[:int(len(val_dataset) * fraction)]
    reduced_val_dataset = Subset(val_dataset, indices)
    val_sampler = DistributedSampler(reduced_val_dataset)
    val_loader = DataLoader(reduced_val_dataset, sampler=val_sampler, batch_size=8, num_workers=7)

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader)

    dist.destroy_process_group()

torch.save(model.state_dict(), 'vit_b_16_imagenet.pth')
