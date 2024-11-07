import torch
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
import sys

sys.path.append('scripts/')
from generics import time

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = vit_b_16(weights=None)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prof = None
    for epoch in range(epochs):
        if epoch == 3:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()
            
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

        print(f'Squashfs Accuracy: {100 * correct / total}%')
        if epoch == 3:
            prof.stop()
            prof.export_chrome_trace("squashfs-trace.json")

@time('visualtransformer-squashfs')
def main():
    from torchvision.datasets import ImageFolder
    
    train_dataset = ImageFolder('/train_images', transform=transform)
    val_dataset = ImageFolder('/val_images', transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    from torch.utils.data import Subset
    fraction = 0.1
    indices = torch.randperm(len(train_dataset))[:int(len(train_dataset) * fraction)]
    reduced_train_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(reduced_train_dataset, batch_size=32, shuffle=True, num_workers=7)

    indices = torch.randperm(len(val_dataset))[:int(len(val_dataset) * fraction)]
    reduced_val_dataset = Subset(val_dataset, indices)
    val_loader = DataLoader(reduced_val_dataset, batch_size=32, shuffle=True, num_workers=7)

    train_model(model, criterion, optimizer, train_loader, val_loader)

    # torch.save(model.state_dict(), 'vit_b_16_imagenet.pth')

    
if __name__ == '__main__':
    main()
