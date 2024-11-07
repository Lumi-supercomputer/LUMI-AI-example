import torch
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
import sys
from tfrecord.torch.dataset import TFRecordDataset
from PIL import Image
import six

sys.path.append('scripts/')
from generics import time

# Define transformations
def transform(dictionary):
    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    buf = six.BytesIO()
    buf.write(dictionary['image'])
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    label = torch.tensor(dictionary['label'][0], dtype=torch.long)
    return (_transform(image), label)

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

        print(f'Epoch {epoch+1}')

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

        print(f'tfrecords Accuracy: {100 * correct / total}%')
        if epoch == 3:
            prof.stop()
            prof.export_chrome_trace("tfrecords-trace.json")

@time('visualtransformer-tfrecords')
def main():
    folder = 'data-formats/tfrecords/'
    train_dataset = TFRecordDataset(folder + 'train/train.tfrecord', folder + 'train/train.tfindex',
                                    shuffle_queue_size=1000, transform=transform)
    val_dataset = TFRecordDataset(folder + 'val/val.tfrecord', folder + 'val/val.tfindex',
                                  shuffle_queue_size=100, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=7)
    
    train_model(model, criterion, optimizer, train_loader, val_loader)

    # torch.save(model.state_dict(), 'vit_b_16_imagenet.pth')

    
if __name__ == '__main__':
    main()
