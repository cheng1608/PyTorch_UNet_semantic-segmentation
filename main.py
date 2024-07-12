import torch
import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm


class ImageSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 确保两个目录中的文件可以一一对应
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

        self.image_paths = [os.path.join(image_dir, img) for img in self.image_files]
        self.mask_paths = [os.path.join(mask_dir, mask) for mask in self.mask_files]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # mask是单通道的

        if self.transform:
            image, mask = self.transform(image), self.transform(mask)

            # 确保mask的数据类型正确（通常是long类型）
        mask = mask.long()

        return image, mask


# 这个数据集（x,y）读取，第一个是source，第二个就是mask

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整为相同的尺寸
    transforms.ToTensor(),
])

# 创建训练和测试数据集
train_dataset = ImageSegmentationDataset(r'D:\py_project\SS1\tem\train_source', r'D:\py_project\SS1\tem\train_mask',
                                         transform=transform)
test_dataset = ImageSegmentationDataset(r'D:\py_project\SS1\tem\test_source', r'D:\py_project\SS1\tem\test_mask',
                                        transform=transform)
print("训练集大小:", len(train_dataset))
print("测试集大小:", len(test_dataset))

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)


##################################################################################################

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # up-conv 2*2
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high, low):
        x1 = self.up(high)
        # print("x1.shape:",x1.shape)
        offset = x1.size()[2] - low.size()[2]
        # print("offset:", offset)
        padding = 2 * [offset // 2, offset // 2]
        if offset % 2 == 1:
            padding[1] += 1
            padding[2] += 1
        # print("padding:",padding)
        # 计算应该填充多少（这里可以是负数）
        # print(low.shape)
        x2 = F.pad(low, padding)  # 这里相当于对低级特征做一个crop操作
        # print("x2.shape:",x2.shape)
        x1 = torch.cat((x1, x2), dim=1)  # 拼起来
        x1 = self.conv_relu(x1)  # 卷积走起
        return x1


class UNet(nn.Module):
    def __init__(self, n_class=2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.decorder4 = Decoder(1024, 512)
        self.decorder3 = Decoder(512, 256)
        self.decorder2 = Decoder(256, 128)
        self.decorder1 = Decoder(128, 64)
        self.last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # Encorder
        layer1 = self.layer1(input)
        # print("layer1.shape:",layer1.shape)
        layer2 = self.layer2(self.maxpool(layer1))
        # print("layer2.shape:",layer2.shape)
        layer3 = self.layer3(self.maxpool(layer2))
        # print("layer3.shape:",layer3.shape)
        layer4 = self.layer4(self.maxpool(layer3))
        # print("layer4.shape:",layer4.shape)
        layer5 = self.layer5(self.maxpool(layer4))
        # print("layer5.shape:",layer5.shape)

        # Decorder
        layer6 = self.decorder4(layer5, layer4)
        # print("layer6.shape:", layer5.shape)
        layer7 = self.decorder3(layer6, layer3)
        # print("layer7.shape:", layer5.shape)
        layer8 = self.decorder2(layer7, layer2)
        # print("layer8.shape:", layer5.shape)
        layer9 = self.decorder1(layer8, layer1)
        # print("layer9.shape:", layer5.shape)
        out = self.last(layer9)  # n_class预测种类数
        # print("outlayer.shape:",out.shape)

        return out


##################################################################################################################33
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, is_pool=True):
        if is_pool:
            x = self.pool(x)
        x = self.conv_relu(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.upconv_relu(x)
        return x


class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 1024)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(128)

        self.conv_2 = Downsample(128, 64)
        self.last = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x1 = self.down1(x, is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x5 = self.up(x5)

        x5 = torch.cat([x4, x5], dim=1)  # 32*32*1024
        x5 = self.up1(x5)  # 64*64*256)
        x5 = torch.cat([x3, x5], dim=1)  # 64*64*512
        x5 = self.up2(x5)  # 128*128*128
        x5 = torch.cat([x2, x5], dim=1)  # 128*128*256
        x5 = self.up3(x5)  # 256*256*64
        x5 = torch.cat([x1, x5], dim=1)  # 256*256*128

        x5 = self.conv_2(x5, is_pool=False)  # 256*256*64

        x5 = self.last(x5)  # 256*256*3
        return x5


######################################################################################


Net = UNet2()
if torch.cuda.is_available():
    Net.to('cuda')
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss
optimizer = optim.Adam(Net.parameters(), lr=0.001)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

num_epochs = 20
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    Net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for image, mask in train_loader:
        if torch.cuda.is_available():
            x, y = image.to('cuda'), mask.to('cuda')
        # print("image.shape:",image.shape)
        # print("mask.shape:",mask.shape)
        # print(y.shape)
        # print(y)

        # 前向
        outputs = Net(x)
        # print("output.shape:",outputs.shape)

        loss = criterion(outputs[:, 0, :, :], y.type(torch.float32).squeeze(1))

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        with torch.no_grad():
            y_pred = torch.argmax(outputs, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / (total * 256 * 256)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    Net.eval()
    with torch.no_grad():
        for x, y in test_loader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            outputs = Net(x)
            loss = criterion(outputs[:, 0, :, :], y.type(torch.float32).squeeze(1))
            y_pred = torch.argmax(outputs, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(test_dataset)
    epoch_test_acc = test_correct / (test_total * 256 * 256)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

for i in range(num_epochs):
    print(f'epoch{i + 1} '
          f'train_loss:{train_loss[i]:4f} '
          f'test_loss:{test_loss[i]:4f} '
          f'test_acc:{test_acc[i]:4f}')

xx = [x for x in range(1,num_epochs+1)]
plt.figure()
plt.plot(xx,train_loss,"r")
plt.show()
