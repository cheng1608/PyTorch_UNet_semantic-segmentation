> WHU大一第三学期实训
> 怎么还没放假啊

### 语义分割，使用U-Net分离背景和人物
数据集是学校给的，很小

属于是难绷之难绷，一开始照着学校给的课件做，点名[这个傻逼博客]([U-Net详解-CSDN博客](https://blog.csdn.net/weixin_55073640/article/details/123060574))，只能说**纯纯傻逼**，代码完全跑不了怎么好意思发的，浪费了一上午，先是`class Decoder`里`forward`函数的 copy and crop操作，也就是下面这部分

```py
def forward(self, high, low):
    x1 = self.up(high)
    offset = x1.size()[2]-low.size()[2]
    padding = 2*[offset//2,offset//2]
    #计算应该填充多少（这里可以是负数）
    x2 = F.pad(low,padding)#这里相当于对低级特征做一个crop操作
    x1 = torch.cat((x1, x2), dim=1)#拼起来
    x1 = self.conv_relu(x1)#卷积走起
    return x1
```
 `padding`乘二是默认图像尺寸一定要一直是偶数吗我请问了，如果是奇数那pad之后还是奇数，后面cat拼接就尺寸不对应了，我跑的时候就是 256*256 的常用图像输入结果一直有如下报错：
 
  *RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 31 for tensor number 1 in th*

打印大小出来就会发现这个计算如果 low的尺寸是奇数就会出错，和助教学长讨论以后找不到好的办法，只能加一个特判，如果offset是奇数就多填充一行一列0
```py
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
```
前向传播能跑出来了，结果输出时发现不对劲，一看大哥你这UNet输出怎么变成 128*128 的了？？？只能说emmmmm

和助教学长反馈以后决定换一个，换成[这个]([【语义分割】unet结构和代码实现_unet模型-CSDN博客](https://blog.csdn.net/weixin_40293999/article/details/129648032))

换完总算能跑了，不过我不会算损失，输出是[5,2,256,256]的，mask是[5,1,256,256]的，实在不知道怎么搞，最后把输出的第二维第二个通道直接丢掉了，变成[5,256,256]，用BCEWithLogitsLoss算 ,等以后弄清楚原理再回来改改
```py
loss = criterion(outputs[:, 0, :, :], y.type(torch.float32).squeeze(1))
```
下面是训练循环的代码，不知道为什么这个acc的算法不行，只能看看loss（其实应该用IOU评估? 不会写
```py
Net = UNet2()
if torch.cuda.is_available():
    Net.to('cuda')
criterion = nn.BCEWithLogitsLoss()
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
```
