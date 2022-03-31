# utils是工具类，用于加载一些常用的方法
import os
from PIL import Image
import torch
from torch.utils.data.dataloader import DataLoader  # 用于数据加载器，比如之后的batch_size
from torchvision import models, transforms  # 导入预训练模型和图片标准化处理库
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device=', device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一设置大小
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model():
    """
    抽取模型
    :return:预训练模型
    """
    model = models.resnet18(pretrained=True)  # 加载模型，表示训练完成
    model.to(device)  # gpu上运行
    model.eval()  # 只预测不训练
    return model


def feature_extract(model, x):
    """
    定义特征提取器

    x相当于输入的图片张量，model是预训练模型。
    我们的核心思想是：利用resnet的前向传播，
    除了最后一层输出结果以外，提取前面一部分特征向量，
    我们提取前面的特征向量即可。
    看resnet源码提取的！！
    :param model:
    :param x:
    :return:
    """
    x = model.conv1(x)  # 复制后要将self改成model，表示是model的成员
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    # x = model.fc(x)

    return x
