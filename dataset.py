# 数据集的初始化
import os
from PIL import Image
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    """
        自定义MyDataset，主要实现三个功能：初始化内容、得到索引、返回长度
    """

    def __init__(self, data_path, transform=None):
        # 初始化信息，加载相关的数据内容
        super().__init__()
        self.transform = transform
        self.data_path = data_path  # 这个是保存所有图片路径的文件
        self.data = []  # 存储的是图片文件名

        img_path = os.path.join(data_path, 'img.txt')
        with open(img_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                img_name = os.path.join(data_path, line)

                img = Image.open(img_name)
                if img.mode == 'RGB':
                    self.data.append(line)

    def __getitem__(self, idx):
        # 建立索引，利用它读取数据
        img_path = os.path.join(self.data_path, self.data[idx])

        # 读图片
        img = Image.open(img_path)

        # 应用变换函数
        if self.transform:
            img = self.transform(img)

        # 返回图片张量内容和索引号（用字典）,这个到后面加载的时候有用！
        dict_data = {
            'index': idx,
            'img': img
        }
        return dict_data

    def __len__(self):
        # 返回长度
        return len(self.data)
