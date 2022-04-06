import warnings
warnings.filterwarnings('ignore')
import faiss
import os
import time
import numpy as np
import torch
from torchvision import models,transforms
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from utils import transform, load_model, feature_extract  # 加载已经封装好的方法
from dataset import MyDataset
import cv2
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_features(val_dataloader):
    for idx, batch in enumerate(val_dataloader):
        # 之所以能用字典，是因为自定义的dataset里面设置了一个字典便于读取
        # enumerate相当于做取下标索引的操作了
        img = batch['img']  # 图片数据表示 -> 图片特征过程
        index = batch['index']

        img = img.to(device)
        feature = feature_extract(model, img)

        feature = feature.data.cpu().numpy()  # 为了能够顺利存储，要转化为numpy
        imgs_path = [os.path.join(img_folder, val_dataset.data[i] + '.txt') for i in index]  # 提取当前的图片文件

        assert len(feature)==len(imgs_path)  # 保证特征数量和路径数相等
        for i in range(len(imgs_path)):
            feature_list = [str(f) for f in feature[i]]  # 保存当前图片所有特征
            img_path = imgs_path[i]  # 保存当前路径，将来要写的文件

            with open(img_path, 'w', encoding='utf-8') as f:
                f.write(" ".join(feature_list))


def create_index():
    def img2feat(pic_file):
        feat = []
        with open(pic_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
            feat = [float(f) for f in lines[0].split()]
        return feat
    ids = []
    data = []
    img_folder = 'VOC2012_small/'
    img_path = os.path.join(img_folder, 'img.txt')
    with open(img_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            img_name = line.strip()
            img_id = img_name.split('.')[0]
            pic_txt_file = os.path.join(img_folder, "{}.txt".format(img_name))

            if not os.path.exists(pic_txt_file):
                continue
            feat = img2feat(pic_txt_file)
            ids.append(int(img_id))
            data.append(np.array(feat).astype('float32'))

    ids = np.array(ids)
    data = np.array(data)
    d = 512
    index = faiss.index_factory(d, "IDMap,Flat")  # 构建索引，第一个参数表示维度数，第二个表示索引类型
    index.add_with_ids(data, ids)  # 将特征和序号加入索引中

    # 索引文件保存磁盘
    faiss.write_index(index,'index_file.index')
    index = faiss.read_index('index_file.index')
    print(index.ntotal)


def index_search(feat, topK):
    """
    索引查询
    :param feat:检索的图片特征
    :param topK: 返回最高的topK相似的图片
    :return:
    """
    feat = np.expand_dims(np.array(feat), axis=0)  # 特征需要是二维的
    feat = feat.astype('float32')

    start_time = time.time()
    dis, ind = index.search(feat,topK)
    end_time = time.time()

    print('index_search consume time:{}ms'.format(int(end_time - start_time) * 1000))

    return dis,ind


def save_distance(img_path):
    """
        用来保存图片之间的距离关系，用于之后构造知识图谱。
    """
    print("现在在处理的图片编号是：",img_path)
    headers = ["图片id1", "图片id2", "距离"]

    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)  # 扩张第0维，变成了二维
    img = img.to(device)
    with torch.no_grad():
        # 图片 -> 图片特征向量
        print('1.图片特征提取')
        feature = feature_extract(model, img)
        # 特征 -> 检索
        feature_list = feature.data.cpu().tolist()[0]  # 要最后取[0]是因为它本来是个二维矩阵（因为最开始unsqueeze了一下）
        print('2.基于特征的检索，从faiss获取相似度的图片')
        # 相似图片可视化
        dis, ind = index_search(feature_list, topK=topK)
        print('ind = ', ind)
        for i in range(1,len(dis[0]),1):
            rows.append(tuple([img_path[-10:-4], ind[0][i], dis[0][i]]))
    print(rows)
    with open("tuples.csv", "w", encoding="GBK", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == '__main__':
    rows = []
    model = load_model()
    img_folder = 'VOC2012_small/'
    val_dataset = MyDataset(img_folder, transform=transform)
    batch_size = 64
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    topK = 20
    # print("DEBUG1")
    create_features(val_dataloader)
    # print("DEBUG2")
    create_index()
    index = faiss.read_index('index_file.index')
    # print("DEBUG3")
    with open(os.path.join(img_folder, 'img.txt')) as f:
        for line in f.readlines():
            img_id = line.strip()
            img_path = os.path.join(img_folder, img_id)
            save_distance(img_path)