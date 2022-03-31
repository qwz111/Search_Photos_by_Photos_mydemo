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


def search_returnPoint(img, template, template_size):
    # print(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if template.shape[0]>img.shape[0] or template.shape[1]>img.shape[1] or template.shape[2]>img.shape[2]:
        return None, None, None
    result = cv2.matchTemplate(img_gray, template_, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    # res大于70%
    loc = np.where(result >= threshold)
    # 使用灰度图像中的坐标对原始RGB图像进行标记
    point = ()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + template_size[1], pt[1] + + template_size[0]), (7, 249, 151), 2)
        point = pt
    if point == ():
        return None, None, None
    return img, point[0] + template_size[1] / 2, point[1]


def find_include_img(img_path):
    scale = 1

    img = cv2.imread(img_path)  # 要找的大图
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # template = cv2.imread('./img_1.png')  # 图中的小图
    # template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
    # template_size = template.shape[:2]

    template_list = []
    with open("VOC2012_small/img.txt","r") as f:
        for line in f.readlines():
            img_name = line.strip()
            template_list.append("VOC2012_small/" + img_name)
    for file_name in template_list:
        if file_name == img_path:
            continue
        print(file_name)
        template = cv2.imread(file_name)  # 图中的小图
        print(template.shape)
        template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        template_size = template.shape[:2]
        return_img, x_, y_ = search_returnPoint(img, template, template_size)
        if (return_img is None):
            print("{}和原图没有包含关系".format(file_name))
        else:
            print("找到图片 位置:" + str(x_) + " " + str(y_))
            include_img.append(int(file_name.split('.')[0][-6:]))
            # plt.figure()
            # plt.imshow(img, animated=True)
            # plt.show()


def visual_plot(ind, dis, topK, query_img=None):
    """
    输入图片id和距离
    """
    # 相似照片
    cols = 4
    rows = int(topK / cols)
    idx = 0

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), tight_layout=True)
    # axes[0,0].imshow(query_img)

    for row in range(rows):  # 打印格式
        for col in range(cols):
            _id = ind[0][idx]
            _dis = dis[0][idx]

            img_path = os.path.join(img_folder, '{}.jpg'.format(_id))
            # print(img_path)

            if query_img is not None and idx == 0:
                axes[row, col].imshow(query_img)
                axes[row, col].set_title('query', fontsize=20)
            else:
                img = plt.imread(img_path)
                axes[row, col].imshow(img)
                # axes[row, col].set_title('matched_-{}_{}'.format(_id, int(_dis)), fontsize=20)
                axes[row, col].set_title('matched_-{}'.format(_id), fontsize=20)
            idx += 1
    plt.savefig('pic')


if __name__ == '__main__':
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
    img_id = '100212.jpg'
    img_path = os.path.join(img_folder, img_id)

    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)  # 扩张第0维，变成了二维
    img = img.to(device)
    include_img = []  # 出现包含子图的情况
    print("预处理：查找有没有包含关系的图片")
    find_include_img(img_path)
    print('include=', include_img)

    with torch.no_grad():
        # 图片 -> 图片特征向量
        print('1.图片特征提取')
        feature = feature_extract(model, img)
        # 特征 -> 检索
        feature_list = feature.data.cpu().tolist()[0]  # 要最后取[0]是因为它本来是个二维矩阵（因为最开始unsqueeze了一下）
        print('2.基于特征的检索，从faiss获取相似度的图片')
        # 相似图片可视化
        dis, temp_ind = index_search(feature_list, topK=topK)
        # dis, ind = index_search(feature_list, topK=topK)
        ind = [temp_ind[0][0]]
        for num in include_img:
            ind.append(num)
        for i in range(1, len(temp_ind[0]), 1):
            if not (temp_ind[0][i] in ind):
                ind.append(temp_ind[0][i])
        ind = np.expand_dims(np.array(ind), axis=0)
        print('ind = ', ind)
        print('3.图片可视化展示')
        # 当前图片
        query_img = plt.imread(img_path)  # 当前图片
        visual_plot(ind, dis, topK, query_img)  # 相似图片
