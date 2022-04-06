# Search_Photos_by_Photos_mydemo
## 功能概述
- 本系统是一个以图搜图系统，本系统的输入是一张待查找的图片，输出是与之相类似的一组图片。
- 相比于原先传统的以图搜图系统，本系统对于包含关系的图片进行了特殊的判定，会优先输出包含关系的图片，然后再输出类似的图片。
## 操作方法
- 首先安装本系统所需要的依赖库:
    ```shell
    pip install -r requirements.txt
    ```
- 安装完相关依赖库以后，输入一下命令执行main.py

    ```shell
    python main.py
    ```

## 各文件的作用
### main.py
- 主程序，执行整个系统流程。
### dataset.py
- 构造pytorch所需要的数据集格式内容。
### utils.py
- 一些图像的操作封装函数、图像特征提取方法。
### save_distance.py
- 预处理，用来保存图像之间两两的距离。