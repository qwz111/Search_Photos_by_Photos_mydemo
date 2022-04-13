# USAGE
# python match.py --template cod_logo.png --images images
# USAGE2 了解实际检测原理及细节
# python match.py --template cod_logo.png --images images --visualize 1

# 导入必要的包
import argparse  # argparse解析命令行参数
import glob  # 获取输入图像的路径

import cv2  # opencv绑定
import imutils  # 图像处理的一些方法
import numpy as np  # numpy进行数值处理

# 构建命令行及解析参数
# --template 模板路径
# --images 原始图像路径
# --visualize 标志是否显示每一个迭代的可视化结果
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
              help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
              help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
# print(args["template"])

# 加载模板图像，转换灰度图，检测边缘
# 使用边缘而不是原始图像进行模板匹配可以大大提高模板匹配的精度。

template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
# print(template)

(tH, tW) = template.shape[:2]
# 遍历图像以匹配模板
# for imagePath in glob.glob(args["images"] + "/*.jpg"):
#   print(imagePath)
#   # 加载图像，转换为灰度图，初始化用于追踪匹配区域的簿记变量
#   image = cv2.imread(imagePath)
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   found = None

#   # 遍历图像尺寸
#   for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#       # 根据scale比例缩放图像，并保持其宽高比
#       resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
#       r = gray.shape[1] / float(resized.shape[1])

#       # 缩放到图像比模板小，则终止
#       if resized.shape[0] < tH or resized.shape[1] < tW:
#           break

#       # 在缩放后的灰度图中检测边缘，进行模板匹配
#       # 使用与模板图像完全相同的参数计算图像的Canny边缘表示；
#       # 使用cv2.matchTemplate应用模板匹配；
#       # cv2.minMaxLoc获取相关结果并返回一个4元组，其中分别包含最小相关值、最大相关值、最小值的（x，y）坐标和最大值的（x，y）坐标。我们只对最大值和（x，y）-坐标感兴趣，所以只保留最大值而丢弃最小值。
#       edged = cv2.Canny(resized, 50, 200)
#       result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
#       (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

#       # 检查是否可视化
#       if args.get("visualize", False):
#           # 在检测到的区域绘制边界框
#           clone = np.dstack([edged, edged, edged])
#           cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
#                         (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
#           cv2.imshow("Visualize", clone)
#           cv2.waitKey(0)

#       # 如果我们找到了一个新的最大校正值，更新簿记变量值
#       if found is None or maxVal > found[0]:
#           found = (maxVal, maxLoc, r)

#   # 解包簿记变量并基于调整大小的比率，计算边界框（x，y）坐标
#   (_, maxLoc, r) = found
#   (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
#   (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

#   # 在检测结果上绘制边界框并展示图像
#   cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
#   cv2.imshow("Image", image)
#   cv2.waitKey(0)
imagePath = '101002.jpg'
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None

# 遍历图像尺寸
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
  # 根据scale比例缩放图像，并保持其宽高比
  resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
  r = gray.shape[1] / float(resized.shape[1])

  # 缩放到图像比模板小，则终止
  if resized.shape[0] < tH or resized.shape[1] < tW:
      break

  # 在缩放后的灰度图中检测边缘，进行模板匹配
  # 使用与模板图像完全相同的参数计算图像的Canny边缘表示；
  # 使用cv2.matchTemplate应用模板匹配；
  # cv2.minMaxLoc获取相关结果并返回一个4元组，其中分别包含最小相关值、最大相关值、最小值的（x，y）坐标和最大值的（x，y）坐标。我们只对最大值和（x，y）-坐标感兴趣，所以只保留最大值而丢弃最小值。
  edged = cv2.Canny(resized, 50, 200)
  result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
  (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

  # 检查是否可视化
#   if args.get("visualize", False):
#       # 在检测到的区域绘制边界框
#       clone = np.dstack([edged, edged, edged])
#       cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
#                         (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
#       cv2.imshow("Visualize", clone)
#     #   cv2.waitKey(0)

  # 如果我们找到了一个新的最大校正值，更新簿记变量值
  if found is None or maxVal > found[0]:
      found = (maxVal, maxLoc, r)

# 解包簿记变量并基于调整大小的比率，计算边界框（x，y）坐标
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# 在检测结果上绘制边界框并展示图像
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)