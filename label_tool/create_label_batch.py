# coding: utf-8
import cv2
import numpy as np
import os
import datetime
from imutils import paths


imagePathsTmp = sorted(list(paths.list_images('../images_raw')))
imagePaths = []
for path in imagePathsTmp:
    if path[-3:] == 'png':
        imagePaths.append(path)

print (imagePaths)

# print (imagePaths)
# for imagePath in imagePaths:
#     print (imagePath)
outputPath = './mask/'

# print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
time_now = datetime.datetime.now().strftime("%m%d_%H%M_%S")
print ('time:',time_now)

image = None
image_bak = None
mask = None
mode = 0 # 标注模式
label_flag = 0 # 标注动作标志
save_cnt = -1 # 保存图片数
raw_img_num = 0 # 当前读取的原始图



# 区域标记
def draw_label(event,x,y,flags,param):
    global prev_pt, label_flag, mask

    pt = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        label_flag = 1
        prev_pt = pt
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     label_flag = 0
    elif event == cv2.EVENT_LBUTTONUP and label_flag==1:
        label_flag = 0
    elif event==cv2.EVENT_MOUSEMOVE and label_flag==1:
        cv2.line(image, prev_pt, pt, (255,255,255), 2)
        cv2.line(mask, prev_pt, pt, (255,255,255), 2)
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)


def read_img(img_num=0):
    global image, image_bak, mask
    # print (imagePaths[img_num][-3:])
    # if imagePaths[img_num][-3:] == 'jpg': # 跳过jpg文件
    #     img_num += 1

    image = cv2.imread(imagePaths[img_num], -1).astype(float) / 1000 #读取图像
    image_bak = image.copy() # 创建副本以便恢复
    print (image.shape[0],image.shape[1])

    cv2.imshow("image", image)
    cv2.moveWindow('image',500,350)
    cv2.setMouseCallback('image', draw_label)

    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    cv2.imshow('mask', mask)
    cv2.moveWindow('mask',1300,350)

    return imagePaths[img_num]




# 读取并显示图片
read_img(raw_img_num)

curr_img_name = imagePaths[raw_img_num]
print(curr_img_name)
basename = os.path.basename(curr_img_name)
print(basename)
prefix = basename[:-9]
print(prefix)

while(True):
    key = cv2.waitKey(50)
    # print (key)
    if key == 27: # ESC退出
        break
    # 保存图片
    if key == ord('s'):
        # print ('\n[INFO] Save image time ' + datetime.datetime.now().strftime("%H:%M:%S"))
        save_cnt += 1

        curr_img_name = imagePaths[raw_img_num]
        basename = os.path.basename(curr_img_name)
        prefix = basename[:-9]
        save_path = outputPath + prefix + 'mask.png'
        print("save_path", save_path)

        cv2.imwrite(save_path, mask)
    # 清除标记
    if key == ord('r'):
        print ('[INFO] Back up')
        image= image_bak.copy()
        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
    # if key == 32: # 空格
    # 下一张图片
    if key == ord('d'):
        raw_img_num += 1
        img_name = read_img(raw_img_num)
        print ('[INFO] The %dth image selected:'%(raw_img_num+1), img_name)
    if key == ord('a'):
        raw_img_num -= 1
        if raw_img_num < 0:
            raw_img_num = 0
        img_name = read_img(raw_img_num)
        print ('[INFO] The %dth image selected:'%(raw_img_num+1), img_name)
    # 切换模式
    if key == ord('w'):
        if mode == 0:
            mode = 1
            print ('[INFO] mode: generate test image')
        elif mode == 1:
            mode = 0
            print ('[INFO] mode: generate every image')


