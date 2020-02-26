# coding: utf-8
import cv2
import numpy as np
import os
import glob

dataset_path = "../images_raw/tabel_obj_data"
exe_path = "./create_label_by_pcd/build/create_label_by_pcd"

imgs_save_dir = os.path.join(dataset_path, "imgs")
masks_save_dir = os.path.join(dataset_path, "masks")

workspace = [-2, 2, -2, 2, 0.27, 2]
out_size = (256, 256)

# 获取所有深度图地址
fls_depth = []
for dirpath, dirnames, files in os.walk(dataset_path):
    for f in files:
        if f.endswith('.png'):
            fls_depth.append(os.path.join(dirpath, f))
print("fls_depth", fls_depth)

cnt = 0
for depth in fls_depth:
    cnt += 1

    color = depth[:-9] + 'color.jpg'
    pose = depth[:-9] + 'pose.txt'
    dir = os.path.dirname(depth)

    new_name = "%05i.jpg" % cnt
    imgs_save_path = os.path.join(imgs_save_dir, new_name)
    masks_save_path = os.path.join(masks_save_dir, new_name)

    cmd = "%s %s %s %s %s %s %s %s %s %s %s %s" % (exe_path, color, depth, dir, pose, masks_save_path,
                                                   workspace[0], workspace[1], workspace[2], workspace[3], workspace[4], workspace[5])

    # print(cmd)
    os.system(cmd)

    # 深度图转换为灰度图
    depth_img = cv2.imread(depth, -1)  # 定义图片位置
    # print("input type:", depth_img.dtype)
    # print("input shape:", depth_img.shape)

    depth_img = depth_img * (255 / 1000)  # mm->m (0.0-1.0)->(0, 255)
    depth_img = depth_img.astype(np.uint8)

    depth_img_resize = cv2.resize(depth_img, out_size)
    cv2.imwrite(imgs_save_path, depth_img_resize)


# 调整mask size
masks_paths = sorted(glob.glob(os.path.join(masks_save_dir, '*.jpg')))
for mask_path in masks_paths:
    mask = cv2.imread(mask_path, -1)
    mask_resize = cv2.resize(mask, out_size)
    cv2.imwrite(mask_path, mask_resize)
