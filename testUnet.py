#!/usr/bin/env python
# coding: utf-8

from model import *
from data import *

import time


# test your model and save predicted results
print("[INFO] test unet")

# 测试图片列表
test_paths = sorted(glob.glob(os.path.join("data/test/imgs",'*.jpg')))
print(test_paths)
# 测试图片数量
imagenum_test = len(test_paths)
# 测试图片生成器
testGene = testGenerator(test_paths)
model = unet()
model.load_weights("model.hdf5")

start = time.time()
results = model.predict_generator(testGene,imagenum_test,verbose=1)
print("FPS:", (time.time()-start)/imagenum_test)

saveResult("data/test/test_output", test_paths, results)

# # 测试单张图片
# model = unet()
# model.load_weights("model.hdf5")
# results = model.predict(image_get(as_gray = True)) # shape: (1, 256, 256, 1)
# img = results[0,:,:,0]
# # io.imsave("predict.png",img)
# cv2.imshow('predict',img)
# cv2.waitKey(0)

