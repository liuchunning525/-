import cv2
import numpy as np

img_pts = np.array([
    [418, 251],   # 左上
    [391, 1246],  # 左下
    [1785, 288],   # 右上
    [1760, 1268]   # 右下
], dtype=np.float32)

world_pts = np.array([
    [0, 21],   # 左上
    [0, 0],   # 左下
    [29, 21],   # 右上
    [29, 0]    # 右下
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)

print("Homography matrix:")
print(H)

np.save("H_matrix.npy", H)