import os
import cv2
import numpy as np


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    calib_path = os.path.join(script_dir, "camera_calibration.npz")
    save_path = os.path.join(script_dir, "camera_pose.npz")

    data = np.load(calib_path, allow_pickle=True)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    # ===== 这里改成你自己点出来的像素点（建议至少 8~12 个）=====
    # 顺序必须和 object_points 一一对应
    image_points = np.array([
        # 第1列（x≈690）从下到上
    [690, 1188],
    [691, 998],
    [695, 801],
    [690, 606],

    # 第2列（x≈886）
    [886, 1191],
    [888, 995],
    [890, 803],
    [888, 610],

    # 第3列（x≈1081）
    [1081, 1191],
    [1081, 1000],
    [1081, 803],
    [1085, 610],

    # 第4列（x≈1275）
    [1275, 1193],
    [1278, 1001],
    [1276, 806],
    [1278, 610],
    ], dtype=np.float32)

    # ===== 对应真实世界坐标（单位统一用 cm）=====
    # 这里示例是假设网格平面 z=0
    object_points = np.array([
       [0, 0, 0],
       [0, 4, 0],
       [0, 8, 0],
       [0, 12, 0],

       [4, 0, 0],
       [4, 4, 0],
       [4, 8, 0],
       [4, 12, 0],

       [8, 0, 0],
       [8, 4, 0],
       [8, 8, 0],
       [8, 12, 0],

       [12, 0, 0],
       [12, 4, 0],
       [12, 8, 0],
       [12, 12, 0],
    ], dtype=np.float32)

    if len(image_points) != len(object_points):
        raise ValueError("image_points 和 object_points 数量必须一致")

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("[ERROR] solvePnP failed.")
        return

    R, _ = cv2.Rodrigues(rvec)

    print("success = ", success)
    print("rvec =\n", rvec)
    print("tvec =\n", tvec)
    print("R =\n", R)

    # ===== 重投影误差检查 =====
    projected_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)

    errors = np.linalg.norm(projected_points - image_points, axis=1)
    mean_error = np.mean(errors)

    print("\nPer-point reprojection error (pixels):")
    for i, e in enumerate(errors):
        print(f"Point {i:02d}: {e:.3f}")

    print(f"\nMean reprojection error: {mean_error:.3f} px")

    np.savez(save_path, rvec=rvec, tvec=tvec, R=R)
    print(f"\nsaved to {save_path}")


if __name__ == "__main__":
    main()