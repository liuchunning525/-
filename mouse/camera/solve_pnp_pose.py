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
        [603.0, 726.0],
        [790.0, 731.0],
        [978.0, 735.0],
        [1210.0, 740.0],

        [606.0, 536.0],
        [795.0, 545.0],
        [985.0, 550.0],
        [1216.0, 553.0],

        [608.0, 303.0],
        [796.0, 305.0],
        [986.0, 315.0],
        [1223.0, 320.0],
    ], dtype=np.float32)

    # ===== 对应真实世界坐标（单位统一用 cm）=====
    # 这里示例是假设网格平面 z=0
    object_points = np.array([
        [4.0, 11.0, 0.0],
        [8.0, 11.0, 0.0],
        [12.0, 11.0, 0.0],
        [17.0, 11.0, 0.0],

        [4.0, 15.0, 0.0],
        [8.0, 15.0, 0.0],
        [12.0, 15.0, 0.0],
        [17.0, 15.0, 0.0],

        [4.0, 20.0, 0.0],
        [8.0, 20.0, 0.0],
        [12.0, 20.0, 0.0],
        [17.0, 20.0, 0.0],
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