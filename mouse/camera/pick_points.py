import cv2

img = cv2.imread("grid.jpg")
scale = 0.6

h, w = img.shape[:2]
show = cv2.resize(img, (int(w * scale), int(h * scale)))

points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ox = int(x / scale)
        oy = int(y / scale)
        points.append((ox, oy))
        print(f"pixel: {ox} {oy}")
        cv2.circle(show, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("grid", show)

cv2.namedWindow("grid", cv2.WINDOW_NORMAL)
cv2.imshow("grid", show)
cv2.setMouseCallback("grid", on_mouse)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected points:", points)