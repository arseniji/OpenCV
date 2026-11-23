import cv2 as cv
import ctypes
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("images/photo.png")
if image is None:
    raise ValueError("Не удалось открыть photo.png")
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()

small = cv.resize(image,None, fx = 0.5,fy = 0.5, interpolation=cv.INTER_AREA)
bic = cv.resize(image,None, fx = 2,fy = 2, interpolation=cv.INTER_LINEAR)
cv.imshow("dih",small)
cv.imshow("doh",bic)

n_height, n_width = image.shape[:2]
s_height, s_width = small.shape[:2]
b_height, b_width = bic.shape[:2]

print(n_height,n_width)
print(s_height,s_width)
print(b_height,b_width)


cv.waitKey(0)
cv.destroyAllWindows()

matrix = np.float32([[1, 0, 100],
                     [0, 1, 50]])

shifted = cv.warpAffine(image, matrix, (n_width, n_height))
cv.imshow("shifted",shifted)
cv.waitKey(0)
cv.destroyAllWindows()

center = ((n_width - 1) / 2, (n_height - 1) / 2)
matrix = cv.getRotationMatrix2D(center, 30, 1)
rotated = cv.warpAffine(image, matrix, (n_width, n_height))
cv.imshow("rotated", rotated)
cv.waitKey(0)
cv.destroyAllWindows()

src_points = np.float32([
 [0.2 * n_width, 0.2 * n_height],
 [0.8 * n_width, 0.2 * n_height],
 [0.2 * n_width, 0.8 * n_height]
])

dst_points = np.float32([
    [0.1 * n_width, 0.3 * n_height],
    [0.9 * n_width, 0.1 * n_height],
    [0.3 * n_width, 0.9 * n_height]
])

matrix = cv.getAffineTransform(src_points, dst_points)
affine = cv.warpAffine(image, matrix, (n_width, n_height))

cv.imshow("affine", affine)
cv.waitKey(0)
cv.destroyAllWindows()



plane = cv.imread("../lab3/plane.png")
if plane is None:
    raise ValueError("Не удалось открыть plane.png")

plt.imshow(cv.cvtColor(plane, cv.COLOR_BGR2RGB))
plt.title("Выберите четыре угла сетки по порядку")
points = plt.ginput(4, timeout=0)
plt.close()

src_points = np.float32(points)
dst_points = np.float32([
    [0,0],
    [500,0],
    [500,500],
    [0,500]
])
homography = cv.getPerspectiveTransform(src_points, dst_points)
rectified = cv.warpPerspective(plane, homography, (500, 500))

cv.imshow("affine", rectified)
cv.waitKey(0)
cv.destroyAllWindows()