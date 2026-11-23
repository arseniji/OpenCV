import cv2 as cv
import matplotlib.pyplot as plt
image = cv.imread("images/rapper.jpg")
if image is None:
    raise ValueError("Не удалось открыть 1.png")
# cv.imshow("govno", image)
# cv.waitKey(0)
# cv.destroyAllWindows()

print(image.shape)
print(image.dtype)

'''запись изображения'''

# cv.imwrite("kaki.png",image)
# cv.imwrite("kaki_2.jpg",image,[cv.IMWRITE_JPEG_QUALITY, 1])


'''в графике'''


# rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# plt.imshow(rgb)
# plt.show()
#
# plt.imshow(image)
# plt.show()


'''маски и вырезанный светофор'''

x, y, width, height = cv.selectROI(image)
cv.destroyAllWindows()
if width == 0 or height == 0:
    raise ValueError("Выделите область со светофором")
roi = image[y:y + height, x:x + width]

hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
value = hsv[:, :, 2]
#
green = cv.inRange(hsv, (40, 80, 0), (90, 255, 255))
yellow = cv.inRange(hsv, (15, 80, 0), (35, 255, 255))
red1 = cv.inRange(hsv, (0, 80, 0), (10, 255, 255))
red2 = cv.inRange(hsv, (170, 80, 0), (179, 255, 255))
red  = cv.bitwise_or(red1, red2)
#
# cv.imshow("ROI",roi)
# cv.imshow("value",value)
# cv.imshow("green",green)
# cv.imshow("yellow",yellow)
# cv.imshow("Red",red)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

'''
На полной картинке
'''

# hsv_full = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#
# green = cv.inRange(hsv_full, (40, 80, 0), (90, 255, 255))
# yellow = cv.inRange(hsv_full, (15, 80, 0), (35, 255, 255))
# red1 = cv.inRange(hsv_full, (0, 80, 0), (10, 255, 255))
# red2 = cv.inRange(hsv_full, (170, 80, 0), (179, 255, 255))
# red  = cv.bitwise_or(red1, red2)
#
# cv.imshow("green",green)
# cv.imshow("yellow",yellow)
# cv.imshow("Red",red)
# cv.waitKey(0)
# cv.destroyAllWindows()


'''гистограмки'''

# hsv_full = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# value_full = hsv_full[:, :, 2]
#
# hist_roi  = cv.calcHist([value],[0], None, [256], [0, 256])
# hist_full = cv.calcHist([value_full], [0], None, [256], [0, 256])
#
# N_roi  = value.size
# N_full = value_full.size
#
# g_roi  = hist_roi  / N_roi
# g_full = hist_full / N_full
#
# plt.plot(g_roi,  label="светофор", linewidth=1.5)
# plt.plot(g_full, label="всё", linewidth=1.0, alpha=0.7)
# plt.xlabel("Значение V")
# plt.ylabel("Доля пикселей")
# plt.legend()
# plt.show()


'''
маски вручную и отсу
'''



_, manual = cv.threshold(value, 127, 255, cv.THRESH_BINARY)

otsu_t, otsu = cv.threshold(value, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)




green_manual = cv.bitwise_and(green, manual)
green_otsu = cv.bitwise_and(green, otsu)

yellow_manual = cv.bitwise_and(yellow, manual)
yellow_otsu = cv.bitwise_and(yellow, otsu)

red_manual = cv.bitwise_and(red, manual)
red_otsu = cv.bitwise_and(red, otsu)

cv.imshow("manual127",manual)
cv.imshow("green_manual",green_manual)
cv.imshow("yellow_manual",yellow_manual)
cv.imshow("red_manual",red_manual)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("otsu",otsu)
cv.imshow("green_otsu",green_otsu)
cv.imshow("yellow_otsu",yellow_otsu)
cv.imshow("red_otsu",red_otsu)
cv.waitKey(0)
cv.destroyAllWindows()