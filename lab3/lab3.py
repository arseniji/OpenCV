
import os
import sys
import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

SHOW = "--no-show" not in sys.argv
OUT = "results"
os.makedirs(OUT, exist_ok=True)



SEED = 42


def show_gray(ax, img, title):
    
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)


def to_uint8(img):
    
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)



gray = cv.imread("image.png", cv.IMREAD_GRAYSCALE)
if gray is None:
    raise ValueError("Не удалось открыть image.png")
original = gray.copy()          
height, width = gray.shape
print(f"1. Размер: {width}x{height} (ширина x высота), тип {gray.dtype}")

fig, ax = plt.subplots(figsize=(5, 5))
show_gray(ax, gray, "Исходная фотография (OpenCV samples)")
finish(fig, "01_original.png")



def gaussian_kernel(k, sigma):
    
    if k % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    r = (k - 1) // 2
    kernel = np.zeros((k, k), dtype=np.float64)
    for v in range(-r, r + 1):          
        for u in range(-r, r + 1):      
            kernel[v + r, u + r] = np.exp(-(u * u + v * v) / (2 * sigma * sigma))
    return kernel / kernel.sum()        


print("\n2. Ядра Гаусса, sigma = 1")
np.set_printoptions(precision=6, suppress=True, linewidth=120)
for k in (3, 5, 7):
    K = gaussian_kernel(k, 1.0)
    symmetric = (np.allclose(K, K.T) and np.allclose(K, K[::-1, :])
                 and np.allclose(K, K[:, ::-1]))
    print(f"\nЯдро {k}x{k}:\n{K}")
    print(f"Сумма элементов = {K.sum():.17f}")
    print(f"Симметрия относительно центра: {symmetric}")



def my_gaussian_filter(image, k, sigma):
    
    K = gaussian_kernel(k, sigma)
    r = (k - 1) // 2
    source = image.astype(np.float64)
    result = source.copy()
    h, w = source.shape
    for y in range(r, h - r):           
        for x in range(r, w - r):       
            window = source[y - r:y + r + 1, x - r:x + r + 1]
            result[y, x] = np.sum(window * K)
    return result


combos = [(3, 1), (3, 2), (7, 1), (7, 2)]
own = {}
print("\n3. Собственная фильтрация")
for k, s in combos:
    t = time.perf_counter()
    own[(k, s)] = my_gaussian_filter(gray, k, s)
    print(f"  k={k}, sigma={s}: {time.perf_counter() - t:.2f} c")
    cv.imwrite(os.path.join(OUT, f"own_gauss_k{k}_s{s}.png"), to_uint8(own[(k, s)]))

fig, axes = plt.subplots(2, 2, figsize=(10, 10.4))
for ax, (k, s) in zip(axes.ravel(), combos):
    show_gray(ax, to_uint8(own[(k, s)]), f"Свой Гаусс {k}×{k}, σ={s}")
finish(fig, "03_own_gauss_full.png")


fy, fx = slice(60, 210), slice(60, 260)
fig, axes = plt.subplots(3, 2, figsize=(10, 11.5))
show_gray(axes[0, 0], gray[fy, fx], "Исходник (фрагмент)")
axes[0, 1].axis("off")
for ax, (k, s) in zip(axes[1:].ravel(), combos):
    show_gray(ax, to_uint8(own[(k, s)])[fy, fx], f"Свой Гаусс {k}×{k}, σ={s}")
finish(fig, "03_own_gauss_crop.png")



print("\n4. Сравнение с cv.GaussianBlur (только внутренняя область)")
for k, s in combos:
    reference = cv.GaussianBlur(gray.astype(np.float64), (k, k), s, sigmaY=s)
    r = (k - 1) // 2
    inner = (slice(r, height - r), slice(r, width - r))
    diff = np.abs(own[(k, s)] - reference)            
    dmax = np.max(diff[inner])
    edge = np.ones(diff.shape, bool)
    edge[inner] = False
    print(f"  k={k}, sigma={s}: max |own - OpenCV| внутри = {dmax:.3e}, "
          f"на краевой полосе = {diff[edge].max():.1f}, "
          f"пикселей с разницей > 1e-9: {np.count_nonzero(diff > 1e-9)} "
          f"(краевая полоса: {np.count_nonzero(edge)})")
    if (k, s) == (7, 2):
        diff_72, inner_72 = diff, inner


fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
im = ax[0].imshow(diff_72[:30, :30], cmap="gray", interpolation="nearest")
ax[0].set_title("|свой − OpenCV|, левый верхний угол 30×30 px\n"
                "светлая полоса 3 px – необработанный край", fontsize=10)
fig.colorbar(im, ax=ax[0], fraction=0.046)
im = ax[1].imshow(diff_72[inner_72] * 1e13, cmap="gray")
ax[1].set_title("Внутренняя область, разница × 10¹³\n(погрешность округления float64)", fontsize=10)
fig.colorbar(im, ax=ax[1], fraction=0.046)
for a in ax:
    a.axis("off")
finish(fig, "04_diff_opencv.png")



rng = np.random.default_rng(SEED)

noise = rng.normal(0, 20, gray.shape)
gaussian_noisy = np.clip(np.rint(gray.astype(np.float64) + noise), 0, 255).astype(np.uint8)

p = 0.04
u = rng.random(gray.shape)
impulse_noisy = gray.copy()
impulse_noisy[u < p / 2] = 0
impulse_noisy[u >= 1 - p / 2] = 255
print(f"\n5. Импульсный шум: выбрано {np.mean((u < p/2) | (u >= 1 - p/2))*100:.2f}% пикселей")

cv.imwrite(os.path.join(OUT, "noisy_gaussian_sigma20.png"), gaussian_noisy)
cv.imwrite(os.path.join(OUT, "noisy_impulse_p4.png"), impulse_noisy)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
show_gray(axes[0], gray, "Исходник")
show_gray(axes[1], gaussian_noisy, "Гауссов шум, σ = 20")
show_gray(axes[2], impulse_noisy, "Импульсный шум, p = 4%")
finish(fig, "05_noise.png")



def mse(reference, img):
    diff = reference.astype(np.float64) - img.astype(np.float64)
    return np.mean(diff ** 2)


noisy_set = {"Гауссов шум": gaussian_noisy, "Импульсный шум": impulse_noisy}
filtered_set = {}
print("\n6. MSE относительно исходника")
print(f"  {'Изображение':<16}{'шумная копия':>14}{'среднее 5x5':>14}"
      f"{'Гаусс 5x5':>14}{'медиана 5':>14}")
for name, noisy in noisy_set.items():
    mean_result = cv.blur(noisy, (5, 5))
    gaussian_result = cv.GaussianBlur(noisy, (5, 5), 1)
    median_result = cv.medianBlur(noisy, 5)
    filtered_set[name] = (mean_result, gaussian_result, median_result)
    vals = [mse(gray, noisy)] + [mse(gray, x) for x in filtered_set[name]]
    print(f"  {name:<16}" + "".join(f"{v:>14.2f}" for v in vals))


cy, cx = slice(100, 210), slice(60, 230)
titles = ["Шумная копия", "Среднее 5×5", "Гаусс 5×5, σ=1", "Медиана 5×5"]
fig, axes = plt.subplots(3, 4, figsize=(14, 8.4))
show_gray(axes[0, 0], gray[cy, cx], "Исходник (фрагмент)")
for j in range(1, 4):
    axes[0, j].axis("off")
for i, (name, noisy) in enumerate(noisy_set.items(), start=1):
    imgs = [noisy, *filtered_set[name]]
    for j in range(4):
        show_gray(axes[i, j], imgs[j][cy, cx], f"{name}: {titles[j]}")
finish(fig, "06_filters_crop.png")



filtered = cv.medianBlur(impulse_noisy, 5)
_, global_mask = cv.threshold(filtered, 127, 255, cv.THRESH_BINARY_INV)
otsu_T, otsu_mask = cv.threshold(filtered, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
adaptive_mask = cv.adaptiveThreshold(
    filtered, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 5
)
print(f"\n7. Порог Оцу: {otsu_T:.0f}")
for name, m in [("общий 127", global_mask), ("Оцу", otsu_mask), ("адаптивный", adaptive_mask)]:
    print(f"  доля белого в маске ({name}): {np.mean(m == 255) * 100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(10, 10.4))
axes = axes.ravel()
show_gray(axes[0], filtered, "Импульсный шум + медиана 5×5")
show_gray(axes[1], global_mask, "Общий порог T = 127")
show_gray(axes[2], otsu_mask, f"Оцу, T = {otsu_T:.0f}")
show_gray(axes[3], adaptive_mask, "Адаптивный гауссов, 31×31, C = 5")
finish(fig, "07_threshold.png")


fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(filtered.ravel(), bins=256, range=(0, 256), color="black")
ax.axvline(127, color="black", linestyle="--", label="T = 127")
ax.axvline(otsu_T, color="gray", linestyle="-", linewidth=2, label=f"Оцу T = {otsu_T:.0f}")
ax.set_xlabel("Яркость")
ax.set_ylabel("Число пикселей")
ax.legend()
finish(fig, "07_histogram.png")



kernel = np.ones((3, 3), np.uint8)
eroded = cv.erode(adaptive_mask, kernel)
dilated = cv.dilate(adaptive_mask, kernel)
opened = cv.morphologyEx(adaptive_mask, cv.MORPH_OPEN, kernel)
closed = cv.morphologyEx(adaptive_mask, cv.MORPH_CLOSE, kernel)

n_comp = lambda m: cv.connectedComponents(m)[0] - 1   
print("\n8. Морфология (число белых компонент связности / доля белого)")
for name, m in [("исходная", adaptive_mask), ("эрозия", eroded), ("дилатация", dilated),
                ("открытие", opened), ("закрытие", closed)]:
    print(f"  {name:<10}: {n_comp(m):5d} компонент, {np.mean(m == 255) * 100:5.1f}% белого")

mtitles = ["Адаптивная маска", "Эрозия 3×3", "Дилатация 3×3", "Открытие 3×3", "Закрытие 3×3"]
masks = [adaptive_mask, eroded, dilated, opened, closed]

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.ravel()
for ax, m, t in zip(axes, masks, mtitles):
    show_gray(ax, m, t)
axes[5].axis("off")
finish(fig, "08_morph_full.png")


my_, mx_ = slice(250, 345), slice(55, 300)
fig, axes = plt.subplots(3, 2, figsize=(12, 7))
axes = axes.ravel()
axes[5].axis("off")
for ax, m, t in zip(axes, masks, mtitles):
    ax.imshow(m[my_, mx_], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(t + " (фрагмент)", fontsize=10)
    ax.axis("off")
finish(fig, "08_morph_crop.png")






final_mask = closed
cv.imwrite(os.path.join(OUT, "final_mask_closing_3x3.png"), final_mask)
print("\nОкончательная маска: закрытие 3x3 -> results/final_mask_closing_3x3.png")