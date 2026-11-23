import os
import sys
from collections import deque

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from ransac_line import fit_line_ransac

SHOW = "--no-show" not in sys.argv
OUT = "results"
os.makedirs(OUT, exist_ok=True)

TARGET_W = 960



SCENES = [
    dict(key="s1", file="photos/scene1_switch.jpg", roi=(300, 300, 360, 160),
         title="Сцена 1 – стрелочный съезд"),
    dict(key="s2", file="photos/scene2_tram_occlusion.jpg", roi=(200, 300, 480, 240),
         title="Сцена 2 – перекрытие рельсов трамваем"),
    dict(key="s3", file="photos/scene3_crossing.jpg", roi=(330, 330, 480, 210),
         title="Сцена 3 – переезд через проезжую часть"),
    dict(key="s4", file="photos/scene4_curve_fog.jpg", roi=(480, 380, 420, 160),
         title="Сцена 4 – поворот (ответвление) пути в тумане"),
]
MAIN = 3



PRIVACY = {
    "s2": [(320, 210, 70, 50), (0, 285, 135, 45)],
    "s3": [(115, 285, 60, 85), (245, 280, 100, 90), (845, 300, 55, 95),
           (935, 325, 25, 70), (160, 310, 80, 40)],
}



def load(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    h, w = img.shape[:2]
    if w != TARGET_W:
        scale = TARGET_W / w
        img = cv.resize(img, (TARGET_W, round(h * scale)), interpolation=cv.INTER_AREA)
    return img


def for_display(img, key):
    out = img.copy()
    for x, y, w, h in PRIVACY.get(key, []):
        out[y:y + h, x:x + w] = cv.GaussianBlur(out[y:y + h, x:x + w], (31, 31), 12)
    return out


def crop(img, roi):
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def to_u8(a):
    m = a.max()
    return np.zeros(a.shape, np.uint8) if m == 0 else np.rint(a / m * 255).astype(np.uint8)


def show(ax, img, title, gray=False):
    if gray:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)



def gradients(smooth):
    gx = cv.Sobel(smooth, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(smooth, cv.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    return gx, gy, mag, ang


def non_max_suppression(mag, ang):
    h, w = mag.shape
    out = np.zeros_like(mag)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = ang[y, x]
            if a < 22.5 or a >= 157.5:
                n1, n2 = mag[y, x - 1], mag[y, x + 1]
            elif a < 67.5:
                n1, n2 = mag[y - 1, x - 1], mag[y + 1, x + 1]
            elif a < 112.5:
                n1, n2 = mag[y - 1, x], mag[y + 1, x]
            else:
                n1, n2 = mag[y - 1, x + 1], mag[y + 1, x - 1]
            if mag[y, x] >= n1 and mag[y, x] >= n2:
                out[y, x] = mag[y, x]
    return out


def hysteresis(nms, low_ratio=0.1, high_ratio=0.2):
    m = nms.max()
    edges = np.zeros(nms.shape, np.uint8)
    if m == 0:
        return edges, 0.0, 0.0
    low, high = low_ratio * m, high_ratio * m
    h, w = nms.shape
    visited = np.zeros(nms.shape, bool)
    queue = deque()
    for y, x in zip(*np.nonzero(nms >= high)):
        visited[y, x] = True
        edges[y, x] = 255
        queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if nms[ny, nx] >= low:
                        visited[ny, nx] = True
                        edges[ny, nx] = 255
                        queue.append((ny, nx))
    return edges, low, high


def part1(scene):
    img = load(scene["file"])
    gray = cv.cvtColor(crop(img, scene["roi"]), cv.COLOR_BGR2GRAY)
    smooth = cv.GaussianBlur(gray, (5, 5), sigmaX=1, sigmaY=1)
    gx, gy, mag, ang = gradients(smooth)
    nms = non_max_suppression(mag, ang)
    own, low, high = hysteresis(nms, 0.1, 0.2)
    cv_a = cv.Canny(smooth, 60, 120, apertureSize=3, L2gradient=True)
    cv_b = cv.Canny(smooth, 30, 60, apertureSize=3, L2gradient=True)
    cv_c = cv.Canny(smooth, 100, 200, apertureSize=3, L2gradient=True)

    print("\nЧАСТЬ 1. Собственный Канни,", scene["title"])
    print(f"  тип Gx: {gx.dtype}, диапазон Gx: [{gx.min():.1f}, {gx.max():.1f}]")
    print(f"  max |grad| = {mag.max():.1f}, max после NMS = {nms.max():.1f}")
    print(f"  пороги: low = {low:.1f}, high = {high:.1f}")
    print(f"  доля ненулевых после NMS: {np.mean(nms > 0) * 100:.1f}%")
    stats = {}
    for name, e in [("свой (0.1/0.2 от max)", own), ("cv.Canny 60/120", cv_a),
                    ("cv.Canny 30/60", cv_b), ("cv.Canny 100/200", cv_c)]:
        stats[name] = int(np.count_nonzero(e))
        print(f"  {name:<24}: {stats[name]:6d} пикселей границ")
    both = np.count_nonzero((own > 0) & (cv_a > 0))
    union = np.count_nonzero((own > 0) | (cv_a > 0))
    print(f"  совпадение своего с 60/120: {both} общих из {union} (IoU {both / union:.2f})")
    print(f"  значения итоговой карты: {np.unique(own)}")

    fig, ax = plt.subplots(3, 2, figsize=(11, 7.6))
    ax = ax.ravel()
    show(ax[0], gray, "Полутоновый фрагмент", True)
    show(ax[1], smooth, "Гаусс 5×5, σ = 1", True)
    show(ax[2], to_u8(mag), "Величина градиента (норм. для показа)", True)
    show(ax[3], to_u8(nms), "После подавления немаксимумов", True)
    show(ax[4], own, f"Свой Канни: пороги 0,1 и 0,2 от max ({low:.0f}/{high:.0f})", True)
    show(ax[5], cv_a, "cv.Canny 60/120, L2gradient", True)
    finish(fig, "p1_own_canny.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 2.6))
    show(ax[0], cv_b, "cv.Canny 30/60", True)
    show(ax[1], cv_a, "cv.Canny 60/120", True)
    show(ax[2], cv_c, "cv.Canny 100/200", True)
    finish(fig, "p1_canny_thresholds.png")


    fig, ax = plt.subplots(figsize=(6, 2.6))
    im = ax.imshow(np.where(mag > high, ang, np.nan), cmap="twilight", vmin=0, vmax=180)
    ax.set_title("Направление градиента, ° (только сильные точки)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03)
    finish(fig, "p1_direction.png")
    return dict(low=low, high=high, max_mag=float(mag.max()), max_nms=float(nms.max()),
                counts=stats, iou=both / union)



def part2(scene):
    img = load(scene["file"])
    x0, y0, w, h = scene["roi"]
    frag = crop(img, scene["roi"]).copy()
    gray = cv.GaussianBlur(cv.cvtColor(frag, cv.COLOR_BGR2GRAY), (5, 5), 1)

    T, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) > 0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]

    print("\nЧАСТЬ 2. Контуры,", scene["title"])
    print(f"  порог Оцу: {T:.0f}, всего внешних контуров: "
          f"{len(cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0])}")
    rows = []
    vis = frag.copy()
    for i, c in enumerate(contours, 1):
        area = cv.contourArea(c)
        per = cv.arcLength(c, True)
        m = cv.moments(c)
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] != 0 else (np.nan, np.nan)
        bx, by, bw, bh = cv.boundingRect(c)
        (rcx, rcy), (rw, rh), rang = cv.minAreaRect(c)
        filled = np.zeros_like(mask)
        cv.drawContours(filled, [c], -1, 255, -1)
        pix = int(np.count_nonzero(filled & mask))
        rows.append(dict(n=i, area=area, per=per, cx=cx, cy=cy, br=(bx, by, bw, bh),
                         mr=(rw, rh, rang), pix=pix, pts=len(c)))
        print(f"  #{i}: площадь {area:.1f} px², пикселей маски {pix}, периметр {per:.1f}, "
              f"центр ({cx:.1f}, {cy:.1f}), boundingRect {bw}x{bh}, "
              f"minAreaRect {rw:.1f}x{rh:.1f} @ {rang:.1f}°, точек {len(c)}")
        color = [(0, 0, 255), (0, 200, 0), (255, 0, 0)][i - 1]
        cv.drawContours(vis, [c], -1, color, 2)
        cv.rectangle(vis, (bx, by), (bx + bw - 1, by + bh - 1), (0, 255, 255), 1)
        box = np.rint(cv.boxPoints(((rcx, rcy), (rw, rh), rang))).astype(np.int32)
        cv.drawContours(vis, [box], -1, (255, 0, 255), 1)
        if m["m00"] != 0:
            cv.circle(vis, (int(round(cx)), int(round(cy))), 4, (255, 255, 255), -1)
            cv.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 0), 1)
        cv.putText(vis, str(i), (bx + 3, by + 15), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                   (255, 255, 255), 2)

    fig, ax = plt.subplots(3, 1, figsize=(8, 9.5))
    show(ax[0], rgb(frag), "Фрагмент")
    show(ax[1], mask, f"Маска Оцу (T = {T:.0f}), белое – светлые области", True)
    show(ax[2], rgb(vis), "Контуры 1–3: жёлтый – boundingRect, "
                          "пурпурный – minAreaRect, точка – центр")
    finish(fig, "p2_contours.png")
    return dict(T=T, rows=rows)



def segment_angle(l):
    x1, y1, x2, y2 = l
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))


def seg_len(l):
    x1, y1, x2, y2 = l
    return float(np.hypot(x2 - x1, y2 - y1))


def select_rails(lines, h):
    keep = []
    for l in lines:
        a = segment_angle(l)
        lower = max(l[1], l[3])
        if 20 <= a <= 80 and lower >= h / 3:
            keep.append(l)
    return keep


def draw_lines(img, lines, color, thick=2):
    out = img.copy()
    for x1, y1, x2, y2 in lines:
        cv.line(out, (x1, y1), (x2, y2), color, thick, cv.LINE_AA)
    return out


def ransac_segment(model, pts):

    a, b, c = model
    d = np.array([-b, a])
    p0 = -c * np.array([a, b])
    t = (pts - p0) @ d
    p1, p2 = p0 + t.min() * d, p0 + t.max() * d
    return p1, p2


def part34(scene):
    img = load(scene["file"])
    disp = for_display(img, scene["key"])
    x0, y0, w, h = scene["roi"]
    frag = crop(img, scene["roi"])
    gray = cv.cvtColor(frag, cv.COLOR_BGR2GRAY)
    smooth = cv.GaussianBlur(gray, (5, 5), sigmaX=1, sigmaY=1)
    edges = cv.Canny(smooth, 60, 120, apertureSize=3, L2gradient=True)

    raw = cv.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=80, maxLineGap=20)
    lines = [] if raw is None else [tuple(int(v) for v in l) for l in raw.reshape(-1, 4)]
    chosen = select_rails(lines, h)
    longest2 = sorted(lines, key=seg_len, reverse=True)[:2]

    ys, xs = np.nonzero(edges)
    points = np.column_stack((xs, ys))
    model, inliers = fit_line_ransac(points, tolerance=3, attempts=100)

    print(f"\nЧАСТИ 3–4. {scene['title']}, ROI {scene['roi']}")
    print(f"  точек Канни: {len(points)}, отрезков Хафа: {len(lines)}, "
          f"по правилу: {len(chosen)}")
    print("  два самых длинных: " + "; ".join(
        f"{l} длина {seg_len(l):.0f}, угол {segment_angle(l):.0f}°" for l in longest2))

    res = dict(n_points=len(points), n_lines=len(lines), n_chosen=len(chosen),
               longest=[(l, seg_len(l), segment_angle(l)) for l in longest2],
               lines=lines, chosen=chosen)
    canvas_r = rgb(frag).copy()
    if model is None:
        print("  RANSAC: модель не найдена (None)")
        res["ransac"] = None
    else:
        a, b, c = model
        p1, p2 = ransac_segment(model, points[inliers])
        ang = np.degrees(np.arctan2(abs(p2[1] - p1[1]), abs(p2[0] - p1[0])))
        L = float(np.hypot(*(p2 - p1)))
        share = inliers.mean()
        print(f"  RANSAC: a={a:.4f}, b={b:.4f}, c={c:.2f}; поддержка {inliers.sum()} "
              f"из {len(points)} ({share * 100:.1f}%), отрезок "
              f"({p1[0]:.0f},{p1[1]:.0f})–({p2[0]:.0f},{p2[1]:.0f}), длина {L:.0f}, "
              f"угол {ang:.0f}°")
        res["ransac"] = dict(a=a, b=b, c=c, support=int(inliers.sum()), share=share,
                             p1=p1, p2=p2, length=L, angle=ang)
        canvas_r[ys[~inliers], xs[~inliers]] = (90, 90, 90)
        canvas_r[ys[inliers], xs[inliers]] = (0, 255, 0)
        cv.line(canvas_r, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)),
                (255, 0, 0), 2, cv.LINE_AA)


    framed = disp.copy()
    cv.rectangle(framed, (x0, y0), (x0 + w - 1, y0 + h - 1), (0, 255, 255), 3)
    frag_d = crop(disp, scene["roi"])
    all_l = draw_lines(rgb(frag_d), lines, (255, 0, 0))
    sel_l = draw_lines(rgb(frag_d), chosen, (0, 255, 0), 3)
    sel_l = draw_lines(sel_l, longest2, (255, 0, 255), 1)

    fig = plt.figure(figsize=(12, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    show(ax, rgb(framed), f"{scene['title']}; область x={x0}, y={y0}, w={w}, h={h}")
    show(fig.add_subplot(gs[1, 0]), rgb(frag_d), "Фрагмент")
    show(fig.add_subplot(gs[1, 1]), edges, "cv.Canny 60/120 (Гаусс 5×5, σ=1)", True)
    show(fig.add_subplot(gs[2, 0]), all_l, f"Хаф: все отрезки ({len(lines)})")
    ax = fig.add_subplot(gs[2, 1])
    show(ax, sel_l, f"Зелёные – по правилу ({len(chosen)}), пурпурные – 2 самых длинных")
    finish(fig, f"p3_{scene['key']}_hough.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 12 * h / w / 2 + 0.6))
    show(ax[0], edges, "Точки Канни", True)
    title = ("RANSAC: модели нет (None)" if model is None else
             f"RANSAC: поддержка {res['ransac']['support']} точек (зелёные), отрезок – красный")
    show(ax[1], canvas_r, title)
    finish(fig, f"p4_{scene['key']}_ransac.png")
    return res



CIRCLE_CASES = [
    dict(key="c1", scene=1, box=(280, 285, 170, 60), r=(5, 12),
         title="Фары трамвая (сцена 2)"),
    dict(key="c2", scene=2, box=(680, 280, 60, 60), r=(12, 22),
         title="Круглый знак (сцена 3)"),
    dict(key="c3", scene=3, box=(705, 170, 75, 110), r=(6, 15),
         title="Светофор (сцена 4)"),
]


def circles():
    print("\nЧАСТЬ 3. Окружности Хафа")
    res = []
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for k, cs in enumerate(CIRCLE_CASES):
        img = load(SCENES[cs["scene"]]["file"])
        frag = crop(img, cs["box"]).copy()
        g = cv.medianBlur(cv.cvtColor(frag, cv.COLOR_BGR2GRAY), 5)
        rmin, rmax = cs["r"]
        found = cv.HoughCircles(g, cv.HOUGH_GRADIENT, dp=1, minDist=2 * rmin,
                                param1=100, param2=12, minRadius=rmin, maxRadius=rmax)
        out = rgb(frag).copy()
        lst = [] if found is None else [tuple(float(v) for v in c) for c in found[0]]
        for cx, cy, r in lst:
            cv.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 1)
            cv.circle(out, (int(round(cx)), int(round(cy))), 1, (255, 0, 0), -1)
        print(f"  {cs['title']}: диапазон радиусов {rmin}–{rmax}, найдено {len(lst)}: "
              + ", ".join(f"центр ({cx:.0f},{cy:.0f}) r={r:.1f}" for cx, cy, r in lst))
        res.append(dict(title=cs["title"], r=cs["r"], found=lst))
        show(ax[0, k], rgb(frag), cs["title"])
        show(ax[1, k], out, f"HoughCircles, r = {rmin}…{rmax}: найдено {len(lst)}")
    finish(fig, "p3_circles.png")
    return res



if __name__ == "__main__":
    for s in SCENES:
        img = load(s["file"])
        print(f"{s['file']}: {img.shape[1]}x{img.shape[0]} после приведения к ширине 960")
    r1 = part1(SCENES[MAIN])
    r2 = part2(SCENES[MAIN])
    r3 = [part34(s) for s in SCENES]
    rc = circles()
