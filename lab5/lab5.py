
import csv
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

SHOW = "--no-show" not in sys.argv
VIDEO = sys.argv[sys.argv.index("--video") + 1] if "--video" in sys.argv else None
DATA = Path("data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

SEQUENCES = ["i_ajuntament", "i_bologna", "i_fenis", "v_adam", "v_bark", "v_boat"]
METHODS = ["ORB", "SIFT"]
N_FEATURES = 2000
RATIO = 0.75
RANSAC_THR = 3.0
CORRECT_THR = 3.0
GOOD_E = 5.0
MAX_SIDE = 960

cv.setRNGSeed(0)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)


def read_gray(path):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать {path}")
    return img



def template_search(seq, second_no, x, y, width, height, name):
    first = read_gray(DATA / seq / "1.ppm")
    second = read_gray(DATA / seq / f"{second_no}.ppm")
    template = first[y:y + height, x:x + width]
    response = cv.matchTemplate(second, template, cv.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv.minMaxLoc(response)
    bottom_right = (top_left[0] + width, top_left[1] + height)


    H = np.loadtxt(DATA / seq / f"H_1_{second_no}")
    corners = np.float32([[x, y], [x + width, y], [x + width, y + height],
                          [x, y + height]]).reshape(-1, 1, 2)
    true_quad = cv.perspectiveTransform(corners, H).reshape(-1, 2)
    true_center = true_quad.mean(axis=0)
    found_center = np.array([top_left[0] + width / 2, top_left[1] + height / 2])
    center_err = float(np.linalg.norm(found_center - true_center))

    cv.imwrite(str(OUT / f"template_{name}.png"), template)
    result = cv.cvtColor(second, cv.COLOR_GRAY2BGR)
    cv.rectangle(result, top_left, bottom_right, (0, 0, 255), 2)
    cv.polylines(result, [np.int32(np.rint(true_quad))], True, (0, 200, 0), 2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4),
                           gridspec_kw={"width_ratios": [0.6, 1, 1]})
    ax[0].imshow(template, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title(f"Шаблон {seq}/1.ppm\nx={x}, y={y}, {width}×{height}", fontsize=10)
    im = ax[1].imshow(response, cmap="gray")
    ax[1].plot(*top_left, "r+", markersize=14, mew=2)
    ax[1].set_title(f"Карта TM_CCOEFF_NORMED, максимум {score:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax[1], fraction=0.035)
    ax[2].imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    ax[2].set_title(f"{seq}/{second_no}.ppm: красный – найдено,\n"
                    f"зелёный – эталонное положение шаблона", fontsize=10)
    for a in ax:
        a.axis("off")
    finish(fig, f"p2_template_{name}.png")
    print(f"  {seq} 1->{second_no}: шаблон {width}x{height} в ({x},{y}); "
          f"карта {response.shape[1]}x{response.shape[0]}; максимум {score:.4f} в {top_left}; "
          f"ошибка центра относительно эталона {center_err:.1f} px")
    return dict(seq=seq, second=second_no, score=score, top_left=top_left,
                center_err=center_err, map_size=response.shape[::-1])



def prepare(img):

    H0, W0 = img.shape[:2]
    if max(H0, W0) > MAX_SIDE:
        s = MAX_SIDE / max(H0, W0)
        work = cv.resize(img, (round(W0 * s), round(H0 * s)), interpolation=cv.INTER_AREA)
    else:
        work = img
    h, w = work.shape[:2]
    return work, W0 / w, H0 / h


def create(method):
    return (cv.ORB_create(nfeatures=N_FEATURES) if method == "ORB"
            else cv.SIFT_create(nfeatures=N_FEATURES))


def detect(detector, image):
    kps, des = detector.detectAndCompute(image, None)
    if des is None or len(kps) == 0:
        return [], None
    if len(kps) > N_FEATURES:
        order = np.argsort([-k.response for k in kps])[:N_FEATURES]
        kps = [kps[i] for i in order]
        des = des[order]
    return list(kps), des


def ratio_matches(des1, des2, method):
    if des1 is None or des2 is None or len(des1) < 1 or len(des2) < 2:
        return []
    norm = cv.NORM_HAMMING if method == "ORB" else cv.NORM_L2
    matcher = cv.BFMatcher(norm)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2:
            first_match, second_match = pair
            if first_match.distance < RATIO * second_match.distance:
                good.append(first_match)
    return good


def corners_of(W, H):
    return np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]).reshape(-1, 1, 2)


def run_pair(seq, j, method):
    img1 = read_gray(DATA / seq / "1.ppm")
    img2 = read_gray(DATA / seq / f"{j}.ppm")
    truth = np.loadtxt(DATA / seq / f"H_1_{j}")
    work1, sx1, sy1 = prepare(img1)
    work2, sx2, sy2 = prepare(img2)

    det = create(method)
    kp1, des1 = detect(det, work1)
    kp2, des2 = detect(det, work2)
    good = ratio_matches(des1, des2, method)
    N = len(good)


    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    src[:, 0] *= sx1; src[:, 1] *= sy1
    dst[:, 0] *= sx2; dst[:, 1] *= sy2


    if N > 0:
        projected = cv.perspectiveTransform(src.reshape(-1, 1, 2), truth).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst, axis=1)
        correct = errors <= CORRECT_THR
    else:
        errors = np.zeros(0)
        correct = np.zeros(0, bool)
    C = int(correct.sum())
    P = C / N if N > 0 else 0.0


    estimated, inlier_mask, inliers, E = None, None, 0, None
    if N >= 4:
        estimated, inlier_mask = cv.findHomography(src, dst, cv.RANSAC, RANSAC_THR)
        if estimated is None or inlier_mask is None:
            estimated, inlier_mask = None, None
        else:
            inliers = int(inlier_mask.ravel().sum())

    H0, W0 = img1.shape[:2]
    corners = corners_of(W0, H0)
    true_quad = cv.perspectiveTransform(corners, truth).reshape(-1, 2)
    est_quad = None
    if estimated is not None:
        est_quad = cv.perspectiveTransform(corners, estimated).reshape(-1, 2)
        if np.all(np.isfinite(est_quad)):
            E = float(np.mean(np.linalg.norm(est_quad - true_quad, axis=1)))
        else:
            est_quad = None
    return dict(seq=seq, group=seq[0], j=j, method=method, kp1=len(kp1), kp2=len(kp2),
                N=N, C=C, P=P, inliers=inliers, E=E, failed=E is None,

                img1=img1, img2=img2, src=src, dst=dst, correct=correct, errors=errors,
                kp1_pts=np.float32([k.pt for k in kp1]).reshape(-1, 2) * [sx1, sy1],
                kp2_pts=np.float32([k.pt for k in kp2]).reshape(-1, 2) * [sx2, sy2],
                true_quad=true_quad, est_quad=est_quad, inlier_mask=inlier_mask,
                work_size=work1.shape[::-1], orig_size=img1.shape[::-1])


def draw_pair(r, title, name, max_lines=150):
    img1, img2 = r["img1"], r["img2"]
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    canvas[:h1, :w1] = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    canvas[:h2, w1:] = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    canvas = (canvas * 0.75).astype(np.uint8)
    for x, y in r["kp1_pts"]:
        cv.circle(canvas, (int(x), int(y)), 1, (255, 200, 0), -1)
    for x, y in r["kp2_pts"]:
        cv.circle(canvas, (int(x) + w1, int(y)), 1, (255, 200, 0), -1)
    rng = np.random.default_rng(0)
    idx = np.arange(r["N"])
    if len(idx) > max_lines:
        idx = rng.choice(idx, max_lines, replace=False)
    for i in idx:
        p = tuple(np.int32(r["src"][i]))
        q = (int(r["dst"][i][0]) + w1, int(r["dst"][i][1]))
        color = (0, 200, 0) if r["correct"][i] else (0, 0, 255)
        cv.line(canvas, p, q, color, 1, cv.LINE_AA)
    shift = np.array([w1, 0])
    cv.polylines(canvas, [np.int32(np.rint(r["true_quad"] + shift))], True, (0, 255, 0), 3)
    if r["est_quad"] is not None:
        cv.polylines(canvas, [np.int32(np.rint(r["est_quad"] + shift))], True, (255, 0, 255), 2)
    fig, ax = plt.subplots(figsize=(13, 13 * canvas.shape[0] / canvas.shape[1] + 0.8))
    ax.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    E = "срыв" if r["failed"] else f"{r['E']:.2f} px"
    ax.set_title(f"{title}: {r['seq']} 1→{r['j']}, {r['method']}. N={r['N']}, C={r['C']}, "
                 f"P={r['P']:.3f}, RANSAC={r['inliers']}, E={E}\n"
                 f"Линии (до {max_lines}): зелёные – правильные, красные – ошибочные. "
                 f"Зелёный 4-угольник – эталон, пурпурный – найденный", fontsize=10)
    ax.axis("off")
    finish(fig, name)


def fmt(v, nd=1):
    return "срыв" if v is None else f"{v:.{nd}f}"



def make_video(path, n=60, size=(480, 360)):

    w, h = size
    rng = np.random.default_rng(1)
    bg = cv.GaussianBlur(rng.integers(60, 200, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(40, 220, 3))
        cv.circle(bg, tuple(int(v) for v in rng.integers(0, [w, h])),
                  int(rng.integers(5, 25)), c, -1)
    obj = np.zeros((70, 90, 3), np.uint8)
    obj[:] = (30, 30, 200)
    for _ in range(25):
        cv.circle(obj, tuple(int(v) for v in rng.integers(5, [85, 65])),
                  int(rng.integers(2, 6)), (20, 20, int(rng.integers(90, 255))), -1)
    cv.rectangle(obj, (0, 0), (89, 69), (10, 10, 120), 2)
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    truth = []
    for t in range(n):
        x = int(30 + 8 * t)
        y = int(140 + 50 * np.sin(t / 9))
        frame = bg.copy()
        x2, y2 = min(x + 90, w), y + 70
        if x < w:
            frame[y:y2, x:x2] = obj[:, :x2 - x]
        frame = cv.add(frame, rng.integers(0, 6, frame.shape, dtype=np.uint8))
        writer.write(frame)
        truth.append((x, y))
    writer.release()
    return truth


def red_mask(hsv):
    mask = cv.bitwise_or(cv.inRange(hsv, (0, 80, 50), (10, 255, 255)),
                         cv.inRange(hsv, (170, 80, 50), (179, 255, 255)))
    k = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    return cv.morphologyEx(mask, cv.MORPH_CLOSE, k)


def track(path, points, box, keep, truth):

    cap = cv.VideoCapture(str(path))
    ok, frame = cap.read()
    prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    box = np.array(box, float)
    frames = [(0, frame.copy(), points.reshape(-1, 2).copy(), box.copy())] if 0 in keep else []
    log, lost_at, t = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t += 1
        cur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if len(points) > 0:
            nxt, status, _ = cv.calcOpticalFlowPyrLK(prev, cur, points, None)
            valid = status.ravel() == 1
            old, new = points[valid].reshape(-1, 2), nxt[valid].reshape(-1, 2)
            inside = ((new[:, 0] >= 0) & (new[:, 0] < cur.shape[1] - 1) &
                      (new[:, 1] >= 0) & (new[:, 1] < cur.shape[0] - 1))
            old, new = old[inside], new[inside]
        else:
            old = new = np.zeros((0, 2), np.float32)
        d = np.median(new - old, axis=0) if len(new) else np.array([np.nan, np.nan])
        if len(new):
            box[:2] += d
        err = None
        if truth is not None:
            tx, ty = truth[t]
            err = float(np.hypot(box[0] + box[2] / 2 - (tx + 45), box[1] + box[3] / 2 - (ty + 35)))
        log.append((t, len(new), d, err))
        if lost_at is None and len(new) < 5:
            lost_at = t
        if t in keep:
            frames.append((t, frame.copy(), new.copy(), box.copy()))
        points = new.reshape(-1, 1, 2).astype(np.float32)
        prev = cur
    cap.release()
    return dict(log=log, frames=frames, lost_at=lost_at)


def optical_flow():
    path, truth = VIDEO, None
    if path is None:
        path = OUT / "synthetic_red_object.mp4"
        truth = make_video(path)
        print(f"\nЧАСТЬ 9. Видео не задано – сгенерирован ролик {path} (60 кадров, 480x360)")
    cap = cv.VideoCapture(str(path))
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"Не удалось прочитать видео {path}")
    gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)


    mask = red_mask(cv.cvtColor(first, cv.COLOR_BGR2HSV))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    obj = max(contours, key=cv.contourArea)
    obj_mask = np.zeros_like(mask)
    cv.drawContours(obj_mask, [obj], -1, 255, -1)
    x, y, w, h = cv.boundingRect(obj)

    m = 25
    rx, ry = max(x - m, 0), max(y - m, 0)
    rw, rh = min(x + w + m, gray.shape[1]) - rx, min(y + h + m, gray.shape[0]) - ry

    pts_box = cv.goodFeaturesToTrack(gray, 100, 0.01, 5, mask=_rect_mask(gray.shape, (rx, ry, rw, rh)))
    pts_hsv = cv.goodFeaturesToTrack(gray, 100, 0.01, 5, mask=obj_mask)
    if pts_hsv is None or pts_box is None:
        raise RuntimeError("В области объекта не найдено характерных точек")
    n_bg = _count_outside(pts_box, obj_mask)
    print(f"  объект по HSV: x={x}, y={y}, w={w}, h={h}; контуров красного на кадре: {len(contours)}")
    print(f"  рамка с запасом: x={rx}, y={ry}, w={rw}, h={rh}: точек {len(pts_box)}, "
          f"из них на фоне {n_bg}; точек по маске HSV: {len(pts_hsv)}")

    keep = {0, 15, 30, 45, 52, 56}
    runs = {"рамка": track(path, pts_box, (rx, ry, rw, rh), keep, truth),
            "HSV": track(path, pts_hsv, (x, y, w, h), keep, truth)}
    for name, r in runs.items():
        print(f"  [{name}] потеря точек (<5): "
              f"{'кадр ' + str(r['lost_at']) if r['lost_at'] else 'не возникла'}")
        for t, n, d, err in r["log"]:
            if t % 10 == 0 or t in (52, 54, 56, 59):
                e = "" if err is None else f", ошибка центра {err:.1f} px"
                print(f"    кадр {t:2d}: точек {n:3d}, смещение ({d[0]:+.1f}, {d[1]:+.1f}){e}")

    fig, ax = plt.subplots(2, 3, figsize=(13, 6.8))
    for a, (t, fr, pts, b) in zip(ax.ravel(), runs["HSV"]["frames"]):
        vis = fr.copy()
        for px, py in pts:
            cv.circle(vis, (int(px), int(py)), 2, (0, 255, 0), -1)
        cv.rectangle(vis, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])),
                     (255, 255, 0), 2)
        a.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
        a.set_title(f"Кадр {t}: точек {len(pts)}", fontsize=10)
        a.axis("off")
    finish(fig, "p9_optical_flow.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].imshow(mask, cmap="gray")
    ax[0].set_title(f"Маска красного по HSV: {len(contours)} обл.", fontsize=10)
    for a, pts, bx, ttl in [
            (ax[1], pts_box, (rx, ry, rw, rh), f"Рамка с запасом: {len(pts_box)} точек, {n_bg} на фоне"),
            (ax[2], pts_hsv, (x, y, w, h), f"Только маска объекта: {len(pts_hsv)} точек")]:
        v = first.copy()
        cv.rectangle(v, (bx[0], bx[1]), (bx[0] + bx[2], bx[1] + bx[3]), (255, 255, 0), 2)
        for px, py in pts.reshape(-1, 2):
            inside = obj_mask[int(round(py)), int(round(px))] > 0
            cv.circle(v, (int(px), int(py)), 3, (0, 255, 0) if inside else (0, 0, 255), -1)
        a.imshow(cv.cvtColor(v, cv.COLOR_BGR2RGB))
        a.set_title(ttl, fontsize=10)
    for a in ax:
        a.axis("off")
    finish(fig, "p9_initial_region.png")

    if truth is not None:
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
        for name, ls in [("рамка", "--"), ("HSV", "-")]:
            lg = runs[name]["log"]
            ax[0].plot([t for t, *_ in lg], [e for *_, e in lg], ls, color="black",
                       label=f"точки: {name}")
            ax[1].plot([t for t, *_ in lg], [n for _, n, *_ in lg], ls, color="black",
                       label=f"точки: {name}")
        ax[0].set_title("Ошибка центра рамки, px")
        ax[1].set_title("Число сохранённых точек")
        for a in ax:
            a.set_xlabel("кадр")
            a.grid(alpha=0.3)
            a.legend()
        finish(fig, "p9_error.png")
    return dict(obj=(x, y, w, h), rough=(rx, ry, rw, rh), n_box=len(pts_box), n_bg=n_bg,
                n_hsv=len(pts_hsv), runs=runs, n_red=len(contours))


def _rect_mask(shape, rect):
    m = np.zeros(shape, np.uint8)
    x, y, w, h = rect
    m[y:y + h, x:x + w] = 255
    return m


def _count_outside(pts, mask):
    return int(sum(mask[int(round(py)), int(round(px))] == 0 for px, py in pts.reshape(-1, 2)))



if __name__ == "__main__":
    print("ЧАСТЬ 2. Поиск по шаблону")
    tpl = [template_search("i_ajuntament", 2, 192, 120, 160, 120, "ajuntament_1_2"),
           template_search("v_adam", 6, 180, 112, 150, 112, "adam_1_6")]

    print("\nЧАСТИ 3–8. ORB и SIFT на 30 парах")
    results = []
    for seq in SEQUENCES:
        for j in range(2, 7):
            for method in METHODS:
                r = run_pair(seq, j, method)
                results.append(r)
                print(f"  {seq:<13} 1->{j} {method:<4}: точек {r['kp1']:4d}/{r['kp2']:4d}, "
                      f"N={r['N']:4d}, C={r['C']:4d}, P={r['P']:.3f}, RANSAC={r['inliers']:4d}, "
                      f"E={fmt(r['E'], 2)}")
    sizes = {(r['work_size'], r['orig_size']) for r in results}
    print("  рабочий и исходный размеры совпадают:", all(a == b for a, b in sizes))

    with open(OUT / "results_60.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["папка", "изобр.", "метод", "точек 1", "точек 2", "N", "C", "P",
                     "RANSAC", "E, px"])
        for r in results:
            wr.writerow([r["seq"], r["j"], r["method"], r["kp1"], r["kp2"], r["N"], r["C"],
                         f"{r['P']:.3f}", r["inliers"], fmt(r["E"], 2)])

    print("\nСВОДКА")
    summary = []
    for group, gname in [("i", "освещение"), ("v", "точка съёмки")]:
        for method in METHODS:
            rs = [r for r in results if r["group"] == group and r["method"] == method]
            Es = [r["E"] for r in rs if r["E"] is not None]
            row = dict(group=gname, method=method,
                       meanC=np.mean([r["C"] for r in rs]),
                       meanP=np.mean([r["P"] for r in rs]),
                       medE=float(np.median(Es)) if Es else None, nE=len(Es),
                       fails=sum(r["failed"] for r in rs),
                       good=sum(r["E"] is not None and r["E"] <= GOOD_E for r in rs))
            summary.append(row)
            print(f"  {gname:<13} {method:<4}: среднее C={row['meanC']:.1f}, "
                  f"среднее P={row['meanP']:.3f}, медиана E={fmt(row['medE'], 2)} "
                  f"(по {row['nE']}), срывов {row['fails']}, E<=5: {row['good']} из 15")
    with open(OUT / "summary_4.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["изменение", "метод", "среднее C", "среднее P", "медиана E",
                     "конечных E", "срывов", "E<=5 из 15"])
        for s in summary:
            wr.writerow([s["group"], s["method"], f"{s['meanC']:.1f}", f"{s['meanP']:.3f}",
                         fmt(s["medE"], 2), s["nE"], s["fails"], s["good"]])


    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    labels = [f"{s['group']}\n{s['method']}" for s in summary]
    shades = ["0.75", "0.25", "0.75", "0.25"]
    ax[0].bar(labels, [s["meanC"] for s in summary], color=shades, edgecolor="black")
    ax[0].set_title("Среднее число правильных соответствий C")
    ax[1].bar(labels, [s["meanP"] for s in summary], color=shades, edgecolor="black")
    ax[1].set_title("Средняя доля правильных P")
    ax[1].set_ylim(0, 1)
    ax[2].bar(labels, [s["good"] for s in summary], color=shades, edgecolor="black")
    ax[2].set_title("Удачных оценок гомографии (E ≤ 5) из 15")
    ax[2].set_ylim(0, 15)
    finish(fig, "p8_summary_bars.png")

    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
    for a, seq in zip(axs.ravel(), SEQUENCES):
        for method, mk, ls in [("ORB", "s", "--"), ("SIFT", "o", "-")]:
            rs = [r for r in results if r["seq"] == seq and r["method"] == method]
            a.plot([r["j"] for r in rs], [r["C"] for r in rs], ls, marker=mk, color="black",
                   label=method)
        a.set_title(seq)
        a.set_xlabel("номер второго изображения")
        a.set_xticks(range(2, 7))
        a.grid(alpha=0.3)
        a.legend()
    axs[0, 0].set_ylabel("C")
    axs[1, 0].set_ylabel("C")
    finish(fig, "p8_C_by_pair.png")


    def badness(r):
        return (r["failed"], -1 if r["E"] is None else r["E"], -r["P"])

    picks = {}
    for method in METHODS:
        rs = [r for r in results if r["method"] == method]
        best = min((r for r in rs if not r["failed"]), key=lambda r: (r["E"], -r["P"]))
        worst = max(rs, key=badness)
        picks[method] = (best, worst)
        draw_pair(best, "Удачная пара", f"p8_best_{method}.png")
        draw_pair(worst, "Худшая пара", f"p8_worst_{method}.png")
        print(f"  {method}: удачная {best['seq']} 1->{best['j']} (E={fmt(best['E'], 2)}), "
              f"худшая {worst['seq']} 1->{worst['j']} (E={fmt(worst['E'], 2)}, P={worst['P']:.3f})")

    flow = optical_flow()
