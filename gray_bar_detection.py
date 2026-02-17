import cv2
import numpy as np

#a slow fix if normal method fails
def count_edge_columns(img, low=50, high=70):
    diff = img.shape[1]-img.shape[0]
    if diff>10:
        left_gray_count = len([x for x in img[:,0] if low<x[0]<high and low<x[1]<high and low<x[2]<high])
        right_gray_count = len([x for x in img[:,-1] if low<x[0]<high and low<x[1]<high and low<x[2]<high])
        
        if left_gray_count>right_gray_count:
            return diff, img.shape[1]
        else:
            return 0, img.shape[1]-diff
    return 0, img.shape[1]


def detect_board_x_edges(img, debug=False):

    if img is None:
        raise ValueError("Could not load image")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        raise RuntimeError("No lines detected")

    # --- collect angles ---
    angles = []
    for l in lines:
        rho, theta = l[0]
        angles.append(theta)

    # --- find dominant angle (histogram peak) ---
    hist, bins = np.histogram(angles, bins=180)
    theta0 = bins[np.argmax(hist)]

    # --- keep lines close to dominant angle ---
    parallel = []
    for l in lines:
        rho, theta = l[0]
        if abs(theta - theta0) < np.deg2rad(6):
            parallel.append((rho, theta))

    if len(parallel) < 4:
        raise RuntimeError("Insufficient parallel lines")

    # --- compute x-intercepts ---
    xs = []
    for rho, theta in parallel:
        # line eq: x*cosθ + y*sinθ = rho
        # at y = 0 → x = rho / cosθ
        if abs(np.cos(theta)) < 1e-3:
            continue
        x = rho / np.cos(theta)
        xs.append(x)

    xs = sorted(xs)

    # --- cluster projected x's ---
    if not xs or len(xs) < 2:
        x_left, x_right = count_edge_columns(img)
        return {
            "board_left_x": x_left,
            "board_right_x": x_right,
            "has_gray_left": x_left > 5,
            "has_gray_right": x_right < w - 5
    }
    clusters = [[xs[0]]]
    for x in xs[1:]:
        if abs(x - clusters[-1][-1]) < 20:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    centers = [int(sum(c) / len(c)) for c in clusters]

    if len(centers) < 2:
        raise RuntimeError("Could not determine board edges")

    # --- choose widest pair (outer edges) ---
    board_left_x = min(centers)
    board_right_x = max(centers)

    has_gray_left = board_left_x > 5
    has_gray_right = board_right_x < w - 5

    if debug:
        vis = img.copy()
        cv2.line(vis, (board_left_x, 0), (board_left_x, h), (0, 255, 0), 2)
        cv2.line(vis, (board_right_x, 0), (board_right_x, h), (0, 255, 0), 2)

        for x in centers:
            cv2.line(vis, (int(x), 0), (int(x), h), (0, 0, 255), 1)


    return {
        "board_left_x": int(board_left_x),
        "board_right_x": int(board_right_x),
        "has_gray_left": has_gray_left,
        "has_gray_right": has_gray_right
    }



