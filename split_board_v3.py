import cv2
import numpy as np
from tqdm import tqdm
import os, sys
from gray_bar_detection import detect_board_x_edges

classes_dict = {"P":0,"R":1,"N":2,"B":3,"Q":4,"K":5,"p":6,"r":7,"n":8,"b":9,"q":10,"k":11,"E":12, "O":13}


def pad_to_size(img, target_h, target_w, pad_value=0):
    h, w = img.shape[:2]

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img.ndim == 2:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    

    return np.pad(img, padding, mode="constant", constant_values=pad_value)[:target_h, :target_w]

def matrix_from_labels(labels = "8/8/8/8/8/8/8/8/"):
    rows = labels.split('/')
    matrix = []
    for row in rows:
        matrix_row = []
        for char in row:
            if char.isdigit():
                matrix_row.extend([12] * int(char))
            else:
                matrix_row.append(classes_dict[char])
        matrix.append(matrix_row)
    return matrix

def isolate_chessboard(input_path):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not load image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        # Chessboard is roughly square
        if not (0.8 < aspect < 1.25):
            continue

        roi = gray[y:y+h, x:x+w]

        # Measure internal structure (grid-like edges)
        roi_edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(roi_edges > 0) / roi_edges.size

        score = area * edge_density

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        raise RuntimeError("Chessboard not found")

    x, y, w, h = cv2.boundingRect(best_cnt)
    board = img[y:y+h, x:x+w]

    return board

def split_board(board,labels = "8/8/8/8/8/8/8/8/", margin = 20, gray_bar_detection = {"board_left_x":0,"board_right_x":0,"has_gray_left":False,"has_gray_right":False}):
    # Split the board into 8x8 squares
    squares = []
    if gray_bar_detection["has_gray_left"]:
        start = gray_bar_detection["board_left_x"]
    else:
        start = 0
    if gray_bar_detection["has_gray_right"]:
        end = gray_bar_detection["board_right_x"]
    else:
        end = board.shape[1]    
    h, w, _ = board.shape
    wt = end-start
    square_h = h // 8
    square_w = wt // 8  
    pad_size = square_h + 2 * margin+ 10
    for row in range(8):
        for col in range(8):
            y1 = max(row * square_h - margin,0)
            y2 = min((row + 1) * square_h + margin,h-1)
            x1 = max(col * square_w - margin,0)
            x2 = min((col + 1) * square_w + margin,w-start-1)
            square = board[max(y1,0):min(y2,h-1), max(x1+start,0):min(x2+start,w-1)]
            square = pad_to_size(square, pad_size, pad_size, pad_value=0)
            squares.append((square,matrix_from_labels(labels)[row][col]))
    return squares

def get_classifications(file_name = "",classification_file = "gt.csv"):
    with open(classification_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            name, labels, viewpoint = line.strip().split(',')
            if name == str(file_name):
                return labels
    raise ValueError("Classification not found for file: " + str(file_name))


    
def main()->list[tuple[np.ndarray,int]]:
    input_folder = "images"
    images_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]
    to_return = []
    for image_file in tqdm(images_files):
        image_path =input_folder+ "\\" + image_file
        try:

            board = isolate_chessboard(image_path)
            gray_bar_info = detect_board_x_edges(board)
            labels = get_classifications(image_file,"gt.csv")
            squares = split_board(board,labels=labels,margin=20,gray_bar_detection=gray_bar_info)
            for image, label in squares:
                to_return.append((image,label))

        except Exception as e:
            print(f"\nError processing {image_file}: {e}", file=sys.stderr)
    return to_return

if __name__ == "__main__":
    #main()
    im_name  = "req0e.png"
    board = isolate_chessboard("images\\"+im_name)
    gray_bar_info = detect_board_x_edges(board)
    labels = get_classifications(im_name,"gt.csv")
    sqs = split_board(board,labels=labels,margin=20,gray_bar_detection=gray_bar_info)
    for sq , l in sqs:
        cv2.imshow("square",sq)
        cv2.waitKey(0)
        cv2.destroyAllWindows()