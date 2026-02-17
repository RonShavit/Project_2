import os
import torch
import numpy as np
from tqdm import tqdm
from split_board_v3 import split_board, isolate_chessboard, get_classifications
from gray_bar_detection import detect_board_x_edges
import torchvision.transforms as transforms

# --- CONFIG ---


def preprocess( ROOT_DIR = "synthetic_dataset_v1",
                CSV_FILE = "gt.csv",
                OUTPUT_FILE = "processed_data.pt"):
    print(f"Pre-processing {ROOT_DIR}")
    
    # Check paths
    img_folder = os.path.join(ROOT_DIR, "images")
    label_path = os.path.join(ROOT_DIR, CSV_FILE)
    
    if not os.path.exists(img_folder):
        print("Error: Could not find 'images' folder")
        return

    images = os.listdir(img_folder)
    
    all_squares = []
    all_labels = []
    
    # Minimal transform to get it into a tensor (we resize to 64x64 here to save space)
    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])


    
    # Loop through all images
    for img_name in tqdm(images):
        try:
            full_path = os.path.join(img_folder, img_name)
            
            # 1. Run the Heavy Processing ONCE
            board = isolate_chessboard(full_path)
            labels = get_classifications(img_name, label_path)
            squares_list = split_board(board, labels=labels, margin=20, gray_bar_detection=detect_board_x_edges(board))
            
            # 2. Save the tiny squares
            for square_img, label in squares_list:
                # Convert to Tensor immediately to save space/time later
                tensor_img = to_tensor(square_img)
                
                # We save as 'half precision' (float16) to save RAM, or keep float32
                all_squares.append(tensor_img)
                all_labels.append(label)
                
        except Exception as e:
            print("",end="")

    # Stack into one big tensor

    data_tensor = torch.stack(all_squares)
    label_tensor = torch.tensor(all_labels)
    
    # Save to disk

    torch.save((data_tensor, label_tensor), OUTPUT_FILE)


if __name__ == "__main__":
    preprocess()