import numpy as np
from torch.utils.data import Dataset
import os
from split_board_v3 import split_board, isolate_chessboard, get_classifications
from gray_bar_detection import detect_board_x_edges
import random


def map_label_to_multihead(label_idx):
    """
    Converts 0-12 classification into:
    Head 1 (Type): 0=P, 1=R, 2=N, 3=B, 4=Q, 5=K, 6=Empty
    Head 2 (Color): 0=White, 1=Black, 2=Empty
    """
    if label_idx == 12: # Empty Square
        return 6, 2 
    
    elif label_idx < 6: # White Pieces (0-5)
        # Type is same as index (e.g., 0 is Pawn)
        return label_idx, 0 
        
    else: # Black Pieces (6-11)
        # Type is index minus 6 (e.g., 6 is Pawn)
        return label_idx - 6, 1
    
class ChessDataset(Dataset):
    def __init__(self, transform=None, root_dir=".", csv_file="gt.csv"):
        """
        root_dir: The folder containing both the 'images' subfolder and the CSV.
        csv_file: The name of the CSV file inside root_dir.
        """
        random.seed(0)
        self.transform = transform

        # 1. Define paths based on the nested folder structure
        self.img_folder = os.path.join(root_dir, "images")  # Looks for root/images
        self.label_path = os.path.join(root_dir, csv_file)  # Looks for root/csv

        # 2. Safety Check: Make sure the folder actually exists
        if not os.path.exists(self.img_folder):
            # Fallback logic: if we are looking in root "." and didn't find it, check if it's just 'images'
            if root_dir == "." and os.path.exists("images"):
                self.img_folder = "images"
                self.label_path = csv_file
            else:
                 # If we still can't find it, just print a warning but try to proceed (or let it crash later)
                 print(f"Warning: Could not find 'images' folder inside {root_dir}")

        self.images = os.listdir(self.img_folder)
        self.curr_split = []
        self.last_image_used = -1

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.images)*64

    def __getitem__(self, idx):
        image_idx = idx // 64
        square_idx = idx % 64
        if image_idx != self.last_image_used:
            self.last_image_used = image_idx
            image_file = self.images[image_idx]

            # Construct full path to the specific image
            image_path = os.path.join(self.img_folder, image_file)

            board = isolate_chessboard(image_path)

            # Pass the FULL PATH to the CSV file
            labels = get_classifications(image_file, self.label_path)

            self.curr_split = split_board(board, labels=labels, margin=20, gray_bar_detection=detect_board_x_edges(board))

        square, original_label = self.curr_split[square_idx]
        # Convert single label (0-12) to Multi-Head (Type, Color)
        type_label, color_label = map_label_to_multihead(original_label)

        if self.transform:
            square = self.transform(square)
            
        return square, type_label, color_label