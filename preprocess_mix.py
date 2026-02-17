import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import sys


try:
    from split_board_v3 import split_board, isolate_chessboard, get_classifications
    from gray_bar_detection import detect_board_x_edges
except ImportError:
    print("Error: Could not find 'split_board_v3.py' or 'gray_bar_detection.py'")
    sys.exit(1)

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASETS = [
    # (Folder Name, CSV Filename)
    ("synthetic_dataset_v1", "gt.csv"),         # Grayscale-style
    ("synthetic_dataset_color", "gt.csv") # Color-style
    ,("synthetic_dataset_extra1", "gt.csv")     # Different chess set
]

PER_DATASET_LIMIT  = 800 #Max number of images from each dataset

OUTPUT_FILE = "processed_data.pt"

# ==========================================
# 2. THE MIXING FUNCTION
# ==========================================
def preprocess_mixed():
    print(f"Starting Mixed Pre-processing")
    
    all_squares = []
    all_labels = []
    
    # Transform: Resize to 64x64 and convert to Tensor
    # Note: This preserves 3 Channels (RGB) for both datasets
    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Loop through both datasets
    for root_dir, csv_file in DATASETS:
        print(f"Processing Dataset: {root_dir}")
        
        img_folder = os.path.join(root_dir, "images")
        label_path = os.path.join(root_dir, csv_file)
        
        if not os.path.exists(img_folder):
            print(f"Skipping {root_dir}: 'images' folder not found")
            continue
        if not os.path.exists(label_path):
            print(f"Skipping {root_dir}: CSV {csv_file} not found")
            continue

        # Get list of images
        images = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


        images = images[:PER_DATASET_LIMIT]
        
        for img_name in tqdm(images, desc=f"   Splitting {root_dir}"):

            try:
                full_path = os.path.join(img_folder, img_name)
                
                # 1. Isolate the Board (Crop)
                board = isolate_chessboard(full_path)
                
                # 2. Get Labels (Parses FEN from CSV)
                # This uses your existing logic to read the CSV
                labels = get_classifications(img_name, label_path)
                
                # 3. Split into 64 Squares
                squares_list = split_board(
                    board, 
                    labels=labels, 
                    margin=20, 
                    gray_bar_detection=detect_board_x_edges(board)
                )
                
                # 4. Collect the tiny squares
                for square_img, label in squares_list:
                    # Convert to Tensor immediately
                    tensor_img = to_tensor(square_img)
                    
                    all_squares.append(tensor_img)
                    all_labels.append(label)

            except Exception as e:
                # If an image fails skip it
                pass

    # ==========================================
    # 3. STACK AND SAVE
    # ==========================================
    if len(all_squares) == 0:
        return

    data_tensor = torch.stack(all_squares)
    label_tensor = torch.tensor(all_labels)
    
    print(f"Saving to {OUTPUT_FILE}")
    torch.save((data_tensor, label_tensor), OUTPUT_FILE)

if __name__ == "__main__":
    preprocess_mixed()