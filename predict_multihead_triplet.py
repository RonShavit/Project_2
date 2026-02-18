import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from preprocess import preprocess
import os
import cv2
from split_board_v3 import get_classifications, matrix_from_labels, split_board
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score







_CACHED_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MultiHeadResNet(nn.Module):
    def __init__(self, device="cpu", embed_dim=128):
        super().__init__()
        from torchvision.models import resnet50

        base_model = resnet50(weights=None)
        base_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        feature_size = base_model.fc.in_features

        # Embedding head 
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        # Type head (7)
        self.type_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

        # Color head (3)
        self.color_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

        self.to(device)

    def forward(self, x):
        features = self.backbone(x)
        embedding = nn.functional.normalize(
            self.embedding_head(features), dim=1
        )
        return self.type_head(features), self.color_head(features), embedding


def _load_model_once():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model_path ="experiment_base_optimized/multihead_model.pth" if parser.parse_args().zero_shot else "experiment_base_optimized_finetuned/finetuned_multihead.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = MultiHeadResNet(device=_DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=_DEVICE))
    model.eval()

    _CACHED_MODEL = model
    return model



def _multihead_to_single_class(type_pred, color_pred):
    
    if type_pred == 6 or color_pred == 2:
        return 12  # Empty
    if color_pred == 0:
        return type_pred          # White
    if color_pred == 1:
        return type_pred + 6      # Black
    return 12


def predict_board(image: np.ndarray, use_tta=True) -> torch.Tensor:
    """
    Predict an 8x8 chessboard using Test Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, and Brightness Boost.
    """
    model = _load_model_once()

    # Standard Preprocess
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # TTA: Brightness Transform
    bright_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=0.3), # Lighten shadows
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    squares = split_board(image, margin=20)
    predictions = []

    with torch.no_grad():
        for sq_img, _ in squares:
            # Prepare Batch of Augmented Images
            img_orig = base_transform(sq_img)
            
            if use_tta:
                # Augment 1: Horizontal Flip
                img_flip = torch.flip(img_orig, [2]) 
                
                # Augment 2: Brightness (Good for shadows)
                img_bright = bright_transform(sq_img)

                # Stack into a batch of 3: [Original, Flip, Bright]
                batch = torch.stack([img_orig, img_flip, img_bright]).to(_DEVICE)
            else:
                batch = img_orig.unsqueeze(0).to(_DEVICE)

            # Predict Batch
            pred_t, pred_c, _ = model(batch)

            # We average the "votes" from all 3 versions
            avg_t = torch.mean(pred_t, dim=0, keepdim=True)
            avg_c = torch.mean(pred_c, dim=0, keepdim=True)

            # 4. Final Decision
            type_idx = avg_t.argmax(1).item()
            color_idx = avg_c.argmax(1).item()

            predictions.append(
                _multihead_to_single_class(type_idx, color_idx)
            )

    return torch.tensor(
        predictions, dtype=torch.int64
    ).reshape(8, 8).cpu()





def test_accuracy(source="test_data_set", gt="gt.csv"):
    success = 0
    col_success = 0
    type_success = 0
    total = 0
    rc, rt, pc, pt = [], [], [], []  # Lists to store data for plotting

    for fname in tqdm(os.listdir(source + "/images/"), desc="Testing accuracy"):
        if not fname.endswith(".jpg"):
            continue
        
        # Load Data
        FEN = get_classifications(fname, source + "/" + gt)
        FEN = matrix_from_labels(FEN)
        image = cv2.imread(source + "/images/" + fname)
        
        # Predict
        pred_board = predict_board(image)  
        
        # Check Accuracy
        for r in range(8):
            for c in range(8):
                p_val = pred_board[r, c].item()
                g_val = FEN[r][c]

                # Total Success
                if p_val == g_val:
                    success += 1

                # Type Success 
                is_empty_correct = (p_val == 12 and g_val == 12)
                is_piece_type_correct = (p_val != 12 and g_val != 12 and p_val % 6 == g_val % 6)
                if is_empty_correct or is_piece_type_correct:
                    type_success += 1

                #Color Success
                if p_val // 6 == g_val // 6:
                    col_success += 1

                total += 1


                # Add Ground Truth to lists
                if g_val == 12:
                    rc.append(2); rt.append(6)  # Map Empty to separate index
                else:
                    rc.append(g_val // 6)
                    rt.append(g_val % 6)

                # Add Prediction to lists
                if p_val == 12:
                    pc.append(2); pt.append(6)
                else:
                    pc.append(p_val // 6)
                    pt.append(p_val % 6)

    print(f"Accuracy: {success}/{total} = {success/total*100:.2f}%")
    print(f"Type Accuracy: {type_success}/{total} = {type_success/total*100:.2f}%")
    print(f"Color Accuracy: {col_success}/{total} = {col_success/total*100:.2f}%")
    if parser.parse_args().plot_path != "None":
        plot_two_confusions(
            real_pred_1=pt, actual_1=rt,
            real_pred_2=pc, actual_2=rc,
            path=parser.parse_args().plot_path
        )


#Plots two confusion matrices. Used for piece type and color
def plot_two_confusions(real_pred_1, actual_1,
                        real_pred_2, actual_2,
                        labels_1=None,
                        labels_2=None,
                        path = "None",
                        title_1="Piece Type Confusion",
                        title_2="Color Confusion"):

    # Default labels
    if labels_1 is None:
        labels_1 = ["P", "R", "N", "B", "Q", "K", "Empty"]
    if labels_2 is None:
        labels_2 = ["White", "Black", "Empty"]

    # Compute confusion matrices
    cm1 = confusion_matrix(actual_1, real_pred_1, labels=range(len(labels_1)))
    cm2 = confusion_matrix(actual_2, real_pred_2, labels=range(len(labels_2)))

    # Compute accuracies
    acc1 = accuracy_score(actual_1, real_pred_1)
    acc2 = accuracy_score(actual_2, real_pred_2)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # First confusion matrix
    sns.heatmap(cm1,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels_1,
                yticklabels=labels_1,
                ax=axes[0])

    axes[0].set_title(f"{title_1} (Acc: {acc1*100:.1f}%)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Second confusion matrix
    sns.heatmap(cm2,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=labels_2,
                yticklabels=labels_2,
                ax=axes[1])

    axes[1].set_title(f"{title_2} (Acc: {acc2*100:.1f}%)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(path)
    

def print_board(pred_board):
    for r in range(8):
        row_str = ""
        for c in range(8):
            val = pred_board[r, c].item()
            row_str += classes_to_letter[val] + " "
        print(row_str)
                    




classes_to_letter = {
    0: "P",  # white pawn
    1: "R",  # white rook
    2: "N",  # white knight
    3: "B",  # white bishop
    4: "Q",  # white queen
    5: "K",  # white king
    6: "p",  # black pawn
    7: "r",  # black rook
    8: "n",  # black knight
    9: "b",  # black bishop
    10: "q", # black queen
    11: "k", # black king
    12: ".",  # empty square
    13: "?"   # out of distribution
}


parser = argparse.ArgumentParser(epilog="Use predtict_multihead_triplet.py --path your_image.jpg to test a single image, or --test True to run on the whole test set and get accuracys")
if __name__ == "__main__":
    # Simple test

    parser.add_argument("--path", type=str, help="Path to single chessboard image for testing", default="None")
    parser.add_argument("--test",type=bool, default=False, help="Run accuracy test on real test set instead of single image")
    parser.add_argument("--zero_shot", type=bool, default=False, help="Use zero shot  for testing")
    parser.add_argument("--plot_path", type=str, default="None", help="Path to save confusion matrix plot (if --test is True)")
    args = parser.parse_args()

    if args.path != "None":
        img = cv2.imread(args.path)
        pred = predict_board(img)
    if not args.test:
        print_board(pred)
    else:
        if not os.path.exists("real_test_processed.pt"):
            preprocess(ROOT_DIR="test_data_set", OUTPUT_FILE="real_test_processed.pt",CSV_FILE="gt.csv")
        test_accuracy()

    
    