import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from preprocess import preprocess



class MultiHeadResNet(nn.Module):
    def __init__(self, device="cpu", embed_dim=128):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        feature_size = base_model.fc.in_features

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        self.type_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

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
        embedding = nn.functional.normalize(self.embedding_head(features), dim=1)
        return self.type_head(features), self.color_head(features), embedding



class MultiHeadWrapper(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset):
        self.ds = tensor_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        label = int(label.item())

        if label == 12:
            type_target = 6
            color_target = 2
        elif label < 6:
            type_target = label
            color_target = 0
        else:
            type_target = label - 6
            color_target = 1

        return img, torch.tensor(type_target), torch.tensor(color_target)


def batch_hard_triplet(embeddings, labels, base_margin=1.0):
    dist = torch.cdist(embeddings, embeddings, p=2)
    type_labels = labels // 10
    color_labels = labels % 10
    type_loss = 0.0; color_loss = 0.0
    type_valid = 0; color_valid = 0

    for i in range(len(embeddings)):
        # Type loss
        t_i = type_labels[i]
        pos_mask = type_labels == t_i
        
        # Define Negatives
        if t_i == 6: neg_mask = type_labels != 6
        else:
            neg_mask = type_labels == 6 # Default neg is Empty
            if t_i == 0: neg_mask |= (type_labels == 3) # Pawn vs Bishop
            elif t_i == 3: neg_mask |= ((type_labels == 0) | (type_labels == 4)) # Bishop vs Pawn OR Queen
            elif t_i == 4: # Queen
                neg_mask |= (type_labels == 3) # Fight Bishop
                neg_mask |= (type_labels == 5) # Fight King 
            elif t_i == 5: # King
                neg_mask |= (type_labels == 4) # Fight Queen

        # If Anchor is Empty(6) -> Push 2.0 distance
        # Everyone else -> Push 1.0 distance
        curr_margin = base_margin
        if t_i == 6:
            curr_margin = 2.0
        if t_i == 0 or t_i == 3 or t_i == 4 or t_i == 5:  # Pawn, Bishop, Queen are "troublemakers"
            curr_margin = 1.3

        if pos_mask.sum() >= 2 and neg_mask.sum() >= 1:
            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            type_loss += torch.relu(hardest_pos - hardest_neg + curr_margin)
            type_valid += 1

        # Color loss
        c_i = color_labels[i]
        pos_mask = color_labels == c_i
        neg_mask = color_labels != c_i
        if pos_mask.sum() >= 2 and neg_mask.sum() >= 1:
            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            # Color uses standard margin
            color_loss += torch.relu(hardest_pos - hardest_neg + base_margin)
            color_valid += 1

    return (type_loss / max(type_valid, 1), color_loss / max(color_valid, 1))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    OUTPUT_FOLDER = f"{args.source}_finetuned"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting fine-tuning")

    if not os.path.exists("real_processed.pt"):
        preprocess(ROOT_DIR="real_data_pack", OUTPUT_FILE="real_processed.pt",CSV_FILE="gt.csv")


    # LOAD REAL DATA
    
    real_train_x, real_train_y = torch.load("real_processed.pt")


    real_train_x = (real_train_x - 0.5) / 0.5


    train_ds = MultiHeadWrapper(TensorDataset(real_train_x, real_train_y))


    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


    model = MultiHeadResNet(device=DEVICE)
    model.load_state_dict(torch.load(f"{args.source}/multihead_model.pth", map_location=DEVICE))

    # Freeze BACKBONE
    for p in model.parameters():
        p.requires_grad = False
    for p in model.backbone[7].parameters():
        p.requires_grad = True
    for p in model.type_head.parameters():
        p.requires_grad = True
    for p in model.color_head.parameters():
        p.requires_grad = True
    for p in model.embedding_head.parameters():
        p.requires_grad = True 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    ce_t = nn.CrossEntropyLoss()
    ce_c = nn.CrossEntropyLoss()

    λ_type = 0.3
    λ_color = 0.3

    # TRAIN
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for imgs, t_lbl, c_lbl in loop:
            imgs, t_lbl, c_lbl = imgs.to(DEVICE), t_lbl.to(DEVICE), c_lbl.to(DEVICE)

            optimizer.zero_grad()
            pt, pc, emb = model(imgs)

            cls_loss = ce_t(pt, t_lbl) + ce_c(pc, c_lbl)

            triplet_labels = t_lbl * 10 + c_lbl
            type_tri, color_tri = batch_hard_triplet(emb, triplet_labels)

            loss = cls_loss + λ_type * type_tri + λ_color * color_tri
            loss.backward()
            optimizer.step()

        print(
            f"→ Epoch {epoch+1}: "
            f"Cls {cls_loss.item():.3f} | "
            f"T_tri {type_tri.item():.3f} | "
            f"C_tri {color_tri.item():.3f}"
        )

    torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/finetuned_multihead.pth")
    print("Fine-tuned model saved.")


if __name__ == "__main__":
    main()
