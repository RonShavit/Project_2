import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


class MultiHeadResNet(nn.Module):
    def __init__(self, device="cpu", embed_dim=128):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        feature_size = base_model.fc.in_features

        # Embedding head (Triplet loss)
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        # head 1: type
        self.type_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

        # head 2: color
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



class MultiHeadDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensors, label_tensors):
        self.data = data_tensors
        self.labels = label_tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = (img - 0.5) / 0.5  # normalize

        label = int(self.labels[idx].item())
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
        # TYPE LOSS
        t_i = type_labels[i]
        pos_mask = type_labels == t_i
        
        # Define Negatives (Symmetric Logic)
        if t_i == 6: neg_mask = type_labels != 6
        else:
            neg_mask = type_labels == 6 # Default neg is Empty
            if t_i == 0: neg_mask |= (type_labels == 3) # Pawn vs Bishop
            elif t_i == 3: neg_mask |= ((type_labels == 0) | (type_labels == 4)) # Bishop vs Pawn OR Queen
            elif t_i == 4: neg_mask |= (type_labels == 3) # Queen vs Bishop
            
        curr_margin = base_margin
        if t_i == 6:
            curr_margin = 2.0  # Empty gets huge margin
        if t_i == 0 or t_i == 3 or t_i == 4:  # Pawn, Bishop, Queen
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
            color_loss += torch.relu(hardest_pos - hardest_neg + base_margin)
            color_valid += 1

    return (type_loss / max(type_valid, 1), color_loss / max(color_valid, 1))



def plot_metrics(train_accs, val_accs, losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.legend()
    plt.title("Combined Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss')
    plt.legend()
    plt.title("Training Loss")

    plt.savefig(save_path)
    plt.close()


def generate_multihead_confusion_matrix(model, loader, device, save_path):
    y_t, y_t_p, y_c, y_c_p = [], [], [], []

    model.eval()
    with torch.no_grad():
        for x, t, c in loader:
            x = x.to(device)
            pt, pc, _ = model(x)

            y_t.extend(t.numpy())
            y_t_p.extend(torch.argmax(pt, 1).cpu().numpy())

            y_c.extend(c.numpy())
            y_c_p.extend(torch.argmax(pc, 1).cpu().numpy())

    cm_t = confusion_matrix(y_t, y_t_p)
    cm_c = confusion_matrix(y_c, y_c_p)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_t, annot=True, fmt='d')
    plt.title("Type Confusion Matrix")
    plt.savefig(save_path.replace(".png", "_TYPE.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_c, annot=True, fmt='d')
    plt.title("Color Confusion Matrix")
    plt.savefig(save_path.replace(".png", "_COLOR.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="experiment_multihead_triplet")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_tensors, label_tensors = torch.load("processed_data.pt")
    dataset = MultiHeadDataset(data_tensors, label_tensors)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)

    model = MultiHeadResNet(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)

    ce_type = nn.CrossEntropyLoss()
    ce_color = nn.CrossEntropyLoss()

    triplet_weight = 0.3

    train_accs, val_accs, losses = [], [], []

    for epoch in range(args.epochs):
        model.train()
        correct, total, run_loss = 0, 0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for imgs, t_lbl, c_lbl in loop:
            imgs = imgs.to(device)
            t_lbl = t_lbl.to(device)
            c_lbl = c_lbl.to(device)

            optimizer.zero_grad()

            pt, pc, emb = model(imgs)

            cls_loss = ce_type(pt, t_lbl) + ce_color(pc, c_lbl)

            triplet_labels = t_lbl * 10 + c_lbl
            tri_loss_type, tri_loss_color = batch_hard_triplet(emb, triplet_labels)

            loss = cls_loss + triplet_weight * (tri_loss_type + tri_loss_color)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()

            pred_t = pt.argmax(1)
            pred_c = pc.argmax(1)
            correct += ((pred_t == t_lbl) & (pred_c == c_lbl)).sum().item()
            total += t_lbl.size(0)

            loop.set_postfix(loss=loss.item(), acc=f"{100*correct/total:.1f}%")

        train_accs.append(100 * correct / total)
        losses.append(run_loss / len(train_loader))

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, t_lbl, c_lbl in val_loader:
                imgs = imgs.to(device)
                t_lbl = t_lbl.to(device)
                c_lbl = c_lbl.to(device)

                pt, pc, _ = model(imgs)
                v_correct += ((pt.argmax(1) == t_lbl) &
                              (pc.argmax(1) == c_lbl)).sum().item()
                v_total += t_lbl.size(0)

        val_accs.append(100 * v_correct / v_total)
        print(f"Train {train_accs[-1]:.1f}% | Val {val_accs[-1]:.1f}%")

    torch.save(model.state_dict(), f"{args.output}/multihead_model.pth")
    plot_metrics(train_accs, val_accs, losses, f"{args.output}/training.png")
    generate_multihead_confusion_matrix(
        model, val_loader, device,
        f"{args.output}/confusion.png"
    )


if __name__ == "__main__":
    main()
