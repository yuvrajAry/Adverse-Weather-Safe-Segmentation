import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AppConfig
from .data.iddaw_dataset import IDDAWDataset
from .models import build_model


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    # pred, target: [N,H,W] int
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        targ_c = target == c
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    train_set = IDDAWDataset(root=AppConfig.DATASET_ROOT, split='train', size=(args.width, args.height), fusion_mode=args.fusion, num_classes=AppConfig.NUM_CLASSES)
    val_set = IDDAWDataset(root=AppConfig.DATASET_ROOT, split='val', size=(args.width, args.height), fusion_mode=args.fusion, num_classes=AppConfig.NUM_CLASSES)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, args.fusion, AppConfig.NUM_CLASSES, None).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_miou = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x = x.to(device)
            y = y.to(device)
            logits = model.forward_logits(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_miou = 0.0
        count = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x = x.to(device)
                y = y.to(device)
                logits = model.forward_logits(x)
                pred = logits.argmax(dim=1)
                val_miou += compute_miou(pred, y, AppConfig.NUM_CLASSES)
                count += 1
        val_miou = val_miou / max(1, count)

        # save latest
        latest_path = os.path.join(args.out_dir, f"{args.model}_{args.fusion}_latest.pt")
        torch.save({
            'epoch': epoch,
            'model': args.model,
            'fusion': args.fusion,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_miou': val_miou,
        }, latest_path)

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(args.out_dir, f"{args.model}_{args.fusion}_best.pt")
            torch.save({
                'epoch': epoch,
                'model': args.model,
                'fusion': args.fusion,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_miou': val_miou,
            }, best_path)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_miou={val_miou:.4f} best_miou={best_miou:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='fast_scnn', choices=['fast_scnn','mobilenetv3_lite'])
    p.add_argument('--fusion', default='early', choices=['early','mid'])
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=4)
    # no longer used; structure inferred
    p.add_argument('--rgb-glob', default='')
    p.add_argument('--out-dir', default='checkpoints')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)


