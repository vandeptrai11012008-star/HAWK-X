"""Train HAWK-X native detector on local, appropriately licensed images."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hawkx_native import HAWKXDetector, PersonBoxDataset, detection_loss


def main():
    parser = argparse.ArgumentParser(description="Train HAWK-X from random weights")
    parser.add_argument("--data", required=True, help="JSON annotations (normalized boxes)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out", default="vision-ai/checkpoints/hawkx_native.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = PersonBoxDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = HAWKXDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Training randomly initialized HAWK-X CNN on {len(dataset)} samples (CPU).")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for pixels, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = detection_loss(model(pixels), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(pixels)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={running / len(dataset):.4f}")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "hawkx-grid-v0.1",
                "state_dict": model.state_dict(),
                "training_samples": len(dataset)}, path)
    print(f"Saved experimental checkpoint to {path}. Accuracy has NOT been validated.")


if __name__ == "__main__":
    main()
