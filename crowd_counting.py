from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from dataloader import SHHA_loader

def train(args):
    train_dataset = SHHA_loader(args.data_path, "train")
    test_dataset = SHHA_loader(args.data_path, "test")
    train_loader = DataLoader(
        train_dataset, args.batch_size, True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, args.batch_size, False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Define model and optimizer

    for epoch in range(args.num_epoch):
        for batch_idx, inputs in enumerate(train_loader):
            images, gt = inputs

            # Forward

            # Backward

            # Update parameters
            
            # Print log info

    # Save model checkpoints

    for batch_idx, inputs in enumerate(test_loader):
        images, gt = inputs
        # Test model performance


    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./ShanghaiTech_Crowd_Counting_Dataset/part_A_final")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=50)
    args = parser.parse_args()
    train(args)