from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from dataloader import SHHA_loader

import matplotlib.pyplot as plt

def data_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data, 0)
    return [data, target]

def draw_and_save(images, coords, save_path, batch_idx):
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    for i in range(images.shape[0]):
        image = images[i].permute((1, 2, 0))
        image = image * std + mean
        image = image.numpy()
        coord = coords[i]
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        ax.plot(coord[:, 0], coord[:, 1], 'ro')
        plt.savefig(f"{save_path}/image_{batch_idx+i}.png")
        plt.close()

def train(args):
    # Create Visualization folder
    import os
    if not os.path.exists("./train_images"):
        os.makedirs("./train_images")
    if not os.path.exists("./test_images"):
        os.makedirs("./test_images")
    # You can delete this part if you don't want to visualize the data

    # Define dataloader
    train_dataset = SHHA_loader(args.data_path, "train", args.output_size)
    test_dataset = SHHA_loader(args.data_path, "test", args.output_size)
    train_loader = DataLoader(
        train_dataset, args.batch_size, True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=data_collate)
    test_loader = DataLoader(
        test_dataset, args.batch_size, False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=data_collate)
    
    # TODO Define model and optimizer

    for epoch in range(args.num_epoch):
        for batch_idx, inputs in enumerate(train_loader):
            images, gt = inputs

            # Visualize data, you can delete this part if you don't want to visualize the data
            draw_and_save(images, gt, "./train_images", batch_idx*args.batch_size)

            # TODO Forward

            # TODO Backward

            # TODO Update parameters
            
            # TODO Print log info

    # Save model checkpoints

    for batch_idx, inputs in enumerate(test_loader):
        images, gt = inputs

        # Visualize data, you can delete this part if you don't want to visualize the data
        draw_and_save(images, gt, "./test_images", batch_idx)

        # TODO Test model performance
        


    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./ShanghaiTech_Crowd_Counting_Dataset/part_A_final")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--output_size', type=int, default=512)
    args = parser.parse_args()
    train(args)