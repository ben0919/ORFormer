import os
import os.path
from os import path
import sys
import os

os.chdir(sys.path[0])

import argparse

from Config import cfg

from Dataloader import WFLW_heatmap_Dataset, W300_heatmap_Dataset, COFW_heatmap_Dataset

import torch

import numpy as np

import torchvision.transforms as transforms

from tqdm import tqdm

import wandb

import torchlm

from Model.VQVAE import VQVAE


def parse_args():
    parser = argparse.ArgumentParser(description='Train Quantized Heatmap Generator')

    parser.add_argument('--lr', help='learning rate', type=float, default=0.0007)
    parser.add_argument('--T_0', help='Cosine Annealing T_0', type=int, default=5)
    parser.add_argument('--T_mult', help='Cosine Annealing T_mult', type=int, default=2)
    parser.add_argument('--epoch', help='epoch number', type=int, default=300)
    parser.add_argument('--dataset', help='dataset', type=str, default="WFLW")
    parser.add_argument('--resultDir', help='result directory', type=str, default="/4TB/jcchiang/results/temp/WFLW")
    parser.add_argument('--name', help='name', type=str, default="")
    parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
    parser.add_argument('--alpha', help='alpha', type=float, default=100)

    args = parser.parse_args()

    return args

def save_img(predicted_image, gt_image, path, epoch, i):
    image = torch.squeeze(predicted_image)
    img = transforms.ToPILImage()(image)
    img.save(f"{path}/Epoch{epoch}_{i}_pre.jpg")
    image = torch.squeeze(gt_image)
    img = transforms.ToPILImage()(image)
    img.save(f"{path}/Epoch{epoch}_{i}_gt.jpg")

def save_model(model, path):
    torch.save(model.state_dict(), f"{path}/best_model.pt") 

def main_function():

    args = parse_args()
    args.name += f"_lr{args.lr}_T0{args.T_0}_Tmult{args.T_mult}_epoch{args.epoch}_batch{args.batch_size}_alpha{args.alpha}"
    resultDir = os.path.join(args.resultDir, args.name)

    wandb.init(
        # set the wandb project where this run will be logged
        project = "train_codebook_" + args.dataset,
        name =  args.name,
        
        # track hyperparameters and run metadata
        config = {
            "learning_rate": args.lr,
            "T_mult": args.T_mult,
            "architecture": "VQVAE",
            "dataset": args.dataset,
            "epochs": args.epoch,
        }
    )

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device("cuda")
    
    if args.dataset == "WFLW":
        model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.WFLW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False)
    if args.dataset == "300W":
        model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.W300.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False)
    if args.dataset == "COFW":
        model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.COFW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False)
    model = model.to(device)

    augmentation_transform = torchlm.LandmarksCompose([
        torchlm.bind(transforms.RandomGrayscale(p=0.5)),
        torchlm.LandmarksRandomRotate(angle=(-30, 30)),
        torchlm.LandmarksRandomTranslate(translate=(-0.04, 0.04)),
        torchlm.LandmarksRandomScale(scale=(-0.05, 0.05)),
        torchlm.LandmarksResize((256, 256), keep_aspect=True)
    ])

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])

    if args.dataset == "WFLW":
        train_dataset = WFLW_heatmap_Dataset(
            cfg, cfg.WFLW.ROOT,
            subset = 'train',
            augmentation_transform = augmentation_transform,
            normalize_transform = normalize_transform,
            edge_type="VQVAE"
        )
        valid_dataset = WFLW_heatmap_Dataset(
            cfg, cfg.WFLW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="VQVAE"
        )
    if args.dataset == "300W":
        train_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'train',
            augmentation_transform = augmentation_transform,
            normalize_transform = normalize_transform,
            edge_type="VQVAE",
            mirror=True
        )
        valid_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="VQVAE",
            mirror=False
        )
    if args.dataset == "COFW":
        train_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'train',
            augmentation_transform = augmentation_transform,
            normalize_transform = normalize_transform,
            edge_type="VQVAE",
            mirror=True
        )
        valid_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="VQVAE",
            mirror=False
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=0.001*args.lr)

    if not path.exists(resultDir):
        os.system(f"mkdir {resultDir}")

    best_heatmap_loss = 10000
    for epoch in range(args.epoch):
        print("Epoch: %d" % (epoch+1))
        # train
        train_emb_loss = 0
        train_heatmap_loss = 0
        total = 0
        for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            embedding_loss, x_hat, perplexity, _, _, _, _, _, _ = model(resized_input.to(device))
            gt_heatmap = meta["Edge_Heatmaps"].cuda()
            heatmap_loss = torch.mean((x_hat-gt_heatmap)**2, axis=1).sum()
            loss = heatmap_loss + embedding_loss * args.alpha
            loss.backward()
            optimizer.step()
            scheduler.step(epoch+i/len(train_loader))
            train_emb_loss += embedding_loss.cpu()
            train_heatmap_loss += heatmap_loss.cpu()
            total += input.shape[0]
        train_emb_loss = train_emb_loss/total
        train_heatmap_loss = train_heatmap_loss/total

        # valid
        if (epoch+1) % 3 == 0:
            model.eval()
            with torch.no_grad():
                valid_emb_loss = 0
                valid_heatmap_loss = 0
                total = 0
                for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(valid_loader)):
                    embedding_loss, x_hat, perplexity, _, _, _, _, _, _ = model(resized_input.to(device))
                    valid_emb_loss += embedding_loss.cpu()
                    gt_heatmap = meta["Edge_Heatmaps"].cuda()
                    heatmap_loss = torch.mean((x_hat-gt_heatmap)**2, axis=1).sum()
                    valid_heatmap_loss += heatmap_loss.cpu()
                    
                    total += input.shape[0]
                valid_emb_loss = valid_emb_loss/total
                valid_heatmap_loss = valid_heatmap_loss/total
                lr = np.float64(scheduler.get_last_lr()[0])
                if valid_heatmap_loss < best_heatmap_loss:
                    best_heatmap_loss = valid_heatmap_loss
                    save_model(model, resultDir)
                wandb.log({"Train_Emb_Loss": train_emb_loss, "Train_Heatmap_Loss": train_heatmap_loss, 
                           "Valid_Emb_Loss": valid_emb_loss,  "Valid_Heatmap_Loss": valid_heatmap_loss, 
                           "lr": lr, "Best_Heatmap_Loss": best_heatmap_loss})
                


if __name__ == '__main__':
    main_function()

