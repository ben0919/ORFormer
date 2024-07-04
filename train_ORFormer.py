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

from torch.nn import CrossEntropyLoss

from tqdm import tqdm

import wandb

import torchlm

from Model.VQVAE import VQVAE

from Model.simple_vit import SimpleViT, ORFormer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--T_0', help='Cosine Annealing T_0', type=int, default=5)
    parser.add_argument('--T_mult', help='Cosine Annealing', type=int, default=2)
    parser.add_argument('--epoch', help='epoch number', type=int, default=300)
    parser.add_argument('--validEpoch', help='valid epoch number', type=int, default=1)
    parser.add_argument('--dataset', help='dataset', type=str, default="WFLW")
    parser.add_argument('--resultDir', help='result directory', type=str, default="/4TB/jcchiang/results/temp/300W")
    parser.add_argument('--name', help='name', type=str, default="")
    parser.add_argument('--batch_size', help='batch_size', type=int, default=64)
    parser.add_argument('--alpha', help='alpha', type=float, default=50)
    parser.add_argument('--vit', help='vit', type=str, default="ORFormer")
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
        project = f"train_{args.vit}_{args.dataset}",
        name = args.name,

        # track hyperparameters and run metadata
        config = {
            "learning_rate": args.lr,
            "T_mult": args.T_mult,
            "architecture": f"VQVAE+{args.vit}",
            "dataset": args.dataset,
            "epochs": args.epoch,
        }
    )

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device("cuda")

    if args.dataset == "WFLW":
        original_vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.WFLW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False)
        original_vqvae_model.load_weights("weights/VQVAE/WFLW/best_model.pt")
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.WFLW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        model.load_weights("weights/VQVAE/WFLW/best_model.pt")
    if args.dataset == "300W":
        original_vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.W300.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False)
        original_vqvae_model.load_weights("weights/VQVAE/300W/best_model.pt")
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.W300.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        model.load_weights("weights/VQVAE/300W/best_model.pt")
    if args.dataset == "COFW":
        original_vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.COFW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False)
        original_vqvae_model.load_weights("weights/VQVAE/COFW/best_model.pt")
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.COFW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        model.load_weights("weights/VQVAE/COFW/best_model.pt")
    
    original_vqvae_model = original_vqvae_model.to(device)
    original_vqvae_model.eval()
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

    mask_transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomMask(mask_ratio=0.1, prob=1),
    ])

    if args.dataset == "WFLW":
        train_dataset = WFLW_heatmap_Dataset(
            cfg, cfg.WFLW.ROOT,
            subset = 'train',
            augmentation_transform = augmentation_transform,
            normalize_transform = normalize_transform,
            mask_transform=mask_transform,
            edge_type="VQVAE"
        )
        valid_dataset = WFLW_heatmap_Dataset(
            cfg, cfg.WFLW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            mask_transform=mask_transform,
            edge_type="VQVAE"
        )
    if args.dataset == "300W":
        train_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'train',
            augmentation_transform = augmentation_transform,
            normalize_transform = normalize_transform,
            mask_transform=mask_transform,
            mirror=True,
            edge_type="VQVAE"
        )
        valid_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            mask_transform=mask_transform,
            mirror=False,
            edge_type="VQVAE"
        )
    if args.dataset == "COFW":
        train_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'train',
            augmentation_transform = augmentation_transform,
            normalize_transform = normalize_transform,
            mask_transform=mask_transform,
            mirror=True,
            edge_type="VQVAE"
        )
        valid_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            mask_transform=mask_transform,
            mirror=False,
            edge_type="VQVAE"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    optimizer = torch.optim.Adam(list(model.vit.parameters()) + list(model.encoder.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.T_0, args.T_mult)

    if not path.exists(resultDir):
        os.system(f"mkdir {resultDir}")

    criterion = CrossEntropyLoss(reduction='sum')
    best_heatmap_loss = 10000
    for epoch in range(args.epoch):
        print("Epoch: %d" % (epoch+1))
        # train
        model.train()
        train_heatmap_loss = 0
        train_entropy_loss = 0
        train_OR_entropy_loss = 0
        total = 0
        for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(train_loader)):
            with torch.no_grad():
                _, _, _, _, min_indices, _, _, _, _ = original_vqvae_model(resized_input.to(device))
                target = min_indices.view(resized_input.shape[0], -1).detach().clone()
            optimizer.zero_grad()

            _, x_hat, _, _, _, predicted_min_indices, OR_predicted_min_indices, _, _ = model(resized_occluded_input.to(device))
            gt_heatmap = meta["Edge_Heatmaps"].cuda()
            heatmap_loss = torch.mean((x_hat-gt_heatmap)**2, axis=1).sum()
            train_heatmap_loss += heatmap_loss.cpu()

            if args.vit == "ORFormer":
                prediction = predicted_min_indices.permute(0, 2, 1).contiguous()
                OR_prediction = OR_predicted_min_indices.permute(0, 2, 1).contiguous()
                loss1 = criterion(prediction, target) 
                loss2 = criterion(OR_prediction, target)
                loss3 = heatmap_loss
                loss = loss1 + loss2 + args.alpha * loss3
            if args.vit == "vit":
                prediction = predicted_min_indices.permute(0, 2, 1).contiguous()
                loss1 = criterion(prediction, target)
                loss2 = heatmap_loss
                loss = loss1  + args.alpha * loss2
            # loss = loss1 + loss2
            loss.backward()

            optimizer.step()
            scheduler.step(epoch+i/len(train_loader))

            train_entropy_loss += loss1.cpu()
            train_OR_entropy_loss += loss2.cpu()
            total += resized_input.shape[0]
            
        train_heatmap_loss = train_heatmap_loss/total
        train_entropy_loss = train_entropy_loss/total
        train_OR_entropy_loss = train_OR_entropy_loss/total

        # valid
        if (epoch+1) % args.validEpoch == 0:
            model.eval()
            with torch.no_grad():
                original_heatmap_loss = 0
                valid_heatmap_loss = 0
                valid_entropy_loss = 0
                valid_OR_entropy_loss = 0
                total = 0
                for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(valid_loader)):
                    _, x_hat, _, _, min_indices, _, _, _, _ = original_vqvae_model(resized_input.to(device))
                    target = min_indices.view(resized_input.shape[0], -1).detach().clone()
                    gt_heatmap = meta["Edge_Heatmaps"].cuda()
                    heatmap_loss = torch.mean((x_hat-gt_heatmap)**2, axis=1).sum()
                    original_heatmap_loss += heatmap_loss.cpu()

                    if args.vit == "ORFormer":
                        _, x_hat, _, _, _, predicted_min_indices, OR_predicted_min_indices, _, _ = model(resized_occluded_input.to(device))
                        prediction = predicted_min_indices.permute(0, 2, 1).contiguous()
                        loss = criterion(prediction, target)
                        valid_entropy_loss += loss.cpu()

                        OR_prediction = OR_predicted_min_indices.permute(0, 2, 1).contiguous()
                        loss = criterion(OR_prediction, target)
                        valid_OR_entropy_loss += loss.cpu()

                        _, x_hat, _, _, _, predicted_min_indices, OR_predicted_min_indices, _, _ = model(resized_input.to(device))

                        heatmap_loss = torch.mean((x_hat-gt_heatmap)**2, axis=1).sum()
                        valid_heatmap_loss += heatmap_loss.cpu()
                    if args.vit == "vit":
                        _, x_hat, _, _, _, predicted_min_indices, OR_predicted_min_indices, _, _ = model(resized_occluded_input.to(device))
                        prediction = predicted_min_indices.permute(0, 2, 1).contiguous()
                        loss = criterion(prediction, target)
                        valid_entropy_loss += loss.cpu()

                        _, x_hat, _, _, _, predicted_min_indices, OR_predicted_min_indices, _, _ = model(resized_input.to(device))

                        heatmap_loss = torch.mean((x_hat-gt_heatmap)**2, axis=1).sum()
                        valid_heatmap_loss += heatmap_loss.cpu()
                    total += resized_input.shape[0]
                original_heatmap_loss = original_heatmap_loss/total
                valid_entropy_loss = valid_entropy_loss/total
                valid_OR_entropy_loss = valid_OR_entropy_loss/total
                valid_heatmap_loss = valid_heatmap_loss/total
                lr = np.float64(scheduler.get_last_lr()[0])
                if valid_heatmap_loss < best_heatmap_loss:
                    best_heatmap_loss = valid_heatmap_loss
                    save_model(model, resultDir)
                wandb.log({
                    "Train_Entropy_loss": train_entropy_loss, "Train_OR_Entropy_loss": train_OR_entropy_loss,
                    "Valid_Entropy_loss": valid_entropy_loss, "Valid_OR_Entropy_loss": valid_OR_entropy_loss, 
                    "Train_Heatmap_loss": train_heatmap_loss, "Valid_Heatmap_Loss": valid_heatmap_loss, 
                    "lr": lr, "Original_Heatmap_loss": original_heatmap_loss,
                    "Best_Heatmap_Loss": best_heatmap_loss})
        torch.cuda.empty_cache()
                
if __name__ == '__main__':
    main_function()
