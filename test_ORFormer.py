import os
import os.path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.chdir(sys.path[0])

import argparse

from Config import cfg

from Dataloader import WFLW_heatmap_Dataset, W300_heatmap_Dataset, COFW_heatmap_Dataset

import numpy as np

import torch
import torchvision.transforms as transforms

from tqdm import tqdm


from Model.VQVAE import VQVAE

from Model.simple_vit import SimpleViT, ORFormer

from cv2 import cv2

import torchinfo

def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')
    parser.add_argument('--dataset', help='dataset', type=str, default="WFLW")
    args = parser.parse_args()

    return args


def calcuate_loss(name, pred, gt, trans):
    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'WFLW':
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name == '300W':
        norm = np.linalg.norm(gt[36, :] - gt[45, :])
    elif name == 'COFW':
        norm = np.linalg.norm(gt[17, :] - gt[16, :])
    else:
        raise ValueError('Wrong Dataset')
    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return error_real

def save_img(predicted_image, gt_image, image, path, i, edge, method):
    image = cv2.resize(torch.squeeze(image).cpu().numpy().astype(np.uint8), (64, 64))
    image = torch.squeeze(gt_image/gt_image.max())
    img = transforms.ToPILImage()(image)
    img.save(f"{path}/img.jpg")

def save_model(model, path):
    torch.save(model.state_dict(), f"{path}/best_model.pt") 

def main_function():

    args = parse_args()

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device('cuda')

    if args.dataset == "WFLW":
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        vqvae = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.WFLW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
            code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        vqvae.load_weights('weights/ORFormer/WFLW/best_model.pt')
    if args.dataset == "300W":
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        vqvae = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.W300.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
            code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        vqvae.load_weights("weights/VQVAE/300W/best_model.pt")
        vqvae = vqvae.to(device) 
    if args.dataset == "COFW":
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        vqvae = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.COFW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
            code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        vqvae.load_weights('weights/ORFormer/COFW/best_model.pt')
        vqvae = vqvae.to(device)

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])

    if args.dataset == "WFLW":
        valid_dataset = WFLW_heatmap_Dataset(
            cfg, cfg.WFLW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="VQVAE"
        )
    if args.dataset == "300W":
        valid_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            mirror=False,
            edge_type="VQVAE"
        )
    if args.dataset == "COFW":
        valid_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="VQVAE",
            mirror=False
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    vqvae.eval()
    torchinfo.summary(vqvae, (1, 3, 64, 64))
    with torch.no_grad():
        valid_heatmap_loss = 0
        pose_valid_heatmap_loss = 0
        expression_valid_heatmap_loss = 0
        illumination_valid_heatmap_loss = 0
        makeup_valid_heatmap_loss = 0
        occlusion_valid_heatmap_loss = 0
        blur_valid_heatmap_loss = 0
        total = 0
        pose_total = 0
        expression_total = 0
        illumination_total = 0
        makeup_total = 0
        occlusion_total = 0
        blur_total = 0
        for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(valid_loader)):
            gt_heatmap = meta["Edge_Heatmaps"].cuda()
            _, x_hat, _, _, _ , _, _, _, attention_weights = vqvae(resized_occluded_input.to(device))

            heatmap_loss = torch.mean((x_hat-gt_heatmap)**2, axis=1).sum().cpu()
            valid_heatmap_loss += heatmap_loss
            total += input.shape[0]
            if args.dataset == "WFLW":
                if meta['Pose'][0] == '1':
                    pose_valid_heatmap_loss += heatmap_loss.cpu()
                    pose_total += input.shape[0]
                if meta['Expression'][0] == '1':
                    expression_valid_heatmap_loss += heatmap_loss.cpu()
                    expression_total += input.shape[0]
                if meta['Illumination'][0] == '1':
                    illumination_valid_heatmap_loss += heatmap_loss.cpu()
                    illumination_total += input.shape[0]
                if meta['Makeup'][0] == '1':
                    makeup_valid_heatmap_loss += heatmap_loss.cpu()
                    makeup_total += input.shape[0]
                if meta['Occlusion'][0] == '1':
                    occlusion_valid_heatmap_loss += heatmap_loss.cpu()
                    occlusion_total += input.shape[0]
                if meta['Blur'][0] == '1':
                    blur_valid_heatmap_loss += heatmap_loss.cpu()
                    blur_total += input.shape[0]
                
    print(f"Full ({total}): {valid_heatmap_loss/total:.2f}")
    if args.dataset == "WFLW":
        print(f"Pose ({pose_total}): {pose_valid_heatmap_loss/pose_total:.2f}")
        print(f"Expression ({expression_total}): {expression_valid_heatmap_loss/expression_total:.2f}")
        print(f"Illumination ({illumination_total}): {illumination_valid_heatmap_loss/illumination_total:.2f}")
        print(f"Makeup ({makeup_total}): {makeup_valid_heatmap_loss/makeup_total:.2f}")
        print(f"Occlusion ({occlusion_total}): {occlusion_valid_heatmap_loss/occlusion_total:.2f}")
        print(f"Blur ({blur_total}): {blur_valid_heatmap_loss/blur_total:.2f}")
    

if __name__ == '__main__':
    main_function()