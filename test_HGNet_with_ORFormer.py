import os
import os.path
from os import path
import sys
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.chdir(sys.path[0])

import argparse

from Config import cfg

from Dataloader import WFLW_heatmap_Dataset, W300_heatmap_Dataset, COFW_heatmap_Dataset, denorm_points

torch.cuda.empty_cache()
import numpy as np

import torchvision.transforms as transforms

import torch.nn as nn

from PIL import ImageDraw, ImageFont

from tqdm import tqdm

from Model.StackedHGNet import IntergrationStackedHGNet

from Model.VQVAE import VQVAE

from Model.simple_vit import SimpleViT, ORFormer

from cv2 import cv2

from scipy.integrate import simps

import torchinfo

def parse_args():
    parser = argparse.ArgumentParser(description='Test HGNet')
    parser.add_argument('--dataset', help='dataset', type=str, default="WFLW")
    parser.add_argument('--nstack', help='nstack', type=int, default=4)

    args = parser.parse_args()

    return args

def calcuate_nmes(name, pred, gt, trans):
    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'WFLW':
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name == '300W':
        norm = np.linalg.norm(gt[36, :] - gt[45, :])
    elif name == 'COFW':
        norm = np.linalg.norm(gt[16, :] - gt[17, :])
    else:
        raise ValueError('Wrong Dataset')
    nmes = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return nmes

def calculate_fr_auc(nmes, thres=None, step=0.0001):
    num_data = len(nmes) 
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return [round(fr, 4), round(auc, 6)]

def save_img(output, ground_truth, image, path, i, error, ours):
    image = torch.squeeze(image).permute(2, 0, 1)
    img = transforms.ToPILImage()(image)
    img = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(img)
    for j in range(output.shape[0]):
        draw.ellipse([output[j][0]-1.5, output[j][1]-1.5, output[j][0]+1.5, output[j][1]+1.5], fill=(255, 0, 0))
        draw.ellipse([ground_truth[j][0]-1, ground_truth[j][1]-1, ground_truth[j][0]+1, ground_truth[j][1]+1], fill=(0, 0, 255))
        draw.line([(output[j][0], output[j][1]), (ground_truth[j][0], ground_truth[j][1])], fill=(0, 255, 0), width=1)
    if ours:
        img.save(f"{path}/img{i}_landmarks_ours.jpg")
    else:
        img.save(f"{path}/img{i}_landmarks_baseline.jpg")

def save_img_with_face(predicted_image, gt_image, image, path, epoch, i):
    image = torch.squeeze(image).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    heatmap_img = cv2.applyColorMap((predicted_image*255).cpu().numpy().astype(np.uint8), cv2.COLORMAP_BONE)
    img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
    cv2.imwrite(f"{path}/Epoch{epoch}_{i}_pre_with_face.jpg", img)

    heatmap_img = cv2.applyColorMap((gt_image*255).cpu().numpy().astype(np.uint8), cv2.COLORMAP_BONE)
    img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
    cv2.imwrite(f"{path}/Epoch{epoch}_{i}_gt_with_face.jpg", img)


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

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device("cuda")
    if args.dataset == "WFLW":
        model = IntergrationStackedHGNet(classes_num=[cfg.WFLW.NUM_POINT, cfg.WFLW.NUM_EDGE, cfg.WFLW.NUM_POINT], 
                               edge_info=cfg.WFLW.EDGE_INFO, nstack=args.nstack)
        model.load_weights('weights/HGNet/WFLW/best_model.pt')
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.WFLW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
            code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
    if args.dataset == "300W":
        model = IntergrationStackedHGNet(classes_num=[cfg.W300.NUM_POINT, cfg.W300.NUM_EDGE, cfg.W300.NUM_POINT],
                               edge_info=cfg.W300.EDGE_INFO, nstack=args.nstack)
        model.load_weights('weights/HGNet/300W/best_model.pt')
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.W300.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
            code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        vqvae_model.load_weights('weights/ORFormer/300W/best_model.pt')
    if args.dataset == "COFW":
        model = IntergrationStackedHGNet(classes_num=[cfg.COFW.NUM_POINT, cfg.COFW.NUM_EDGE, cfg.COFW.NUM_POINT],
                               edge_info=cfg.COFW.EDGE_INFO, nstack=args.nstack)
        model.load_weights('weights/HGNet/COFW/best_model.pt')
        vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
        vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.COFW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
            code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
        vqvae_model.load_weights('weights/ORFormer/COFW/best_model.pt')
    
    model = model.to(device)
    model.eval()
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()
    torchinfo.summary(model, (1, 3, 256, 256))

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])

    if args.dataset == "WFLW":
        valid_dataset = WFLW_heatmap_Dataset(
            cfg, cfg.WFLW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="ADNet"
        )
    if args.dataset == "300W":
        valid_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            mirror=False
        )
    if args.dataset == "COFW":
        valid_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            mirror=False
        )
        
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    error_list = []
    if args.dataset == "WFLW":
        baseline_error_list = []
        non_error_list = []
        pose_error_list = []
        expression_error_list = []
        illumination_error_list = []
        makeup_error_list = []
        occlusion_error_list = []
        blur_error_list = []
    if args.dataset == "300W":
        common_error_list = []
        challenging_error_list = []
    with torch.no_grad():
        for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(valid_loader)):
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]

            _, reference_heatmaps, _, _, _, _, _, _, _ = vqvae_model(resized_input.to(device))
            y, landmarks = model(input.to(device), reference_heatmaps=reference_heatmaps)
            landmarks = denorm_points(landmarks, 64, 64)[0].cpu().numpy()
            landmarks = landmarks * 4
            error = calcuate_nmes(args.dataset, landmarks, Annotated_Points, Trans)
            error_list.append(error)

            ground_truth = meta['Landmarks']
            ground_truth = denorm_points(ground_truth, 64, 64)[0].cpu().numpy()
            ground_truth = ground_truth * 4

            if args.dataset == "WFLW":
                if meta['Pose'][0] == '1':
                    pose_error_list.append(error)
                if meta['Expression'][0] == '1':
                    expression_error_list.append(error)
                if meta['Illumination'][0] == '1':
                    illumination_error_list.append(error)
                if meta['Makeup'][0] == '1':
                    makeup_error_list.append(error)
                if meta['Occlusion'][0] == '1':
                    occlusion_error_list.append(error)
                if meta['Blur'][0] == '1':
                    blur_error_list.append(error)
                if meta['Pose'][0] == '0' and meta['Expression'][0] == '0' and meta['Illumination'][0] == '0' and meta['Makeup'][0] == '0' and meta['Occlusion'][0] == '0' and meta['Blur'][0] == '0':
                    non_error_list.append(error)
            if args.dataset == "300W":
                if meta['Challenge'][0] == True:
                    challenging_error_list.append(error)
                elif meta['Challenge'][0] == False:
                    common_error_list.append(error)
    fr, auc = calculate_fr_auc(error_list, 0.1)
    print(f"FR: {fr}, AUC: {auc}")
    print("NME:")
    print(f"Full ({len(error_list)}): {np.mean(np.array(error_list)) * 100:.2f}")
    if args.dataset == 'WFLW':
        print(f"Pose ({len(pose_error_list)}): {np.mean(np.array(pose_error_list)) * 100:.2f}")
        print(f"Expression ({len(expression_error_list)}): {np.mean(np.array(expression_error_list)) * 100:.2f}")
        print(f"Illumination ({len(illumination_error_list)}): {np.mean(np.array(illumination_error_list)) * 100:.2f}")
        print(f"Makeup ({len(makeup_error_list)}): {np.mean(np.array(makeup_error_list)) * 100:.2f}")
        print(f"Occlusion ({len(occlusion_error_list)}): {np.mean(np.array(occlusion_error_list)) * 100:.2f}")
        print(f"Blur ({len(blur_error_list)}): {np.mean(np.array(blur_error_list)) * 100:.2f}")
    if args.dataset == '300W':
        print(f"Common ({len(common_error_list)}): {np.mean(np.array(common_error_list)) * 100:.2f}")
        print(f"Challenge ({len(challenging_error_list)}): {np.mean(np.array(challenging_error_list)) * 100:.2f}")


if __name__ == '__main__':
    main_function()
