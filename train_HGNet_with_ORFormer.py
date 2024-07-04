import sys
import os
os.chdir(sys.path[0]) 

import argparse

from Config import cfg

from Dataloader import WFLW_heatmap_Dataset, W300_heatmap_Dataset, COFW_heatmap_Dataset

from Dataloader import denorm_points

import torch
torch.cuda.empty_cache()

import numpy as np

import torchvision.transforms as transforms

import torch.nn as nn

from PIL import ImageDraw

from tqdm import tqdm

import wandb

import torchlm

from Model.StackedHGNet import IntergrationStackedHGNet

from Model.VQVAE import VQVAE

from Model.simple_vit import SimpleViT, ORFormer

class NME_loss(nn.Module):
    def __init__(self, name):
        super(NME_loss, self).__init__()
        self.name = name
    
    def forward(self, pred, gt):
        if self.name == 'WFLW':
            norm = torch.linalg.vector_norm(gt[:, 60, :] - gt[:, 72, :], dim=1)
        elif self.name == '300W':
            norm = torch.linalg.vector_norm(gt[:, 36, :] - gt[:, 45, :], dim=1)
        elif self.name == 'COFW':
            norm = torch.linalg.vector_norm(gt[:, 17, :] - gt[:, 16, :], dim=1)
        else:
            raise ValueError('Wrong Dataset')
        norm = norm[:, None]
        return torch.mean(torch.linalg.vector_norm(pred-gt, 2, dim=2) / norm, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--T_0', help='Cosine Annealing T_0', type=int, default=5)
    parser.add_argument('--T_mult', help='Cosine Annealing T_mult', type=int, default=2)
    parser.add_argument('--epoch', help='epoch number', type=int, default=300)
    parser.add_argument('--validEpoch', help='valid epoch number', type=int, default=1)
    parser.add_argument('--dataset', help='dataset', type=str, default="WFLW")
    parser.add_argument('--resultDir', help='result directory', type=str, default="/4TB/jcchiang/results/temp/WFLW")
    parser.add_argument('--name', help='name', type=str, default="")
    parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
    parser.add_argument('--nstack', help='nstack of HGNet', type=int, default=4)
    parser.add_argument('--ratio', help='ratio between image and feature map', type=int, default=4)
    parser.add_argument('--heatmap', help='fusion heatmap type', type=str, default="None")
    parser.add_argument('--alpha', help='learning rate balance term', type=float, default=0.05)

    args = parser.parse_args()

    return args


def calcuate_loss(name, pred, gt, trans):
    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'WFLW':
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name == '300W':
        norm = np.linalg.norm(gt[36, :] - gt[45, :])
    elif name == 'COFW':
        norm = np.linalg.norm(gt[16, :] - gt[17, :])
    else:
        raise ValueError('Wrong Dataset')
    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

    return error_real

def save_img(output, ground_truth, image, path, epoch, i):
    image = torch.squeeze(image)
    img = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(img)
    for j in range(output.shape[0]):
        draw.ellipse([output[j][0]-1, output[j][1]-1, output[j][0]+1, output[j][1]+1], fill=(255, 0, 0))
        draw.ellipse([ground_truth[j][0]-1, ground_truth[j][1]-1, ground_truth[j][0]+1, ground_truth[j][1]+1], fill=(0, 0, 255))
        draw.line([(output[j][0], output[j][1]), (ground_truth[j][0], ground_truth[j][1])], fill=(0, 255, 0), width=1)
    img.save(f"{path}/Epoch{epoch}_{i}.jpg")

def save_model(model, path):
    torch.save(model.state_dict(), f"{path}/best_model.pt") 

def main_function():

    args = parse_args()
    args.name += f"_lr{args.lr}_T0{args.T_0}_Tmult{args.T_mult}_epoch{args.epoch}_batch{args.batch_size}_alpha{args.alpha}"
    resultDir = os.path.join(args.resultDir, args.name)

    wandb.init(
        # set the wandb project where this run will be logged
        project = "ADNetV4_" + args.dataset,
        name = args.name,
        
        # track hyperparameters and run metadata
        config = {
            "learning_rate": args.lr,
            "T_mult": args.T_mult,
            "architecture": "HGNet+VQVAE",
            "dataset": args.dataset,
            "epochs": args.epoch,
        }
    )

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device("cuda")
    if args.dataset == "WFLW":
        model = IntergrationStackedHGNet(classes_num=[cfg.WFLW.NUM_POINT, cfg.WFLW.NUM_EDGE, cfg.WFLW.NUM_POINT], 
                               edge_info=cfg.WFLW.EDGE_INFO, nstack=args.nstack)
    if args.dataset == "300W":
        model = IntergrationStackedHGNet(classes_num=[cfg.W300.NUM_POINT, cfg.W300.NUM_EDGE, cfg.W300.NUM_POINT],
                               edge_info=cfg.W300.EDGE_INFO, nstack=args.nstack)
    if args.dataset == "COFW":
        model = IntergrationStackedHGNet(classes_num=[cfg.COFW.NUM_POINT, cfg.COFW.NUM_EDGE, cfg.COFW.NUM_POINT],
                               edge_info=cfg.COFW.EDGE_INFO, nstack=args.nstack)
    model = model.to(device)
            
    if args.heatmap == 'ORFormer':
        if args.dataset == "WFLW":
            vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
            vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.WFLW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
            vqvae_model.load_weights("weights/ORFormer/WFLW/best_model.pt")
        if args.dataset == "300W":
            vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
            vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.W300.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
            vqvae_model.load_weights("weights/ORFormer/300W/best_model.pt")
        if args.dataset == "COFW":
            vit_model = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
            vqvae_model = VQVAE(h_dim=128, res_h_dim=32, output_dim=cfg.COFW.NUM_EDGE, n_res_layers=2, n_embeddings=2048, embedding_dim=256,
                code_dim=256, beta=0.25, save_img_embedding_map=False, vit=vit_model)
            vqvae_model.load_weights("weights/ORFormer/COFW/best_model.pt")
        vqvae_model = vqvae_model.to(device)
        vqvae_model.eval()

    train_transform = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomMask(mask_ratio=0.1, prob=0.4),
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
            augmentation_transform = train_transform,
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            ratio = args.ratio
        )
        valid_dataset = WFLW_heatmap_Dataset(
            cfg, cfg.WFLW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            ratio = args.ratio
        )
    if args.dataset == "300W":
        train_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'train',
            augmentation_transform = train_transform,
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            mirror=True,
            ratio = args.ratio
        )
        valid_dataset = W300_heatmap_Dataset(
            cfg, cfg.W300.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            mirror=False,
            ratio = args.ratio
        )
    if args.dataset == "COFW":
        train_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'train',
            augmentation_transform = train_transform,
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            mirror=True,
            ratio = args.ratio
        )
        valid_dataset = COFW_heatmap_Dataset(
            cfg, cfg.COFW.ROOT,
            subset = 'test',
            normalize_transform = normalize_transform,
            edge_type="ADNet",
            mirror=False,
            ratio = args.ratio
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=0.001*args.lr)

    if not os.path.exists(resultDir):
        os.system(f"mkdir {resultDir}")

    criterion = NME_loss(args.dataset)

    best_nme = 100
    for epoch in range(args.epoch):
        print("Epoch: %d" % (epoch+1))
        # train
        model.train()
        train_edge_heatmap_loss = 0
        train_point_heatmap_loss = 0
        train_nme = 0
        total = 0
        error_list = []
        for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            if args.heatmap == 'None':
                y, landmarks = model(input.to(device))
            elif args.heatmap == 'ORFormer':
                with torch.no_grad():
                    _, reference_heatmaps, _, _, _, _, _, _, _ = vqvae_model(resized_input.to(device))
                y, landmarks = model(input.to(device), reference_heatmaps=reference_heatmaps)
            elif args.heatmap == 'GT':
                y, landmarks = model(input.to(device), reference_heatmaps=meta["Edge_Heatmaps"].to(device))

            gt_edge_heatmap = meta["Edge_Heatmaps"].to(device)
            gt_point_heatmap = meta["Point_Heatmaps"].to(device)
            targets = meta['Landmarks'].to(device)
            
            total_loss = 0
            for i in range(args.nstack):
                predicted_landmarks = y[3*i]
                nme_loss = criterion(predicted_landmarks, targets).sum()
                train_nme += nme_loss.cpu() / args.nstack

                predicted_edge_heatmap = y[3*i+1]
                edge_heatmap_loss = torch.mean((predicted_edge_heatmap - gt_edge_heatmap)**2, axis=1).sum()
                train_edge_heatmap_loss += edge_heatmap_loss.cpu() / args.nstack

                predicted_point_heatmap = y[3*i+2]
                point_heatmap_loss = torch.mean((predicted_point_heatmap - gt_point_heatmap)**2, axis=1).sum()
                train_point_heatmap_loss += point_heatmap_loss.cpu() / args.nstack
            
                total_loss += nme_loss + args.alpha * (edge_heatmap_loss+point_heatmap_loss)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step(epoch+i/len(train_loader))
            total += input.shape[0]
            
        train_edge_heatmap_loss = train_edge_heatmap_loss / total
        train_point_heatmap_loss = train_point_heatmap_loss / total
        train_nme = train_nme / total * 100

        # test
        if (epoch+1) % args.validEpoch == 0:
            model.eval()
            test_edge_heatmap_loss = 0
            test_point_heatmap_loss = 0
            total = 0
            error_list = []
            with torch.no_grad():
                for i, (input, resized_input, resized_occluded_input, meta, image, resized_image) in enumerate(tqdm(valid_loader)):
                    if args.heatmap == 'None':
                        y, landmarks = model(input.to(device))
                    elif args.heatmap == 'VQVAE':
                        _, reference_heatmaps, _, _, _, _, _, _, _ = vqvae_model(resized_input.to(device))
                        y, landmarks = model(input.to(device), reference_heatmaps=reference_heatmaps)
                    elif args.heatmap == 'CodeFormer':
                        _, reference_heatmaps, _, _, _, _, _, _, _ = vqvae_model(resized_input.to(device))
                        y, landmarks = model(input.to(device), reference_heatmaps=reference_heatmaps)
                    elif args.heatmap == 'ORFormer':
                        _, reference_heatmaps, _, _, _, _, _, _, _ = vqvae_model(resized_input.to(device))
                        y, landmarks = model(input.to(device), reference_heatmaps=reference_heatmaps)
                    elif args.heatmap == 'GT':
                        y, landmarks = model(input.to(device), reference_heatmaps=meta["Edge_Heatmaps"].to(device))

                    gt_edge_heatmap = meta["Edge_Heatmaps"]
                    gt_point_heatmap = meta["Point_Heatmaps"]
                    
                    landmarks = denorm_points(landmarks, 64, 64)[0].cpu().numpy()
                    landmarks = landmarks * args.ratio

                    Annotated_Points = meta['Annotated_Points'].numpy()[0]
                    Trans = meta['trans'].numpy()[0]
                    error = calcuate_loss(args.dataset, landmarks, Annotated_Points, Trans)
                    error_list.append(error)
                    
                    # test_heatmap_loss += heatmap_loss.cpu()
                    for i in range(args.nstack):
                        predicted_edge_heatmap = y[3*i+1].detach().cpu()
                        edge_heatmap_loss = torch.mean((predicted_edge_heatmap - gt_edge_heatmap)**2, axis=1).sum()
                        test_edge_heatmap_loss += edge_heatmap_loss.cpu() / args.nstack

                        predicted_point_heatmap = y[3*i+2].detach().cpu()
                        point_heatmap_loss = torch.mean((predicted_point_heatmap - gt_point_heatmap)**2, axis=1).sum()
                        test_point_heatmap_loss += point_heatmap_loss.cpu() / args.nstack
                        
                    total += input.shape[0]
                test_edge_heatmap_loss = test_edge_heatmap_loss / total
                test_point_heatmap_loss = test_point_heatmap_loss / total
                test_nme = np.mean(np.array(error_list)) * 100
                lr = np.float64(scheduler.get_last_lr()[0])
                # save model
                if test_nme < best_nme:
                    best_nme = test_nme
                    save_model(model, resultDir)
            wandb.log({"Best_NME": best_nme, 
                       "Test_NME": test_nme, "Test_Edge_Heatmap_Loss": test_edge_heatmap_loss, "Test_Point_Heatmap_Loss": test_point_heatmap_loss,
                       "Train_NME": train_nme, "Train_Edge_Heatmap_Loss": train_edge_heatmap_loss, "Train_Point_Heatmap_Loss": train_point_heatmap_loss,
                       "Learning_Rate": lr})
    print(f"Finish with best nme {best_nme}")


if __name__ == '__main__':
    main_function()

