import cv2, copy, logging, os
import numpy as np

import utils

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from scipy import interpolate
import csv


import json

logger = logging.getLogger(__name__)

def circle_VQVAE(img, pt, color=255, interpolate_mode=cv2.INTER_AREA, scale=1):
    h, w = img.shape
    img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
    pt_scale = (pt * scale + 0.5).astype(np.int32)
    img_scale[pt_scale[1], pt_scale[0]] = color
    img_scale = -1*img_scale + 255 #Flip white and black
    distance_map = cv2.distanceTransform(img_scale.astype(np.uint8), cv2.DIST_L2, 0)
    std = np.std(distance_map)
    threshold = std * 3
    heatmap = np.exp(-1 * np.square(distance_map) / (2 * np.square(std)))
    heatmap[distance_map >= threshold] = 0
    heatmap = cv2.resize(heatmap, (w, h), interpolation=interpolate_mode)
    return heatmap

def circle_ADNet(img, pt, sigma=1.0, label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] > img.shape[1] - 1 or ul[1] > img.shape[0] - 1 or
            br[0]-1 < 0 or br[1]-1 < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = 255 * g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def polylines_VQVAE(img, lmks, is_closed, color=255, thickness=1, draw_mode=cv2.LINE_AA, interpolate_mode=cv2.INTER_AREA, scale=1):
    h, w = img.shape
    img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
    lmks_scale = (lmks * scale + 0.5).astype(np.int32)
    img_scale = cv2.polylines(img_scale, [lmks_scale], is_closed, color, thickness * scale, draw_mode)
    img_scale = -1*img_scale + 255 #Flip white and black
    distance_map = cv2.distanceTransform(img_scale.astype(np.uint8), cv2.DIST_L2, 0)
    std = np.std(distance_map)
    threshold = std * 3
    heatmap = np.exp(-1 * np.square(distance_map) / (2 * np.square(std)))
    heatmap[distance_map >= threshold] = 0
    heatmap = cv2.resize(heatmap, (w, h), interpolation=interpolate_mode)
    return heatmap

def polylines_ADNet(img, lmks, is_closed, color=255, thickness=1, draw_mode=cv2.LINE_AA, interpolate_mode=cv2.INTER_AREA, scale=4):
    h, w = img.shape
    img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
    lmks_scale = (lmks * scale + 0.5).astype(np.int32)
    cv2.polylines(img_scale, [lmks_scale], is_closed, color, thickness * scale, draw_mode)
    img = cv2.resize(img_scale, (w, h), interpolation=interpolate_mode)
    return img

def fit_curve(lmks, is_closed=False, density=5):
    try:
        x = lmks[:,0].copy()
        y = lmks[:,1].copy()
        if is_closed:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        tck, u = interpolate.splprep([x, y], s=0, per=is_closed, k=3)
        #bins = (x.shape[0] - 1) * density + 1
        #lmk_x, lmk_y = interpolate.splev(np.linspace(0, 1, bins), f)
        intervals = np.array([])
        for i in range(len(u)-1):
            intervals = np.concatenate((intervals, np.linspace(u[i], u[i+1], density, endpoint=False)))
        if not is_closed:
            intervals = np.concatenate((intervals, [u[-1]]))
        lmk_x, lmk_y = interpolate.splev(intervals, tck, der=0)
        #der_x, der_y = interpolate.splev(intervals, tck, der=1)
        curve_lmks = np.stack([lmk_x, lmk_y], axis=-1)
        #curve_ders = np.stack([der_x, der_y], axis=-1)
        #origin_indices = np.arange(0, curve_lmks.shape[0], density)
        
        return curve_lmks
    except:
        return lmks

def norm_points(points, h, w, align_corners=True):
    if align_corners:
        # [0, SIZE-1] -> [-1, +1]
        des_points = points / torch.tensor([w-1, h-1]).to(points).view(1, 2) * 2 - 1
    else:
        # [-0.5, SIZE-0.5] -> [-1, +1]
        des_points = (points * 2 + 1) / torch.tensor([w, h]).to(points).view(1, 2) - 1
    des_points = torch.clamp(des_points, -1, 1)
    return des_points

def denorm_points(points, h, w, align_corners=True):
    if align_corners:
        # [-1, +1] -> [0, SIZE-1]
        des_points = (points + 1) / 2 * torch.tensor([w-1, h-1]).to(points).view(1, 1, 2)
    else:
        # [-1, +1] -> [-0.5, SIZE-0.5]
        des_points = ((points + 1) * torch.tensor([w, h]).to(points).view(1, 1, 2) - 1) / 2
    return des_points

class WFLW_heatmap_Dataset(Dataset):
    def __init__(self, cfg, root, subset=None, augmentation_transform=None, normalize_transform=None, mask_transform=None, edge_type="VQVAE", ratio=4):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.root = root
        self.number_landmarks = cfg.WFLW.NUM_POINT
        self.Fraction = cfg.WFLW.FRACTION

        self.subset = subset
        self.augmentation_transform = augmentation_transform
        self.normalize_transform = normalize_transform
        self.mask_transform = mask_transform
        self.edge_type = edge_type
        self.ratio = ratio

        # Path to dataset
        if self.subset == 'train':
            self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                'list_98pt_rect_attr_train.txt')
        elif self.subset == 'test':
            self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                'list_98pt_rect_attr_test.txt')
        else:
            self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_test', 
                                                f'list_98pt_test_{subset}.txt')
 
        self.database = self.get_file_information()
        self.edge_info = cfg.WFLW.EDGE_INFO
        self.flip_mapping = cfg.WFLW.FLIP_MAPPING

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()
        for i in range(len(info_list)):
            temp_info = info_list[i]
            temp_point = []
            temp_info = temp_info.split(' ')
            # if temp_info[204] != '1':
            #     continue
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, 'WFLW_images', temp_info[-1])
            Data_base.append({'Img': temp_name,
                            'bbox': temp_box,
                            'point': point_coord,
                            'pose': temp_info[200],
                            'expression': temp_info[201],
                            'illumination': temp_info[202],
                            'makeup': temp_info[203],
                            'occlusion': temp_info[204],
                            'blur': temp_info[205]})
        return Data_base

    def __len__(self):
        return len(self.database)
        
    def generate_pointmap(self, points, scale=0.25, sigma=1.5):
        h, w = self.Image_size, self.Image_size
        pointmaps = []
        for i in range(len(points)):
            pointmap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            point = copy.deepcopy(points[i])
            point[0] = max(0, min(w-1, point[0]))
            point[1] = max(0, min(h-1, point[1]))
            pointmap = circle_ADNet(pointmap, point, sigma=sigma) / 255.0
            
            pointmaps.append(pointmap)
        pointmaps = np.stack(pointmaps, axis=0)
        pointmaps = torch.from_numpy(pointmaps).float().unsqueeze(0)
        pointmaps = F.interpolate(pointmaps, size=(int(w * scale), int(h * scale)), mode='bilinear', align_corners=False).squeeze()
        return pointmaps

    def generate_edgemap(self, points, scale=0.25, thickness=1):
        h, w = self.Image_size, self.Image_size
        edgemaps = []
        for is_closed, indices in self.edge_info:
            edgemap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            part = copy.deepcopy(points[np.array(indices)])
            
            part = fit_curve(part, is_closed)
            part[:, 0] = np.clip(part[:, 0], 0, w-1)
            part[:, 1] = np.clip(part[:, 1], 0, h-1)
            if self.edge_type == "VQVAE":
                edgemap = polylines_VQVAE(edgemap, part, is_closed, 255, thickness)
            elif self.edge_type == "ADNet":
                edgemap = polylines_ADNet(edgemap, part, is_closed, 255, thickness) / 255.0
            #offset = 0.5
            #part = (part + offset).astype(np.int32)
            #part[:, 0] = np.clip(part[:, 0], 0, w-1)
            #part[:, 1] = np.clip(part[:, 1], 0, h-1)
            #cv2.polylines(edgemap, [part], is_closed, 255, thickness, cv2.LINE_AA)
            
            edgemaps.append(edgemap)
        edgemaps = np.stack(edgemaps, axis=0)
        edgemaps = torch.from_numpy(edgemaps).float().unsqueeze(0)
        edgemaps = F.interpolate(edgemaps, size=(int(w * scale), int(h * scale)), mode='bilinear', align_corners=False).squeeze()
        return edgemaps
        
    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])

        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        
        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        
        if self.subset == "train":
            prob = np.random.uniform(0, 1)
            if prob > 0.5:
                width = Img.shape[1]-1
                temp_box = [width-BBox[2]-BBox[0], BBox[1], BBox[2], BBox[3]]
                Img = cv2.flip(Img, 1)
                Points_flip = Points.copy()
                for (q, p) in self.flip_mapping:
                    Points_flip[p] = Points[q]
                    Points_flip[q] = Points[p]
                Points_flip[:, 0] = width - Points_flip[:, 0]
                Points = Points_flip
                BBox = temp_box

        Annotated_Points = Points.copy()

        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

        if self.augmentation_transform is not None:
            input, Points = self.augmentation_transform(input, Points)

        image = copy.deepcopy(input)
        occluded_input = copy.deepcopy(input)

        if self.mask_transform is not None:
            occluded_input, _ = self.mask_transform(occluded_input, Points)

        resized_Points = Points / self.ratio
        landmarks = norm_points(torch.from_numpy(resized_Points), 64, 64)
        
        edgemap = self.generate_edgemap(Points)
        pointmap = self.generate_pointmap(Points)

        resized_input = cv2.resize(input, (64, 64), interpolation=cv2.INTER_LINEAR)
        resized_occluded_input = cv2.resize(occluded_input, (64, 64), interpolation=cv2.INTER_LINEAR)
        resized_image = copy.deepcopy(resized_occluded_input)
        
        if self.normalize_transform is not None:
            # input, _ = self.normalize_transform(input, Points)
            # resized_input, _ = self.normalize_transform(resized_input, resized_Points)
            # resized_occluded_input, _ = self.normalize_transform(resized_occluded_input, resized_Points)
            input = self.normalize_transform(input)
            resized_input = self.normalize_transform(resized_input)
            resized_occluded_input = self.normalize_transform(resized_occluded_input)

        meta = {
            'Annotated_Points': Annotated_Points,
            'Img_path': Img_path,
            'Points': Points,
            'Landmarks': landmarks,
            'Edge_Heatmaps': edgemap,
            'Point_Heatmaps': pointmap,
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
            'Pose': db_slic['pose'],
            'Expression': db_slic['expression'],
            'Illumination': db_slic['illumination'],
            'Makeup': db_slic['makeup'],
            'Occlusion': db_slic['occlusion'],
            'Blur': db_slic['blur']
        }
        return input.float(), resized_input.float(), resized_occluded_input.float(), meta, image, resized_image

class W300_heatmap_Dataset(Dataset):
    def __init__(self, cfg, root, subset=None, mirror=True, augmentation_transform=None, normalize_transform=None, mask_transform=None, edge_type="VQVAE", ratio=4):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.root = root
        self.number_landmarks = cfg.W300.NUM_POINT
        self.Fraction = cfg.W300.FRACTION
        self.mirror = mirror

        self.subset = subset
        self.augmentation_transform = augmentation_transform
        self.normalize_transform = normalize_transform
        self.mask_transform = mask_transform
        self.edge_type = edge_type
        self.ratio = ratio

        if self.subset == 'train':
            # self.annotation_file = os.path.join(root, 'face_landmarks_300w_train.json')
            self.annotation_file = os.path.join(root, 'train.tsv')
        elif self.subset == 'test':
            # self.annotation_file = os.path.join(root, 'face_landmarks_300w_valid.json')
            self.annotation_file = os.path.join(root, 'test.tsv')
 
        self.database = self.get_file_information()

        self.edge_info = cfg.W300.EDGE_INFO
        self.flip_mapping = cfg.W300.FLIP_MAPPING

    def get_file_information(self):
        Data_base = []
        with open(self.annotation_file, "r") as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            for row in tsv_reader:
                temp_name = str(row[0])
                temp_name = temp_name.replace("./rawImages", self.root)
                point_coord = np.array([float(x) for x in row[2].split(',')]).reshape(self.number_landmarks, 2)
                max_index = np.max(point_coord, axis=0)
                min_index = np.min(point_coord, axis=0)
                temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                max_index[1] - min_index[1]])  
                Data_base.append({'Img': temp_name, 'bbox': temp_box, 'point': point_coord})
                if self.mirror:
                    if ".jpg" in temp_name:
                        mirror_name = temp_name.replace(".jpg", "_mirror.jpg")
                    elif ".png" in temp_name:
                        mirror_name = temp_name.replace(".png", "_mirror.jpg")
                    Data_base.append({'Img': mirror_name, 'bbox': temp_box, 'point': point_coord})  

        return Data_base


    def __len__(self):
        return len(self.database)
    
    def generate_pointmap(self, points, scale=0.25, sigma=1.5):
        h, w = self.Image_size, self.Image_size
        pointmaps = []
        for i in range(len(points)):
            pointmap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            point = copy.deepcopy(points[i])
            point[0] = max(0, min(w-1, point[0]))
            point[1] = max(0, min(h-1, point[1]))
            pointmap = circle_ADNet(pointmap, point, sigma=sigma) / 255.0
            
            pointmaps.append(pointmap)
        pointmaps = np.stack(pointmaps, axis=0)
        pointmaps = torch.from_numpy(pointmaps).float().unsqueeze(0)
        pointmaps = F.interpolate(pointmaps, size=(int(w * scale), int(h * scale)), mode='bilinear', align_corners=False).squeeze()
        return pointmaps

    def generate_edgemap(self, points, scale=0.25, thickness=1):
        h, w = self.Image_size, self.Image_size
        edgemaps = []
        for is_closed, indices in self.edge_info:
            edgemap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            part = copy.deepcopy(points[np.array(indices)])
            
            part = fit_curve(part, is_closed)
            part[:, 0] = np.clip(part[:, 0], 0, w-1)
            part[:, 1] = np.clip(part[:, 1], 0, h-1)
            if self.edge_type == "VQVAE":
                edgemap = polylines_VQVAE(edgemap, part, is_closed, 255, thickness)
            elif self.edge_type == "ADNet":
                edgemap = polylines_ADNet(edgemap, part, is_closed, 255, thickness) / 255.0
            #offset = 0.5
            #part = (part + offset).astype(np.int32)
            #part[:, 0] = np.clip(part[:, 0], 0, w-1)
            #part[:, 1] = np.clip(part[:, 1], 0, h-1)
            #cv2.polylines(edgemap, [part], is_closed, 255, thickness, cv2.LINE_AA)
            
            edgemaps.append(edgemap)
        edgemaps = np.stack(edgemaps, axis=0)
        edgemaps = torch.from_numpy(edgemaps).float().unsqueeze(0)
        edgemaps = F.interpolate(edgemaps, size=(int(w * scale), int(h * scale)), mode='bilinear', align_corners=False).squeeze()
        return edgemaps
        
    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])

        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if self.mirror:
            if "mirror" in Img_path:
                width = Img.shape[1]-1
                temp_box = [width-BBox[2]-BBox[0], BBox[1], BBox[2], BBox[3]]
                for xy in Points:
                    xy[0] = width - xy[0]
                Points_flip = Points.copy()
                for (q, p) in self.flip_mapping:
                    Points_flip[p] = Points[q]
                    Points_flip[q] = Points[p]
                Points = Points_flip
                BBox = temp_box

        Annotated_Points = Points.copy()

        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

        if self.augmentation_transform is not None:
            input, Points = self.augmentation_transform(input, Points)

        image = copy.deepcopy(input)
        occluded_input = copy.deepcopy(input)

        if self.mask_transform is not None:
            occluded_input, _ = self.mask_transform(occluded_input, Points)

        resized_Points = Points / self.ratio
        landmarks = norm_points(torch.from_numpy(resized_Points), 64, 64)
        
        edgemap = self.generate_edgemap(Points)
        pointmap = self.generate_pointmap(Points)

        resized_input = cv2.resize(input, (64, 64), interpolation=cv2.INTER_LINEAR)
        resized_occluded_input = cv2.resize(occluded_input, (64, 64), interpolation=cv2.INTER_LINEAR)
        resized_image = copy.deepcopy(resized_occluded_input)
        
        if self.normalize_transform is not None:
            # input, _ = self.normalize_transform(input, Points)
            # resized_input, _ = self.normalize_transform(resized_input, resized_Points)
            # resized_occluded_input, _ = self.normalize_transform(resized_occluded_input, resized_Points)
            input = self.normalize_transform(input)
            resized_input = self.normalize_transform(resized_input)
            resized_occluded_input = self.normalize_transform(resized_occluded_input)

        if 'ibug' in Img_path:
            challenge_label = True
        else:
            challenge_label = False
        meta = {
            'Annotated_Points': Annotated_Points,
            'Img_path': Img_path,
            'Points': Points,
            'Landmarks': landmarks,
            'Edge_Heatmaps': edgemap,
            'Point_Heatmaps': pointmap,
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
            'Challenge': challenge_label
        }
        return input.float(), resized_input.float(), resized_occluded_input.float(), meta, image, resized_image
    
class COFW_heatmap_Dataset(Dataset):
    def __init__(self, cfg, root, subset=None, mirror=True, augmentation_transform=None, normalize_transform=None, mask_transform=None, edge_type="VQVAE", ratio=4):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.root = root
        self.number_landmarks = cfg.COFW.NUM_POINT
        self.Fraction = cfg.COFW.FRACTION

        self.mirror = mirror
        self.subset = subset
        self.augmentation_transform = augmentation_transform
        self.normalize_transform = normalize_transform
        self.mask_transform = mask_transform
        self.edge_type = edge_type
        self.ratio = ratio

        if self.subset == 'train':
            self.annotation_file = os.path.join(root, 'annotations', 'cofw_train.json')
        elif self.subset == 'test':
            self.annotation_file = os.path.join(root, 'annotations', 'cofw_test.json')

        self.database = self.get_file_information()

        self.edge_info = cfg.COFW.EDGE_INFO 
        self.flip_mapping = cfg.COFW.FLIP_MAPPING

    def get_file_information(self):
        Data_base = []
        f = open(self.annotation_file, "r")
        data = json.loads(f.read())
        for idx in range(len(data["images"])):
            assert data["images"][idx]["id"] == data["annotations"][idx]["image_id"]
            temp_name = os.path.join(self.root, 'images', data["images"][idx]["file_name"])
            temp_points = data["annotations"][idx]["keypoints"]
            point_coord = np.array(temp_points, dtype=np.float).reshape(self.number_landmarks, 3)
            point_coord = point_coord[:, :2]
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            if self.mirror:
                if ".jpg" in temp_name:
                    mirror_name = temp_name.replace(".jpg", "_mirror.jpg")
                elif ".png" in temp_name:
                    mirror_name = temp_name.replace(".png", "_mirror.jpg")
                Data_base.append({'Img': mirror_name, 'bbox': temp_box, 'point': point_coord})
            Data_base.append({'Img': temp_name, 'bbox': temp_box, 'point': point_coord})

        return Data_base

    def __len__(self):
        return len(self.database)
    
    def generate_pointmap(self, points, scale=0.25, sigma=1.5):
        h, w = self.Image_size, self.Image_size
        pointmaps = []
        for i in range(len(points)):
            pointmap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            point = copy.deepcopy(points[i])
            point[0] = max(0, min(w-1, point[0]))
            point[1] = max(0, min(h-1, point[1]))
            pointmap = circle_ADNet(pointmap, point, sigma=sigma) / 255.0
            
            pointmaps.append(pointmap)
        pointmaps = np.stack(pointmaps, axis=0)
        pointmaps = torch.from_numpy(pointmaps).float().unsqueeze(0)
        pointmaps = F.interpolate(pointmaps, size=(int(w * scale), int(h * scale)), mode='bilinear', align_corners=False).squeeze()
        return pointmaps

    def generate_edgemap(self, points, scale=0.25, thickness=1):
        h, w = self.Image_size, self.Image_size
        edgemaps = []
        for is_closed, indices in self.edge_info:
            edgemap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            part = copy.deepcopy(points[np.array(indices)])
            
            part = fit_curve(part, is_closed)
            part[:, 0] = np.clip(part[:, 0], 0, w-1)
            part[:, 1] = np.clip(part[:, 1], 0, h-1)
            if self.edge_type == "VQVAE":
                edgemap = polylines_VQVAE(edgemap, part, is_closed, 255, thickness)
            elif self.edge_type == "ADNet":
                edgemap = polylines_ADNet(edgemap, part, is_closed, 255, thickness) / 255.0
            #offset = 0.5
            #part = (part + offset).astype(np.int32)
            #part[:, 0] = np.clip(part[:, 0], 0, w-1)
            #part[:, 1] = np.clip(part[:, 1], 0, h-1)
            #cv2.polylines(edgemap, [part], is_closed, 255, thickness, cv2.LINE_AA)
            
            edgemaps.append(edgemap)
        edgemaps = np.stack(edgemaps, axis=0)
        edgemaps = torch.from_numpy(edgemaps).float().unsqueeze(0)
        edgemaps = F.interpolate(edgemaps, size=(int(w * scale), int(h * scale)), mode='bilinear', align_corners=False).squeeze()
        return edgemaps
        
    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])

        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        
        if self.mirror:
            if "mirror" in Img_path:
                width = Img.shape[1]-1
                temp_box = [width-BBox[2]-BBox[0], BBox[1], BBox[2], BBox[3]]
                for xy in Points:
                    xy[0] = width - xy[0]
                Points_flip = Points.copy()
                for (q, p) in self.flip_mapping:
                    Points_flip[p] = Points[q]
                    Points_flip[q] = Points[p]
                Points = Points_flip
                BBox = temp_box

        Annotated_Points = Points.copy()

        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)
        
        if self.augmentation_transform is not None:
            input, Points = self.augmentation_transform(input, Points)

        image = copy.deepcopy(input)
        occluded_input = copy.deepcopy(input)

        if self.mask_transform is not None:
            occluded_input, _ = self.mask_transform(occluded_input, Points)

        resized_Points = Points / self.ratio
        landmarks = norm_points(torch.from_numpy(resized_Points), 64, 64)
        
        edgemap = self.generate_edgemap(Points)
        pointmap = self.generate_pointmap(Points)

        resized_input = cv2.resize(input, (64, 64), interpolation=cv2.INTER_LINEAR)
        resized_occluded_input = cv2.resize(occluded_input, (64, 64), interpolation=cv2.INTER_LINEAR)
        resized_image = copy.deepcopy(resized_occluded_input)
        
        if self.normalize_transform is not None:
            # input, _ = self.normalize_transform(input, Points)
            # resized_input, _ = self.normalize_transform(resized_input, resized_Points)
            # resized_occluded_input, _ = self.normalize_transform(resized_occluded_input, resized_Points)
            input = self.normalize_transform(input)
            resized_input = self.normalize_transform(resized_input)
            resized_occluded_input = self.normalize_transform(resized_occluded_input)

        meta = {
            'Annotated_Points': Annotated_Points,
            'Img_path': Img_path,
            'Points': Points,
            'Landmarks': landmarks,
            'Edge_Heatmaps': edgemap,
            'Point_Heatmaps': pointmap,
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
        }
        return input.float(), resized_input.float(), resized_occluded_input.float(), meta, image, resized_image