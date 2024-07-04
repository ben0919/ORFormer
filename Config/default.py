from yacs.config import CfgNode as CN

import os

_C = CN()

_C.MODEL = CN()
_C.MODEL.IMG_SIZE = 256

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.HYPERPARAMETERS = CN()

_C.WFLW = CN()
_C.WFLW.ROOT = './Dataset/WFLW'
_C.WFLW.NUM_POINT = 98
_C.WFLW.NUM_EDGE = 15
_C.WFLW.FRACTION = 1.20
_C.WFLW.EDGE_INFO = [
    [False, [0, 1, 2, 3, 4, 5 ,6 ,7 ,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]], 
    [False, [33, 34, 35, 36, 37]],
    [False, [38, 39, 40, 41, 33]],
    [False, [42, 43, 44, 45, 46]],
    [False, [46, 47, 48, 49, 50]],
    [False, [51, 52, 53, 54]],
    [False, [55, 56, 57, 58, 59]],
    [False, [60, 61, 62, 63, 64]],
    [False, [64, 65, 66, 67, 60]],
    [False, [68, 69, 70, 71, 72]],
    [False, [72, 73, 74, 75, 68]],
    [False, [76, 77, 78, 79, 80, 81, 82]],
    [False, [82, 83, 84, 85, 86, 87, 76]],
    [False, [88, 89, 90, 91, 92]],
    [False, [92, 93, 94, 95, 88]],
]
_C.WFLW.FLIP_MAPPING = [
    [0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
    [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],
    [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],
    [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
    [55, 59], [56, 58],
    [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
    [88, 92], [89, 91], [95, 93], [96, 97]
]
_C.WFLW.SCALE = 0.05
_C.WFLW.ROTATION = 15
_C.WFLW.TRANSLATION = 0.05
_C.WFLW.OCCLUSION_MEAN = 0.20
_C.WFLW.OCCLUSION_STD = 0.08
_C.WFLW.DATA_FORMAT = "RGB"
_C.WFLW.FLIP = True
_C.WFLW.CHANNEL_TRANSFER = True
_C.WFLW.OCCLUSION = True
_C.WFLW.INITIAL_PATH = './Config/init_98.npz'

_C.W300 = CN()
_C.W300.ROOT = './Dataset/300W'
_C.W300.NUM_POINT = 68
_C.W300.NUM_EDGE = 13
_C.W300.FRACTION = 1.20
_C.W300.EDGE_INFO = [
    [False, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
    [False, [17, 18, 19, 20, 21]], 
    [False, [22, 23, 24, 25, 26]], 
    [False, [27, 28, 29, 30]], 
    [False, [31, 32, 33, 34, 35]],
    [False, [36, 37, 38, 39]], 
    [False, [39, 40, 41, 36]], 
    [False, [42, 43, 44, 45]], 
    [False, [45, 46, 47, 42]],
    [False, [48, 49, 50, 51, 52, 53, 54]], 
    [False, [54, 55, 56, 57, 58, 59, 48]],
    [False, [60, 61, 62, 63, 64]],
    [False, [64, 65, 66, 67, 60]]
]
_C.W300.FLIP_MAPPING = [
    [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
    [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
    [31, 35], [32, 34],
    [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
    [48, 54], [49, 53], [50, 52], [61, 63], [60, 64], [67, 65], [58, 56], [59, 55],
]
_C.W300.SCALE = 0.05
_C.W300.ROTATION = 15
_C.W300.TRANSLATION = 0.05
_C.W300.OCCLUSION_MEAN = 0.20
_C.W300.OCCLUSION_STD = 0.08
_C.W300.DATA_FORMAT = "RGB"
_C.W300.FLIP = True
_C.W300.CHANNEL_TRANSFER = True
_C.W300.OCCLUSION = True
_C.W300.INITIAL_PATH = './Config/init_68.npz'


_C.COFW = CN()
_C.COFW.ROOT = './Dataset/COFW'
_C.COFW.NUM_POINT = 29
_C.COFW.NUM_EDGE = 14
_C.COFW.FRACTION = 1.30
_C.COFW.EDGE_INFO = [
    [False, [0, 4, 2]],  
    [False, [2, 5, 0]], 
    [False, [1, 6, 3]],
    [False, [3, 7, 1]],
    [False, [8, 12, 10]], 
    [False, [10, 13, 8]],
    [False, [9, 14, 11]],
    [False, [11, 15, 9]], 
    [False, [18, 21, 19]],
    [False, [20, 21]],  
    [False, [22, 26, 23]],
    [False, [23, 27, 22]],  
    [False, [22, 24, 23]],
    [False, [23, 25, 22]],  
]
_C.COFW.FLIP_MAPPING = [
    [0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11], [12, 14], [16, 17], [13, 15], [18, 19], [22, 23],
]
