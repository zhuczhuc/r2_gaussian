# -*- coding:utf-8 -*-
# ===============================================================
#
#    Copyright (C) TINAVI, Inc. All Rights Reserved.
#
#    @Create Author : zhuc
#    @Create Time   : 2024/11/6 19:47
#
# ===============================================================
import os

import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
import math

I0 = 4500
SAVE_PATH = R"../data/cios/"
PROJ_PATH = r"E:\code\tinavi\tai-cb-sdk\data\cbct_recon\demo_data\raw_proj.raw"
FDK_IMG = r"E:\code\tinavi\tai-cb-sdk\data\cbct_recon\a.raw"
ROWS = 976
COLUMNS = 976
PROJS = 200
MARGIN = 30
GEO_PARAMETER = r"E:\code\tinavi\tai-cb-sdk\data\cbct_recon\demo_data\param\geo_para.raw"
GEO_NUM = 14

DSD = 1164
DSO = 622
sDetector = [296.7] * 2
nVoxel = [512] * 3
sVoxel = [160] * 3
offOrigin = [0] * 3
offDetector = [0] * 3

train_num = 100
accuracy = 0.5
angle_last = 360
angle_start = 0


def normalize_angle(angle):
    normalized_angle = angle % 360
    if normalized_angle < 0:
        normalized_angle += 360
    return normalized_angle


if __name__ == "__main__":
    # # 预处理投影数据
    proj0 = np.fromfile(PROJ_PATH, dtype=np.uint16)
    proj1 = np.reshape(proj0, (200, 976, 976))
    proj1 = -np.log(proj1 / I0)

    # proj_max = np.max(proj1)
    # proj1[:, 0:MARGIN, :] = proj_max
    # proj1[:, :, 0:MARGIN] = proj_max
    # proj1[:, (ROWS - MARGIN):, :] = proj_max
    # proj1[:, :, (COLUMNS - MARGIN):] = proj_max
    # proj1.tofile(r"E:\a.raw")

    geo_para = np.fromfile(GEO_PARAMETER, dtype=np.float32)
    geo_para = np.reshape(geo_para, (PROJS, GEO_NUM))
    angles = geo_para[:, 0]

    nDetector = [ROWS, COLUMNS]
    sDetector = sDetector
    nVoxel = nVoxel
    sVoxel = sVoxel
    offOrigin = offOrigin
    bbox = np.array(
        [
            np.array(offOrigin) - np.array(sVoxel) / 2,
            np.array(offOrigin) + np.array(sVoxel) / 2,
        ]
    ).tolist()

    scanner_cfg = {
        "mode": "cone",
        "DSD": DSD,
        "DSO": DSO,
        "nDetector": nDetector,
        "sDetector": sDetector,
        "nVoxel": nVoxel,
        "sVoxel": sVoxel,
        "offOrigin": offOrigin,
        "offDetector": offDetector,
        "accuracy": accuracy,
        "totalAngle": angle_last - angle_start,
        "startAngle": angle_start,
        "noise": True,
        "filter": None,
    }
    ct_gt_save_path = "vol_gt.npy"
    ct_img = np.fromfile(FDK_IMG, dtype=np.float32).reshape(512, 512, 512)
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(SAVE_PATH, ct_gt_save_path), ct_img)

    train_path = Path(os.path.join(SAVE_PATH, "proj_train"))
    test_path = Path(os.path.join(SAVE_PATH, "proj_test"))
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    projection_train_list = []
    projection_test_list = []

    for i in range(PROJS):
        # angle = angles[i]
        # angle_deg = normalize_angle(math.degrees(angle))
        # angle_rad = math.radians(angle_deg)

        # angle_rad = math.radians(math.degrees(angle))

        # 不用几何校正
        angle_rad = math.radians(1 * i)

        proj_name = f"{i:04d}" + ".npy"
        proj_arr = proj1[i, :, :].astype(np.float32)

        if i < train_num:
            proj_path = str(train_path / proj_name)
            np.save(proj_path, proj_arr)
            projection_train_list.append({
                "file_path": os.path.join("proj_train", proj_name),
                "angle": angle_rad,
            })
        else:
            proj_path = str(test_path / proj_name)
            np.save(proj_path, proj_arr)
            projection_test_list.append({
                "file_path": os.path.join("proj_test", proj_name),
                "angle": angle_rad,
            })

    meta_data = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "radius": 1.0,
        "bbox": bbox,
        "proj_train": projection_train_list,
        "proj_test": projection_test_list,
    }
    with open(os.path.join(SAVE_PATH, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

