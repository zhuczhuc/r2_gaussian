# -*- coding:utf-8 -*-
# ===============================================================
#
#    Copyright (C) TINAVI, Inc. All Rights Reserved.
#
#    @Create Author : zhuc
#    @Create Time   : 2024/11/17 19:20
#
# ===============================================================

import numpy as np
import tigre
import tigre.algorithms as algs

I0 = 4500

PROJ_PATH = r"E:\code\tinavi\tai-cb-sdk\data\cbct_recon\demo_data\raw_proj.raw"
GEO_PARAMETER = r"E:\code\tinavi\tai-cb-sdk\data\cbct_recon\demo_data\param\geo_para.raw"
GEO_NUM = 14
PROJS = 200
nDetector = [976, 976]

DSD = 1164
DSO = 622
sDetector = [296.7] * 2
nVoxel = [512, 768, 768]
sVoxel = [160, 240, 240]
offOrigin = [0] * 3
offDetector = [0] * 3

accuracy = 0.5
angle_last = 360
angle_start = 0

if __name__ == "__main__":
    # vol_cgls = np.fromfile(r"E:\vol_gcls.raw", dtype=np.float32)
    # v98 = np.percentile(vol_cgls, 99)
    # vol_cgls[vol_cgls > v98] *= 0.6
    # vol_cgls.tofile(r"E:\vol_gcls_1.raw")
    # exit(1)

    proj0 = np.fromfile(PROJ_PATH, dtype=np.uint16)
    proj1 = np.reshape(proj0, (200, 976, 976))
    proj1 = -np.log(proj1 / I0).astype(np.float32)

    geo_para = np.fromfile(GEO_PARAMETER, dtype=np.float32)
    geo_para = np.reshape(geo_para, (PROJS, GEO_NUM))

    geo = tigre.geometry(mode="cone")
    geo.DSD = DSD
    geo.DSO = DSO
    geo.nDetector = np.array(nDetector)
    geo.sDetector = np.array(sDetector)
    geo.dDetector = geo.sDetector / geo.nDetector

    geo.nVoxel = np.array(nVoxel)
    geo.sVoxel = np.array(sVoxel)
    geo.dVoxel = geo.sVoxel / geo.nVoxel

    off_origin = geo_para[:, 3:6]
    geo.offOrigin = off_origin

    off_detector = geo_para[:, 6:8]
    geo.offDetector = np.array(off_detector)
    # geo.offDetector = np.array(offDetector)

    geo.accuracy = 0.5

    angles = geo_para[:, 0:3]

    vol_fdk = algs.fdk(proj1, geo, angles)
    vol_fdk.tofile(r"E:\vol_fdk.raw")

    # vol_cgls, _ = algs.cgls(proj1, geo, angles, 15, computel2=True)
    # vol_cgls.tofile(r"E:\vol_gcls.raw")

    a = 0
