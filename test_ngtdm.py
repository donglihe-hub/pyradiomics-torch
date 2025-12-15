import numpy as np
import torch

from radiomics_ngtdm_numpy import RadiomicsNGTDM as RadiomicsNGTDM_Numpy
from radiomics_ngtdm_torch import RadiomicsNGTDM as RadiomicsNGTDM_Torch


def build_2d_example():
    img = np.array(
        [
            [1, 2, 5, 2],
            [3, 5, 1, 3],
            [1, 3, 5, 5],
            [3, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(img, dtype=np.uint8)
    return img, mask


def run_compare(voxel_based=False):
    img, mask = build_2d_example()
    settings = {
        "distances": [1],
        "force2D": False,
        "force2Ddimension": 0,
        "kernelRadius": 1,
    }

    feat_np = RadiomicsNGTDM_Numpy(img, mask, **settings)
    feat_np.voxelBased = voxel_based

    feat_torch = RadiomicsNGTDM_Torch(img, mask, **settings)
    feat_torch.voxelBased = voxel_based

    if voxel_based:
        coords = np.argwhere(mask > 0)
        feat_np._initCalculation(coords)
        feat_torch._initCalculation(coords)
    else:
        feat_np._initCalculation()
        feat_torch._initCalculation()

    print(f"\n==== voxelBased = {voxel_based} ====")

    feature_names = [
        "Coarseness",
        "Contrast",
        "Busyness",
        "Complexity",
        "Strength",
    ]

    for name in feature_names:
        fn = f"get{name}FeatureValue"
        f_np = np.asarray(getattr(feat_np, fn)(), dtype=np.float64)
        f_t = getattr(feat_torch, fn)()
        if isinstance(f_t, torch.Tensor):
            f_t = f_t.detach().cpu().numpy().astype(np.float64)
        else:
            f_t = np.asarray(f_t, dtype=np.float64)

        print(f"{name}:")
        print("  numpy :", f_np)
        print("  torch :", f_t)
        print("  max |Δ| =", np.max(np.abs(f_np - f_t)))


if __name__ == "__main__":
    run_compare(voxel_based=False)
    run_compare(voxel_based=True)
