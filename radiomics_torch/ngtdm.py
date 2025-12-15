from __future__ import annotations
import torch
import torch.nn.functional as F

from radiomics_torch import base


class RadiomicsNGTDM(base.RadiomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super().__init__(inputImage, inputMask, **kwargs)
        self.P_ngtdm = None

        # 保持原版逻辑，binning 后 imageArray 为离散灰度（int）
        self.imageArray = self._applyBinning(self.imageArray)

    # ------------------ 主入口：与原版一致 ------------------

    def _initCalculation(self, voxelCoordinates=None):
        self.P_ngtdm = self._calculateMatrix(voxelCoordinates)
        self._calculateCoefficients()

    # ------------------ 计算 NGTDM 矩阵 ------------------

    def _calculateMatrix(self, voxelCoordinates=None):
        """
        返回形状：
            voxelBased=False → (1, Ng_eff, 3)
            voxelBased=True  → (Nvox, Ng_eff, 3)
        """
        img = torch.as_tensor(self.imageArray, dtype=torch.float32)
        mask = torch.as_tensor(self.maskArray, dtype=torch.float32)

        distances = self.settings.get("distances", [1])
        delta = int(distances[0])
        kernelRadius = self.settings.get("kernelRadius", 1)

        force2D = self.settings.get("force2D", False)
        force2Ddim = self.settings.get("force2Ddimension", 0)

        Ng = int(img[mask > 0].max().item())
        ivector = torch.arange(1, Ng + 1, dtype=torch.float32)

        # ---------------------- non-voxel-based ----------------------
        if not self.voxelBased:
            if img.ndim == 2 or (force2D and img.ndim == 3):
                P = self._ngtdm_whole_2d(img, mask, delta, Ng)
            else:
                P = self._ngtdm_whole_3d(img, mask, delta, Ng)

            non_empty = (P[:, :, 0].sum(dim=0) != 0)
            return P[:, non_empty, :]

        # ---------------------- voxel-based ----------------------
        coords = torch.as_tensor(voxelCoordinates, dtype=torch.long)
        Nvox = coords.shape[0]

        P_vox = torch.zeros((Nvox, Ng, 3), dtype=torch.float32)
        P_vox[:, :, 2] = ivector  # 灰度值

        for v in range(Nvox):
            if img.ndim == 3:
                z, y, x = coords[v]
                z0, z1 = max(z - kernelRadius, 0), min(z + kernelRadius + 1, img.shape[0])
                y0, y1 = max(y - kernelRadius, 0), min(y + kernelRadius + 1, img.shape[1])
                x0, x1 = max(x - kernelRadius, 0), min(x + kernelRadius + 1, img.shape[2])

                img_win = img[z0:z1, y0:y1, x0:x1]
                mask_win = mask[z0:z1, y0:y1, x0:x1]

                P = self._ngtdm_whole_3d(img_win, mask_win, delta, Ng)
            else:
                y, x = coords[v]
                y0, y1 = max(y-kernelRadius,0), min(y+kernelRadius+1,img.shape[0])
                x0, x1 = max(x-kernelRadius,0), min(x+kernelRadius+1,img.shape[1])

                img_win = img[y0:y1, x0:x1]
                mask_win = mask[y0:y1, x0:x1]

                P = self._ngtdm_whole_2d(img_win, mask_win, delta, Ng)

            P_vox[v] = P[0]

        # 删除空灰度级（与原版一致）
        non_empty = (P_vox[:, :, 0].sum(dim=0) != 0)
        P_vox = P_vox[:, non_empty, :]
        return P_vox

    # --------------------- 整块 ROI 的 2D NGTDM ---------------------

    def _ngtdm_whole_2d(self, img, mask, delta, Ng):
        H, W = img.shape
        kernel = torch.ones((1, 1, 2*delta+1, 2*delta+1))
        kernel[:, :, delta, delta] = 0  # 移除中心像素

        img_in = (img * mask).unsqueeze(0).unsqueeze(0)
        mask_in = mask.unsqueeze(0).unsqueeze(0)

        neigh_sum = F.conv2d(img_in, kernel, padding=delta)[0, 0]
        neigh_count = F.conv2d(mask_in, kernel, padding=delta)[0, 0]

        valid = (mask > 0) & (neigh_count > 0)
        if not torch.any(valid):
            return torch.zeros((1, Ng, 3))

        img_valid = img[valid]
        avg_valid = neigh_sum[valid] / neigh_count[valid]

        idx = img_valid.long() - 1
        n_i = torch.bincount(idx, minlength=Ng).float()

        diff = torch.abs(img_valid - avg_valid)
        s_i = torch.zeros(Ng)
        s_i.scatter_add_(0, idx, diff)

        ivector = torch.arange(1, Ng + 1).float()
        P = torch.stack([n_i, s_i, ivector], dim=-1).unsqueeze(0)
        return P

    # --------------------- 整块 ROI 的 3D NGTDM ---------------------

    def _ngtdm_whole_3d(self, img, mask, delta, Ng):
        D, H, W = img.shape
        kernel = torch.ones((1, 1, 2*delta+1, 2*delta+1, 2*delta+1))
        kernel[:, :, delta, delta, delta] = 0

        img_in = (img * mask).unsqueeze(0).unsqueeze(0)
        mask_in = mask.unsqueeze(0).unsqueeze(0)

        neigh_sum = F.conv3d(img_in, kernel, padding=delta)[0, 0]
        neigh_count = F.conv3d(mask_in, kernel, padding=delta)[0, 0]

        valid = (mask > 0) & (neigh_count > 0)
        if not torch.any(valid):
            return torch.zeros((1, Ng, 3))

        img_valid = img[valid]
        avg_valid = neigh_sum[valid] / neigh_count[valid]

        idx = img_valid.long() - 1
        n_i = torch.bincount(idx, minlength=Ng).float()

        diff = torch.abs(img_valid - avg_valid)
        s_i = torch.zeros(Ng)
        s_i.scatter_add_(0, idx, diff)

        ivector = torch.arange(1, Ng + 1).float()
        P = torch.stack([n_i, s_i, ivector], dim=-1).unsqueeze(0)
        return P

    # ---------------------- 系数计算（原版逻辑） ----------------------

    def _calculateCoefficients(self):
        P = self.P_ngtdm  # (Nvox, Ng_eff, 3)

        n_i = P[:, :, 0]
        s_i = P[:, :, 1]
        i = P[:, :, 2]

        Nvp = n_i.sum(dim=1)
        self.coefficients["Nvp"] = Nvp

        p_i = n_i / Nvp[:, None]
        self.coefficients["p_i"] = p_i

        self.coefficients["s_i"] = s_i
        self.coefficients["ivector"] = i

        Ngp = (n_i > 0).sum(dim=1)
        self.coefficients["Ngp"] = Ngp

        self.coefficients["p_zero"] = (p_i == 0).nonzero(as_tuple=True)

    # ---------------------- 以下特征实现保持不变 ----------------------

    def getCoarsenessFeatureValue(self):
        p_i = self.coefficients["p_i"]
        s_i = self.coefficients["s_i"]
        sum_coarse = (p_i * s_i).sum(dim=1)
        return torch.where(sum_coarse != 0, 1.0 / sum_coarse, torch.full_like(sum_coarse, 1e6))

    def getContrastFeatureValue(self):
        Ngp = self.coefficients["Ngp"]
        Nvp = self.coefficients["Nvp"]
        p_i = self.coefficients["p_i"]
        s_i = self.coefficients["s_i"]
        i = self.coefficients["ivector"]

        diff_sq = (i[:, :, None] - i[:, None, :]) ** 2
        pij = p_i[:, :, None] * p_i[:, None, :]
        contrast = (pij * diff_sq).sum(dim=(1, 2))

        contrast = contrast * s_i.sum(dim=1) / Nvp
        div = Ngp * (Ngp - 1)
        return torch.where(div != 0, contrast / div, torch.zeros_like(contrast))

    def getBusynessFeatureValue(self):
        p_i = self.coefficients["p_i"]
        s_i = self.coefficients["s_i"]
        i = self.coefficients["ivector"]
        p_zero = self.coefficients["p_zero"]

        i_pi = i * p_i
        absdiff = torch.abs(i_pi[:, :, None] - i_pi[:, None, :])

        absdiff[p_zero[0], :, p_zero[1]] = 0
        absdiff[p_zero[0], p_zero[1], :] = 0

        absdiff_sum = absdiff.sum(dim=(1, 2))
        busyness = (p_i * s_i).sum(dim=1)

        return torch.where(absdiff_sum != 0, busyness / absdiff_sum, torch.zeros_like(busyness))

    def getComplexityFeatureValue(self):
        Nvp = self.coefficients["Nvp"]
        p_i = self.coefficients["p_i"]
        s_i = self.coefficients["s_i"]
        i = self.coefficients["ivector"]
        p_zero = self.coefficients["p_zero"]

        pi_si = p_i * s_i
        numerator = pi_si[:, :, None] + pi_si[:, None, :]
        numerator[p_zero[0], :, p_zero[1]] = 0
        numerator[p_zero[0], p_zero[1], :] = 0

        divisor = p_i[:, :, None] + p_i[:, None, :]
        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)

        diff = torch.abs(i[:, :, None] - i[:, None, :])
        complexity = (diff * numerator / divisor).sum(dim=(1, 2)) / Nvp
        return complexity

    def getStrengthFeatureValue(self):
        p_i = self.coefficients["p_i"]
        s_i = self.coefficients["s_i"]
        i = self.coefficients["ivector"]
        p_zero = self.coefficients["p_zero"]

        sum_s_i = s_i.sum(dim=1)
        strength = (p_i[:, :, None] + p_i[:, None, :]) * (i[:, :, None] - i[:, None, :]) ** 2

        strength[p_zero[0], :, p_zero[1]] = 0
        strength[p_zero[0], p_zero[1], :] = 0

        strength = strength.sum(dim=(1, 2))
        return torch.where(sum_s_i != 0, strength / sum_s_i, torch.zeros_like(strength))
