import torch


def _get_angle_count_core(size_list, dist_list, bidirectional=True, force2Ddim=-1):
    """和 get_angle_count_np 完全同逻辑，只用 Python 标量和 list。"""
    Nd = len(size_list)
    Ndist = len(dist_list)

    Na = 0
    for dist_idx in range(Ndist):
        d = int(dist_list[dist_idx])
        if d < 1:
            return 0

        Na_d = 1
        Na_dd = 1
        for dim_idx in range(Nd):
            if dim_idx == force2Ddim:
                continue

            if d < size_list[dim_idx]:
                Na_d *= (2 * d + 1)
                Na_dd *= (2 * d - 1)
            else:
                max_step = 2 * (size_list[dim_idx] - 1) + 1
                Na_d *= max_step
                Na_dd *= max_step

        Na += (Na_d - Na_dd)

    if not bidirectional:
        Na //= 2

    return Na


def _build_angles_core_to_tensor(size_list, dist_list, Nd, Ndist, force2Ddim, Na, device, dtype):
    """和 build_angles_np 完全同逻辑，但写入的是 Torch Tensor。"""
    # 找最大距离
    max_distance = 0
    for d in dist_list:
        if d < 1:
            raise ValueError("Invalid distance (< 1) encountered in build_angles")
        if max_distance < d:
            max_distance = d

    n_offsets = 2 * max_distance + 1

    # offset_stride[Nd-1] = 1; 其余反向累乘
    offset_stride = [0] * Nd
    offset_stride[Nd - 1] = 1
    for dim_idx in range(Nd - 2, -1, -1):
        offset_stride[dim_idx] = offset_stride[dim_idx + 1] * n_offsets

    angles = torch.empty((Na, Nd), dtype=dtype, device=device)

    new_a_idx = 0
    a_idx = 0

    while a_idx < Na:
        a_dist = 0
        valid_angle = True

        for dim_idx in range(Nd):
            offset = max_distance - ((new_a_idx // offset_stride[dim_idx]) % n_offsets)

            if ((dim_idx == force2Ddim and offset != 0) or
                offset >= size_list[dim_idx] or
                offset <= -size_list[dim_idx]):
                a_dist = -1
                valid_angle = False
                break

            angles[a_idx, dim_idx] = int(offset)

            if a_dist < offset:
                a_dist = offset
            elif a_dist < -offset:
                a_dist = -offset

        new_a_idx += 1

        if a_dist < 1:
            continue
        if not valid_angle:
            continue

        for d in dist_list:
            if a_dist == d:
                a_idx += 1
                break

    return angles


def generate_angles_torch(
    size,
    distances=None,
    bidirectional=True,
    force2D=False,
    force2Ddimension=0,
    dtype=torch.int64,
    device=None,
):
    """
    Torch 版：行为与 cmatrices_generate_angles_np / C 实现对齐。

    参数：
        size: 1D tensor 或 array-like
        distances: 1D tensor 或 array-like 或 None
        bidirectional, force2D, force2Ddimension: 语义同上
        dtype: 输出 dtype（整数）
        device: 输出 device（如果 size / distances 是 tensor，会自动继承）
    """
    # 处理 device
    if torch.is_tensor(size):
        if device is None:
            device = size.device
        size_t = size.to(device=device, dtype=torch.int64).view(-1)
    else:
        device = device or "cpu"
        size_t = torch.as_tensor(size, device=device, dtype=torch.int64).view(-1)

    if size_t.ndim != 1:
        raise ValueError("Expected size to be 1D")
    Nd = int(size_t.numel())

    if distances is None:
        dist_t = torch.tensor([1], device=device, dtype=torch.int64)
    else:
        if torch.is_tensor(distances):
            dist_t = distances.to(device=device, dtype=torch.int64).view(-1)
        else:
            dist_t = torch.as_tensor(distances, device=device, dtype=torch.int64).view(-1)

    if dist_t.ndim != 1:
        raise ValueError("Expecting distances to be 1D")
    Ndist = int(dist_t.numel())

    # force2D 逻辑
    force2Ddim = int(force2Ddimension) if force2D else -1

    size_list = [int(x) for x in size_t.tolist()]
    dist_list = [int(x) for x in dist_t.tolist()]

    Na = _get_angle_count_core(size_list, dist_list, bidirectional=bool(bidirectional), force2Ddim=force2Ddim)
    if Na == 0:
        raise RuntimeError("Error getting angle count.")

    angles = _build_angles_core_to_tensor(
        size_list, dist_list, Nd, Ndist, force2Ddim, Na, device=device, dtype=dtype
    )

    return angles


# -----------------------------------------------------

import numpy as np


def get_angle_count_np(size, distances, bidirectional=True, force2Ddim=-1):
    """
    直接翻译 C 版 get_angle_count。
    size: 1D int array-like, shape [Nd]
    distances: 1D int array-like, shape [Ndist]
    """
    size = np.asarray(size, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.int64)

    if size.ndim != 1:
        raise ValueError("Expected size to be 1D")
    if distances.ndim != 1:
        raise ValueError("Expected distances to be 1D")

    Nd = int(size.shape[0])
    Ndist = int(distances.shape[0])

    Na = 0
    for dist_idx in range(Ndist):
        d = int(distances[dist_idx])
        if d < 1:
            # C 里直接 return 0 表示错误
            return 0

        Na_d = 1
        Na_dd = 1
        for dim_idx in range(Nd):
            # 不在 out-of-plane 维度上生成角度
            if dim_idx == force2Ddim:
                continue

            if d < size[dim_idx]:
                # 全距离可用：±d
                Na_d *= (2 * d + 1)
                Na_dd *= (2 * d - 1)
            else:
                # 距离超过图像大小，只能用 size-1
                max_step = 2 * (int(size[dim_idx]) - 1) + 1
                Na_d *= max_step
                Na_dd *= max_step

        Na += (Na_d - Na_dd)

    if not bidirectional:
        Na //= 2

    return Na


def build_angles_np(size, distances, Nd, Ndist, force2Ddim, Na):
    """
    直接翻译 C 版 build_angles，返回 shape = (Na, Nd) 的 int 数组。

    注意：假定 size/distances/force2Ddim 与 get_angle_count_np 保持一致，
    且 Na 已经由 get_angle_count_np 计算好（包含 bidirectional 逻辑）。
    """
    size = np.asarray(size, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.int64)

    size_list = [int(s) for s in size.tolist()]
    dist_list = [int(d) for d in distances.tolist()]

    # 找最大距离
    max_distance = 0
    for d in dist_list:
        if d < 1:
            # C 里返回 1 表示错误，这里直接抛异常更直观
            raise ValueError("Invalid distance (< 1) encountered in build_angles_np")
        if max_distance < d:
            max_distance = d

    n_offsets = 2 * max_distance + 1

    # offset_stride[Nd-1] = 1; 其余反向累乘
    offset_stride = np.empty(Nd, dtype=np.int64)
    offset_stride[Nd - 1] = 1
    for dim_idx in range(Nd - 2, -1, -1):
        offset_stride[dim_idx] = offset_stride[dim_idx + 1] * n_offsets

    # 预分配角度数组
    angles = np.empty((Na, Nd), dtype=np.int64)

    new_a_idx = 0  # 控制 offset 组合的计数器
    a_idx = 0      # 在 angles 中当前要填充的行

    while a_idx < Na:
        a_dist = 0       # 当前角度的 ∞ 范数
        valid_angle = True

        for dim_idx in range(Nd):
            # C: offset = max_distance - (new_a_idx / stride) % n_offsets;
            offset = max_distance - (
                (new_a_idx // int(offset_stride[dim_idx])) % n_offsets
            )

            # 条件同 C 版：
            # 1. force2D 维度上 offset 必须是 0
            # 2. offset 范围不能超出 [-(size[d]-1), (size[d]-1)]
            if ((dim_idx == force2Ddim and offset != 0) or
                offset >= size_list[dim_idx] or
                offset <= -size_list[dim_idx]):
                a_dist = -1  # 标记为非法
                valid_angle = False
                break

            # 写入当前候选角度（只有最终被确认“合法并需要”时才会推进 a_idx）
            angles[a_idx, dim_idx] = offset

            # 更新 ∞ 范数
            if a_dist < offset:
                a_dist = offset
            elif a_dist < -offset:
                a_dist = -offset

        new_a_idx += 1  # 不管角度合法与否，组合计数器都向前走

        # a_dist < 1: 非法 (-1) 或者是 (0,0,0)
        if a_dist < 1:
            continue

        if not valid_angle:
            continue

        # 检查此距离是否在 distances 中
        for d in dist_list:
            if a_dist == d:
                a_idx += 1  # 接受该角度
                break

    return angles


def generate_angles_np(
    size,
    distances=None,
    bidirectional=True,
    force2D=False,
    force2Ddimension=0,
):
    """
    Python/NumPy 版：重现 C 的 cmatrices_generate_angles 行为。

    参数：
        size: 1D array-like, 图像尺寸
        distances: 1D array-like 或 None。None 时默认 [1]
        bidirectional: 是否双向（True/False）
        force2D: 是否强制 2D
        force2Ddimension: 哪一维当作“out-of-plane”

    返回：
        numpy.ndarray, shape = (Na, Nd), dtype=int32（匹配 NPY_INT）
    """
    size_arr = np.asarray(size, dtype=np.int64)
    if size_arr.ndim != 1:
        raise ValueError("Expected a 1D array for size")
    Nd = int(size_arr.shape[0])

    if distances is None:
        distances_arr = np.array([1], dtype=np.int64)
    else:
        distances_arr = np.asarray(distances, dtype=np.int64)
        if distances_arr.ndim != 1:
            raise ValueError("Expecting distances array to be 1D")
    Ndist = int(distances_arr.shape[0])

    # C 里：如果不 force2D，则强制维度 = -1
    force2Ddim = int(force2Ddimension) if force2D else -1

    Na = get_angle_count_np(size_arr, distances_arr, bidirectional=bool(bidirectional), force2Ddim=force2Ddim)
    if Na == 0:
        raise RuntimeError("Error getting angle count.")

    angles = build_angles_np(size_arr, distances_arr, Nd, Ndist, force2Ddim, Na)

    # C 里是 NPY_INT，一般对应 C 的 int（常见是 int32）
    return angles.astype(np.int32, copy=False)