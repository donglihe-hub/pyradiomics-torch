import torch

def torch_delete(arr, obj, axis=None):
    """Delete sub-arrays from a tensor along a given axis.

    This function mimics the behavior of numpy.delete for PyTorch tensors.

    Parameters
    ----------
    arr : torch.Tensor
        Input tensor.
    obj : int, slice, or sequence of ints
        Indices of sub-arrays to delete.
    axis : int, optional
        Axis along which to delete the sub-arrays. If None, the input tensor is flattened.

    Returns
    -------
    torch.Tensor
        Tensor with specified sub-arrays deleted.
    """
    if axis is None:
        arr = arr.flatten()
        axis = 0

    mask = torch.ones(arr.size(axis), dtype=torch.bool, device=arr.device)
    if isinstance(obj, int):
        obj = [obj]
    mask[obj] = False

    return torch.index_select(arr, axis, torch.nonzero(mask, as_tuple=False).squeeze())