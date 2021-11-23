import sys

sys.path.append("rxfm-net/src/")

import numpy as np
import torch
from nilearn.datasets import MNI152_FILE_PATH
from functools import partial

import utils
import losses
import custom_image3d as ci3d
import rxfm_net


def load_data(img_size=[96, 96, 96]):

    with torch.no_grad():
        img_vol, _, _ = utils.load_scale_and_pad(
            MNI152_FILE_PATH,
            img_size,
            initial_resize=[128, 128, 128],
            rescale=[96, 96, 96],
        )

        print(f"Image volume size: {img_vol.size()}")

        img_vol = img_vol.float()
        mask = (img_vol > 0).float()

        rx_train = 1
        ry_train = 1
        rz_train = 1
        tx_train = 5
        ty_train = 5
        tz_train = 5

        mat = ci3d.create_transform(
            rx=rx_train,
            ry=ry_train,
            rz=rz_train,
            tx=2.0 * tx_train / img_size[0],
            ty=2.0 * ty_train / img_size[1],
            tz=2.0 * tz_train / img_size[2],
        )

        mat = mat[np.newaxis, :, :]
        mat = mat[:, 0:3, :]
        mat = torch.tensor(mat).float()

        print(mat)
        grids = torch.nn.functional.affine_grid(mat, [1, 1] + img_size)
        second_img_vol = torch.nn.functional.grid_sample(
            img_vol, grids, mode="bilinear"
        ).detach()
        second_mask = (second_img_vol > 0).float()

    loader_dict = {
        "scan_1": img_vol,
        "mask_1": mask,
        "scan_2": second_img_vol,
        "mask_2": second_mask,
        "xfm_1to2": mat,
    }

    return loader_dict


def initialize_net(img_size=[96, 96, 96, 1], loss_func_name="xfm_6D", n_chan=64):

    # Initialize CUDA
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"using {device} as device")

    # Load the loss function
    net_obj = rxfm_net.RXFM_Net_Wrapper(img_size[0:3], n_chan, masks_as_input=False)

    if loss_func_name == "xfm_MSE":
        loss_func = partial(losses.xfm_loss_MSE, weight_R=1.0, weight_T=5.0)
    elif loss_func_name == "xfm_6D":
        loss_func = partial(losses.xfm_loss_6D, weight_R=1.0, weight_T=5.0)
    else:
        print("Loss function not recognized")
        exit(1)

    net_obj = net_obj.to(device)
    LR = 0.000025
    optimizer = torch.optim.Adam(net_obj.parameters(), lr=LR)

    return net_obj, optimizer, loss_func, device


def normalized_cross_correlation(x, y, return_map, reduction="mean", eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x, y)
    dev_xx = torch.mul(x, x)
    dev_yy = torch.mul(y, y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(
        dev_xy + eps / dev_xy.shape[1],
        torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps,
    )
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == "mean":
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == "sum":
        ncc = torch.sum(ncc)
    else:
        raise KeyError("unsupported reduction type: %s" % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map


def pairwise_ncc(filter_bank):
    n_filters = len(filter_bank)
    nccs = 0
    for i in range(n_filters):
        for j in range(i + 1, n_filters):
            ncc = normalized_cross_correlation(
                filter_bank[i], filter_bank[j], return_map=False
            )
            nccs += ncc
    return nccs


def train_func(net_obj, optimizer, loss_func, device, loader_dict, batch_size=1):
    input_1 = loader_dict["scan_1"].to(device)
    input_2 = loader_dict["scan_2"].to(device)

    optimizer.zero_grad()

    # remember xfm is flipped during affine_grid
    # from
    # https://discuss.pytorch.org/t/unexpected-behaviour-for-affine-grid-and-grid-sample-with-3d-inputs/75010/5
    # net_obj MUST take both scans as input, and output a transform between them.
    # for rxfm net, the forward operator with only one scan should be implemented
    # separately!!!
    xfm_1to2, output_1, output_2 = net_obj.forward((input_1, input_2))

    # XFM LOSS
    real_xfm_1to2 = loader_dict["xfm_1to2"].to(device)
    loss_val = loss_func(real_xfm_1to2, xfm_1to2)
    # loss_val += pairwise_ncc(output_1) + pairwise_ncc(output_2)
    del output_1, output_2

    loss_val.backward(retain_graph=True)
    optimizer.step()

    train_loss = loss_val.item() * batch_size

    print(real_xfm_1to2)
    print(xfm_1to2)

    del real_xfm_1to2
    del loss_val
    del xfm_1to2
    del input_1, input_2
    torch.cuda.empty_cache()

    return train_loss


def main(n_epochs=100):

    # Initialize the network
    loader_dict = load_data()
    net_obj, optimizer, loss_func, device = initialize_net()

    # Train the network
    for epoch in range(n_epochs):
        print(epoch, flush=True)
        train_loss = 0
        train_loss = train_func(net_obj, optimizer, loss_func, device, loader_dict)
        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss), flush=True)

    # Save the network
    input_1 = loader_dict["scan_1"]  # [0, 0, ...].cpu()
    input_2 = loader_dict["scan_2"]  # [0, 0, ...].cpu()
    net_obj.eval()
    xfm_1to2, output_1, output_2 = net_obj.forward(
        (input_1.to(device), input_2.to(device))
    )
    xfm_1to2 = xfm_1to2.cpu()

    torch.save(output_1, "scripts/normal/output_1.pt")
    torch.save(output_2, "scripts/normal/output_2.pt")
    torch.save(input_1, "scripts/normal/input_1.pt")
    torch.save(input_2, "scripts/normal/input_2.pt")
    torch.save(net_obj, "scripts/normal/normal_weights.pth")


if __name__ == "__main__":
    main()
