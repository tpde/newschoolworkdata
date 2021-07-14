import math
import numpy as np
import torch as tc

def get_rotate_matrix( angle ):
    theta = math.pi * angle/180.
    return tc.FloatTensor([ [math.cos(theta), -math.sin(theta), 0],
                            [math.sin(theta), math.cos(theta), 0],
                            [0, 0, 1]])


def get_translate_matrix(tsl_h, tsl_w):
    """
        assert( -img_h<=tsl_h and tsl_h<=img_h )
        assert( -img_w<=tsl_w and tsl_w<=img_w )
    """
    tx, ty = tsl_h, tsl_w
    return tc.FloatTensor([ [1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]])



def get_shear_matrix(angle): 
    theta = math.pi * angle/180.
    return tc.FloatTensor( [[1, -math.sin(theta), 0],
                            [0, math.cos(theta), 0],
                            [0,         0,      1]] )

        
def get_zoom_matrix(zx, zy):
    """
    zx, zy : float, zoom rate.
            =1 : no zoom
            >1 : zoom-in (value-1)
            <1 : zoom-out(1-value)
    """
    return tc.FloatTensor([[zx, 0, 0],
                           [0, zy, 0],
                           [0, 0,  1]])



def apply_affine(*inputs, tsf_matrix, interp='bilinear'):
    """
    Perform an affine transforms with various sub-transforms, using
    only one interpolation and without having to instantiate each
    sub-transform individually.

    Arguments
    ---------
    tform_matrix : a 2x3 or 3x3 matrix
        affine transformation matrix to apply
    interp : enum{'bilinear', 'nearest'} or ['bilinear','nearest']
    """
    interp = interp if isinstance(interp, (tuple,list)) else [interp]*len(inputs)

    outputs = []
    for idx, input in enumerate(inputs):
        input_tf = th_affine2d(input, tsf_matrix, mode=interp[idx])
        outputs.append(input_tf)
    return outputs



def th_affine2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on tc.Tensor
    
    Arguments
    ---------
    x : tc.Tensor of size (C, H, W)
        image tensor to be transformed

    matrix : tc.Tensor of size (3, 3) or (2, 3)
        transformation matrix

    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use

    center : boolean
        whether to alter the bias of the transform 
        so the transform is applied about the center
        of the image rather than the origin

    Example
    ------- 
    >>> import torch
    >>> from torchsample.utils import *
    >>> x = tc.zeros(2,1000,1000)
    >>> x[:,100:1500,100:500] = 10
    >>> matrix = tc.FloatTensor([[1.,0,-50],
    ...                             [0,1.,-50]])
    >>> xn = th_affine2d(x, matrix, mode='nearest')
    >>> xb = th_affine2d(x, matrix, mode='bilinear')
    """

    if matrix.dim() == 2:
        matrix = matrix[:2,:]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() == 3:
        if matrix.size()[1:] == (3,3):
            matrix = matrix[:,:2,:]

    A_batch = matrix[:,:,:2]
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0),1,1)
    b_batch = matrix[:,:,2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1),x.size(2))
    coords = _coords.unsqueeze(0).repeat(x.size(0),1,1).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(1) / 2. - 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(2) / 2. - 0.5)
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(1) / 2. - 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(2) / 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp2d(x.contiguous(), new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp2d(x.contiguous(), new_coords)

    return x_transformed


def th_nearest_interp2d(input, coords):
    """
    2d nearest neighbor interpolation tc.Tensor
    """
    # take clamp of coords so they're in the image bounds
    x = tc.clamp(coords[:,:,0], 0, input.size(1)-1).round()
    y = tc.clamp(coords[:,:,1], 0, input.size(2)-1).round()

    stride = tc.LongTensor(input.stride())
    x_ix = x.mul(stride[1]).long()
    y_ix = y.mul(stride[2]).long()

    input_flat = input.view(input.size(0),-1)

    mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

    return mapped_vals.view_as(input)


def th_bilinear_interp2d(input, coords):
    """
    bilinear interpolation in 2d
    """
    x = tc.clamp(coords[:,:,0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = tc.clamp(coords[:,:,1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = tc.LongTensor(input.stride())
    x0_ix = x0.mul(stride[1]).long()
    x1_ix = x1.mul(stride[1]).long()
    y0_ix = y0.mul(stride[2]).long()
    y1_ix = y1.mul(stride[2]).long()

    input_flat = input.view(input.size(0),-1)

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))
    
    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)


def th_affine3d(x, matrix, mode='trilinear', center=True):
    """
    3D Affine image transform on tc.Tensor
    """
    A = matrix[:3,:3]
    b = matrix[:3,3]

    # make a meshgrid of normal coordinates
    coords = th_iterproduct(x.size(1),x.size(2),x.size(3)).float()


    if center:
        # shift the coordinates so center is the origin
        coords[:,0] = coords[:,0] - (x.size(1) / 2. - 0.5)
        coords[:,1] = coords[:,1] - (x.size(2) / 2. - 0.5)
        coords[:,2] = coords[:,2] - (x.size(3) / 2. - 0.5)

    
    # apply the coordinate transformation
    new_coords = coords.mm(A.t().contiguous()) + b.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,0] = new_coords[:,0] + (x.size(1) / 2. - 0.5)
        new_coords[:,1] = new_coords[:,1] + (x.size(2) / 2. - 0.5)
        new_coords[:,2] = new_coords[:,2] + (x.size(3) / 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp3d(x, new_coords)
    elif mode == 'trilinear':
        x_transformed = th_trilinear_interp3d(x, new_coords)
    else:
        x_transformed = th_trilinear_interp3d(x, new_coords)

    return x_transformed


def th_nearest_interp3d(input, coords):
    """
    2d nearest neighbor interpolation tc.Tensor
    """
    # take clamp of coords so they're in the image bounds
    coords[:,0] = tc.clamp(coords[:,0], 0, input.size(1)-1).round()
    coords[:,1] = tc.clamp(coords[:,1], 0, input.size(2)-1).round()
    coords[:,2] = tc.clamp(coords[:,2], 0, input.size(3)-1).round()

    stride = tc.LongTensor(input.stride())[1:].float()
    idx = coords.mv(stride).long()

    input_flat = th_flatten(input)

    mapped_vals = input_flat[idx]

    return mapped_vals.view_as(input)


def th_trilinear_interp3d(input, coords):
    """
    trilinear interpolation of 3D tc.Tensor image
    """
    # take clamp then floor/ceil of x coords
    x = tc.clamp(coords[:,0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    # take clamp then floor/ceil of y coords
    y = tc.clamp(coords[:,1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1
    # take clamp then floor/ceil of z coords
    z = tc.clamp(coords[:,2], 0, input.size(3)-2)
    z0 = z.floor()
    z1 = z0 + 1

    stride = tc.LongTensor(input.stride())[1:]
    x0_ix = x0.mul(stride[0]).long()
    x1_ix = x1.mul(stride[0]).long()
    y0_ix = y0.mul(stride[1]).long()
    y1_ix = y1.mul(stride[1]).long()
    z0_ix = z0.mul(stride[2]).long()
    z1_ix = z1.mul(stride[2]).long()

    input_flat = th_flatten(input)

    vals_000 = input_flat[x0_ix+y0_ix+z0_ix]
    vals_100 = input_flat[x1_ix+y0_ix+z0_ix]
    vals_010 = input_flat[x0_ix+y1_ix+z0_ix]
    vals_001 = input_flat[x0_ix+y0_ix+z1_ix]
    vals_101 = input_flat[x1_ix+y0_ix+z1_ix]
    vals_011 = input_flat[x0_ix+y1_ix+z1_ix]
    vals_110 = input_flat[x1_ix+y1_ix+z0_ix]
    vals_111 = input_flat[x1_ix+y1_ix+z1_ix]

    xd = x - x0
    yd = y - y0
    zd = z - z0
    xm1 = 1 - xd
    ym1 = 1 - yd
    zm1 = 1 - zd

    x_mapped = (vals_000.mul(xm1).mul(ym1).mul(zm1) +
                vals_100.mul(xd).mul(ym1).mul(zm1) +
                vals_010.mul(xm1).mul(yd).mul(zm1) +
                vals_001.mul(xm1).mul(ym1).mul(zd) +
                vals_101.mul(xd).mul(ym1).mul(zd) +
                vals_011.mul(xm1).mul(yd).mul(zd) +
                vals_110.mul(xd).mul(yd).mul(zm1) +
                vals_111.mul(xd).mul(yd).mul(zd))

    return x_mapped.view_as(input)

def th_iterproduct(*args):
    return tc.from_numpy(np.indices(args).reshape((len(args),-1)).T)

def th_flatten(x):
    """Flatten tensor"""
    return x.contiguous().view(-1)




