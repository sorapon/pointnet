#!/usr/bin/env python

from __future__ import print_function
import argparse
from distutils.util import strtobool
import os
import numpy as np
import time

import chainer
from chainer import iterators
from chainer import serializers
from chainer.dataset import to_device
from chainer.datasets import TransformDataset
from chainer.cuda import to_cpu
import chainer.computational_graph as c

from chainer_pointnet.models.kdcontextnet.kdcontextnet_cls import KDContextNetCls
from chainer_pointnet.models.kdnet.kdnet_cls import KDNetCls
from chainer_pointnet.models.pointnet.pointnet_cls import PointNetCls
from chainer_pointnet.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
from chainer_pointnet.models.pointnet2.pointnet2_cls_ssg import PointNet2ClsSSG

from ply_dataset import get_train_dataset, get_test_dataset, get_test_data

from chainer_pointnet.utils.kdtree import calc_max_level


def main():
    parser = argparse.ArgumentParser(
        description='ModelNet40 classification')
    # parser.add_argument('--conv-layers', '-c', type=int, default=4)
    parser.add_argument('--method', '-m', type=str, default='point_cls')
    parser.add_argument('--batchsize', '-b', type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--data_index', '-d', type=int, default=0)
    # parser.add_argument('--unit-num', '-u', type=int, default=16)
    parser.add_argument('--model_filename', type=str, default='model.npz')
    parser.add_argument('--trans', type=strtobool, default='true')
    parser.add_argument('--use_bn', type=strtobool, default='true')
    parser.add_argument('--normalize', type=strtobool, default='false')
    parser.add_argument('--residual', type=strtobool, default='false')
    args = parser.parse_args()

    method = args.method
    num_point = args.num_point
    out_dir = args.out
    data_index = args.data_index
    num_class = 40

    # Dataset preparation
    val = get_test_dataset(num_point=num_point)
    if method == 'kdnet_cls' or method == 'kdcontextnet_cls':
        from chainer_pointnet.utils.kdtree import TransformKDTreeCls
        max_level = calc_max_level(num_point)
        print('kdnet_cls max_level {}'.format(max_level))
        return_split_dims = (method == 'kdnet_cls')
        val = TransformDataset(val, TransformKDTreeCls(
            max_level=max_level, return_split_dims=return_split_dims))
        if method == 'kdnet_cls':
            # Debug print
            points, split_dims, t = val[0]
            print('converted to kdnet dataset val', points.shape, split_dims.shape, t)
        if method == 'kdcontextnet_cls':
            # Debug print
            points, t = val[0]
            print('converted to kdcontextnet dataset val', points.shape, t)

    # Network
    # n_unit = args.unit_num
    # conv_layers = args.conv_layers
    trans = args.trans
    use_bn = args.use_bn
    normalize = args.normalize
    residual = args.residual
    dropout_ratio = args.dropout_ratio
    from chainer.dataset.convert import concat_examples
    converter = concat_examples

    if method == 'point_cls':
        print('Test PointNetCls model... trans={} use_bn={} dropout={}'
              .format(trans, use_bn, dropout_ratio))
        model = PointNetCls(
            out_dim=num_class, in_dim=3, middle_dim=64, dropout_ratio=dropout_ratio,
            trans=trans, trans_lam1=0.001, trans_lam2=0.001, use_bn=use_bn,
            residual=residual)
    elif method == 'point2_cls_ssg':
        print('Test PointNet2ClsSSG model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = PointNet2ClsSSG(
            out_dim=num_class, in_dim=3,
            dropout_ratio=dropout_ratio, use_bn=use_bn, residual=residual)
    elif method == 'point2_cls_msg':
        print('Test PointNet2ClsMSG model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = PointNet2ClsMSG(
            out_dim=num_class, in_dim=3,
            dropout_ratio=dropout_ratio, use_bn=use_bn, residual=residual)
    elif method == 'kdnet_cls':
        print('Test KDNetCls model... use_bn={} dropout={}'
              .format(use_bn, dropout_ratio))
        model = KDNetCls(
            out_dim=num_class, in_dim=3,
            dropout_ratio=dropout_ratio, use_bn=use_bn, max_level=max_level,)

        def kdnet_converter(batch, device=None, padding=None):
            # concat_examples to CPU at first.
            result = concat_examples(batch, device=None, padding=padding)
            out_list = []
            for elem in result:
                if elem.dtype != object:
                    # Send to GPU for int/float dtype array.
                    out_list.append(to_device(device, elem))
                else:
                    # Do NOT send to GPU for dtype=object array.
                    out_list.append(elem)
            return tuple(out_list)

        converter = kdnet_converter
    elif method == 'kdcontextnet_cls':
        print('Test KDContextNetCls model... use_bn={} dropout={}'
              'normalize={} residual={}'
              .format(use_bn, dropout_ratio, normalize, residual))
        model = KDContextNetCls(
            out_dim=num_class, in_dim=3,
            dropout_ratio=dropout_ratio, use_bn=use_bn,
            # Below is for non-default customization
            levels=[3, 6, 9],
            feature_learning_mlp_list=[
                [32, 32, 128], [64, 64, 256], [128, 128, 512]],
            feature_aggregation_mlp_list=[[128], [256], [512]],
            normalize=normalize, residual=residual
        )
    else:
        raise ValueError('[ERROR] Invalid method {}'.format(method))

    device = args.gpu
    classifier = model
    load_model = True
    if load_model:
        serializers.load_npz(
            os.path.join(out_dir, args.model_filename), classifier)
    if device >= 0:
        print('using gpu {}'.format(device))
        chainer.cuda.get_device_from_id(device).use()
        classifier.to_gpu()  # Copy the model to the GPU

    if args.gpu >= 0:
        xp = chainer.cuda.cupy
    else:
        xp = np

#    data = np.array(val.__getitem__(data_index)) ## < 2468
#    x = classifier.xp.asarray([data[0]], dtype=xp.float32)

##    points = []
##    for i in range(0, num_point):
##        point = [[[data[0][0][i][0]]], [[data[0][1][i][0]]], [[data[0][2][i][0]]]]
##        points.append(point)
##
##    label = data[1]
##    x = classifier.xp.asarray(points, dtype=xp.float32)
##    print(data)
##    print(x.ndim)
##    print(x[0].shape)
##    ##print("x : {}".format(x))
##    ##print("x dimension : {}".format(x[0].ndim))
##    ##print("x shape : {}".format(x[0].shape[-1]))


    time1 = time.time()
    val_iter = iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)
    x = val_iter.next()
    x_test, t_test = concat_examples(x, device)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y, t1, t2 = classifier.calc(x_test)
    y_arr = y.array
    y_arr = to_cpu(y_arr)

    time2 = time.time()
    elapsed_time = time2-time1
    ## show result
    print("Estimated Label is : ", end="")
    for k in range (0, args.batchsize-1):
        print(y_arr[k].argmax(), end=" "),
    print(y_arr[args.batchsize-1].argmax())
    print("Acutual Label is   : {}".format(t_test))
    print("Elapsed Time : {}".format(elapsed_time))


if __name__ == '__main__':
    main()
