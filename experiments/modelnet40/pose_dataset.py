import os

import chainer
from chainer.datasets.concatenated_dataset import ConcatenatedDataset
import numpy as np

# data is downloaded when importing provider
import provider


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PoseDataset(chainer.dataset.DatasetMixin):
    """This if partial dataset"""

    def __init__(self, h5_filepath, num_point=1024, augment=False):
        print('loading ', h5_filepath)
        data, pose = provider.loadDataFile_with_pose(h5_filepath)
        assert len(data) == len(pose)
        # data: (2048, 2048, 3) - (batchsize, point, xyz)
        # Reduce num point here.
        # self.data = data[:, :num_point, :].astype(np.float32)
        self.data = data
        # (2048,) - (batchsize,)
        self.pose = np.squeeze(pose).astype(np.float32)
        self.augment = augment
        self.num_point = num_point
        self.length = len(data)
        print('length ', self.length)

    def __len__(self):
        """return length of this dataset"""
        return self.length

    def get_example(self, i):
        """Return i-th data"""
        if self.augment:
            rotated_data = provider.rotate_point_cloud(
                self.data[i:i + 1, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            point_data = jittered_data[0]
        else:
            point_data = self.data[i]
        # pint_data (2048, 3): (num_point, k) --> convert to (k, num_point, 1)
        point_data = np.transpose(
            point_data.astype(np.float32), (1, 0))[:, :, None]
        assert point_data.dtype == np.float32
        assert self.pose[i].dtype == np.float32
        return point_data, self.pose[i]


def get_train_dataset(num_point=1024):
    print('get train num_point ', num_point)
    train_files = provider.getDataFiles(
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_pose_files.txt'))
    return ConcatenatedDataset(
        *(PoseDataset(filepath, num_point=num_point, augment=True) for filepath in train_files))


def get_test_dataset(num_point=1024):
    print('get test num_point ', num_point)
    test_files = provider.getDataFiles(
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_pose_files.txt'))
    return ConcatenatedDataset(
        *(PoseDataset(filepath, num_point=num_point, augment=False) for filepath in test_files))

def get_test_data(num_point=1024):
    print('get test num_point ', num_point)
    test_files = provider.getDataFiles(
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_pose_files.txt'))
    print("test_file : {}".format(test_files))
    data = (PoseDataset(filepath, num_point=num_point, augment=False) for filepath in test_files)
    return data




if __name__ == '__main__':
    # --- PlyDataset check ---
    train_files = provider.getDataFiles(
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_pose_files.txt'))
    d0 = PoseDataset(train_files[0], augment=True)
    data_point, pose = d0[3]
    print('data_point', data_point.shape, 'pose', pose)

    # --- Total dataset check ---
    train = get_train_dataset()
    test = get_test_dataset()
    # import IPython; IPython.embed()
    print('train', len(train), 'test', len(test))

    convert_to_kdtree = True
    if convert_to_kdtree:
        from chainer.datasets import TransformDataset
        from chainer_pointnet.utils.kdtree import TransformKDTreeCls
        train = TransformDataset(train, TransformKDTreeCls())
        points, split_dims, t = train[1]
        print(points.shape, split_dims.shape, t)
