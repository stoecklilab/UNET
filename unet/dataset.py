import glob
import os
from abc import abstractmethod
from itertools import chain
import collections
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import h5py

import unet.transforms as transforms
from unet.utils import get_logger, get_class

logger = get_logger('Dataset')


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the patch and stride shape.

    Args:
        raw_dataset (ndarray): raw data
        label_dataset (ndarray): ground truth labels
        weight_dataset (ndarray): weights for the labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        kwargs: additional metadata
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs):
        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset, patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x),
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, ignore_index=None,
                 threshold=0.6, slack_acceptance=0.01, **kwargs):
        super().__init__(raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs)
        if label_dataset is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_dataset[label_idx]
            if ignore_index is not None:
                patch = np.copy(patch)
                patch[patch == ignore_index] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        logger.info('Filtering slices...')
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


def _loader_classes(class_name):
    modules = [
        'unet.dataset',
    ]
    return get_class(class_name, modules)


def get_slice_builder(raws, labels, weight_maps, config):
    assert 'name' in config
    logger.info(f"Slice builder config: {config}")
    slice_builder_cls = _loader_classes(config['name'])
    return slice_builder_cls(raws, labels, weight_maps, **config)


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    train_datasets = dataset_class.create_datasets(loaders_config, phase='train')

    val_datasets = dataset_class.create_datasets(loaders_config, phase='val')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, pin_memory=True,
                          num_workers=num_workers)
    }


def get_test_loaders(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase='test')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         collate_fn=collate_fn)


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def calculate_stats(img: np.array, skip: bool = False) -> dict[str, Any]:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    if not skip:
        pmin, pmax, mean, std = np.percentile(img, 1), np.percentile(img, 99.6), np.mean(img), np.std(img)
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {
        'pmin': pmin,
        'pmax': pmax,
        'mean': mean,
        'std': std
    }


def mirror_pad(image, padding_shape):
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert len(padding_shape) == 3, "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode='reflect')


def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]


def _create_padded_indexes(indexes, halo_shape):
    return tuple(slice(index.start, index.stop + 2 * halo) for index, halo in zip(indexes, halo_shape))


def traverse_h5_paths(file_paths):
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to H5 file containing raw data as well as labels and per pixel weights (optional)
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        slice_builder_config (dict): configuration of the SliceBuilder
        transformer_config (dict): data augmentation configuration
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        weight_internal_path (str or list): H5 internal path to the per pixel weights (optional)
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, raw_internal_path='raw',
                 label_internal_path='label', weight_internal_path=None, global_normalization=True):
        assert phase in ['train', 'val', 'test']

        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        self.weight_internal_path = weight_internal_path

        self.halo_shape = slice_builder_config.get('halo_shape', [0, 0, 0])

        if global_normalization:
            logger.info('Calculating mean and std of the raw data...')
            with h5py.File(file_path, 'r') as f:
                raw = f[raw_internal_path][:]
                stats = calculate_stats(raw)
        else:
            stats = calculate_stats(None, True)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()

            if weight_internal_path is not None:
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_transform = None

            self._check_volume_sizes()
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None
            self.weight_map = None

            # compare patch and stride configuration
            patch_shape = slice_builder_config.get('patch_shape')
            stride_shape = slice_builder_config.get('stride_shape')
            if sum(self.halo_shape) != 0 and patch_shape != stride_shape:
                logger.warning(f'Found non-zero halo shape {self.halo_shape}. '
                               f'In this case: patch shape and stride shape should be equal for optimal prediction '
                               f'performance, but found patch_shape: {patch_shape} and stride_shape: {stride_shape}!')

        with h5py.File(file_path, 'r') as f:
            raw = f[raw_internal_path]
            label = f[label_internal_path] if phase != 'test' else None
            weight_map = f[weight_internal_path] if weight_internal_path is not None else None
            # build slice indices for raw and label data sets
            slice_builder = get_slice_builder(raw, label, weight_map, slice_builder_config)
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices
            self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    @abstractmethod
    def get_raw_patch(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_label_patch(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_weight_patch(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_raw_padded_patch(self, idx):
        raise NotImplementedError

    def volume_shape(self):
        with h5py.File(self.file_path, 'r') as f:
            raw = f[self.raw_internal_path]
            if raw.ndim == 3:
                return raw.shape
            else:
                return raw.shape[1:]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        if self.phase == 'test':
            if len(raw_idx) == 4:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                raw_idx = raw_idx[1:]  # Remove the first element if raw_idx has 4 elements
                raw_idx_padded = (slice(None),) + _create_padded_indexes(raw_idx, self.halo_shape)
            else:
                raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

            raw_patch_transformed = self.raw_transform(self.get_raw_padded_patch(raw_idx_padded))
            return raw_patch_transformed, raw_idx
        else:
            raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))

            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.get_label_patch(label_idx))
            if self.weight_internal_path is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.get_weight_patch(weight_idx))
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    def _check_volume_sizes(self):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        with h5py.File(self.file_path, 'r') as f:
            raw = f[self.raw_internal_path]
            label = f[self.label_internal_path]
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'
            if self.weight_internal_path is not None:
                weight_map = f[self.weight_internal_path]
                assert weight_map.ndim in [3, 4], 'Weight map dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
                assert _volume_shape(raw) == _volume_shape(weight_map), 'Raw and weight map have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = traverse_h5_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config,
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
                 global_normalization=True):
        super().__init__(file_path=file_path, phase=phase, slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config, raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path, weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)
        self._raw = None
        self._raw_padded = None
        self._label = None
        self._weight_map = None

    def get_raw_patch(self, idx):
        if self._raw is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.raw_internal_path in f, f'Dataset {self.raw_internal_path} not found in {self.file_path}'
                self._raw = f[self.raw_internal_path][:]
        return self._raw[idx]

    def get_label_patch(self, idx):
        if self._label is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.label_internal_path in f, f'Dataset {self.label_internal_path} not found in {self.file_path}'
                self._label = f[self.label_internal_path][:]
        return self._label[idx]

    def get_weight_patch(self, idx):
        if self._weight_map is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.weight_internal_path in f, f'Dataset {self.weight_internal_path} not found in {self.file_path}'
                self._weight_map = f[self.weight_internal_path][:]
        return self._weight_map[idx]

    def get_raw_padded_patch(self, idx):
        if self._raw_padded is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.raw_internal_path in f, f'Dataset {self.raw_internal_path} not found in {self.file_path}'
                self._raw_padded = mirror_pad(f[self.raw_internal_path][:], self.halo_shape)
        return self._raw_padded[idx]


class LazyHDF5Dataset(AbstractHDF5Dataset):
    """Implementation of the HDF5 dataset which loads the data lazily. It's slower, but has a low memory footprint."""

    def __init__(self, file_path, phase, slice_builder_config, transformer_config,
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
                 global_normalization=False):
        super().__init__(file_path=file_path, phase=phase, slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config, raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path, weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)

        logger.info("Using LazyHDF5Dataset")

    def get_raw_patch(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            return f[self.raw_internal_path][idx]

    def get_label_patch(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            return f[self.label_internal_path][idx]

    def get_weight_patch(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            return f[self.weight_internal_path][idx]

    def get_raw_padded_patch(self, idx):
        with h5py.File(self.file_path, 'r+') as f:
            if 'raw_padded' in f:
                return f['raw_padded'][idx]

            raw = f[self.raw_internal_path][:]
            raw_padded = mirror_pad(raw, self.halo_shape)
            f.create_dataset('raw_padded', data=raw_padded, compression='gzip')
            return raw_padded[idx]
