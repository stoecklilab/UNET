import torch

class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input_tensor, target):
        """
        :param input_tensor: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input_tensor.dim() == 5

        n_classes = input_tensor.size()[1]

        if target.dim() == 4:
            target = self._expand_as_one_hot(target, n_classes, self.ignore_index)

        assert input_tensor.size() == target.size()

        per_batch_iou = []
        for _input_tensor, _target in zip(input_tensor, target):
            binary_prediction = self._binarize_predictions(_input_tensor, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input_tensor, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input_tensor tensor.
        """
        if n_classes == 1:
            # for single channel input_tensor just threshold the probability map
            result = input_tensor > 0.5
            return result.long()

        _, max_index = torch.max(input_tensor, dim=0, keepdim=True)
        return torch.zeros_like(input_tensor, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)

    def _expand_as_one_hot(self, input_image, channels, ignore_index=None):
        """
        Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
        It is assumed that the batch dimension is present.
        Args:
            input_image (torch.Tensor): 3D/4D input image
            C (int): number of channels/labels
            ignore_index (int): ignore index to be kept during the expansion
        Returns:
            4D/5D output torch.Tensor (NxCxSPATIAL)
        """
        assert input_image.dim() == 4

        # expand the input tensor to Nx1xSPATIAL before scattering
        input_image = input_image.unsqueeze(1)
        # create output tensor shape (NxCxSPATIAL)
        shape = list(input_image.size())
        shape[1] = channels

        if ignore_index is not None:
            # create ignore_index mask for the result
            mask = input_image.expand(shape) == ignore_index
            # clone the src tensor and zero out ignore_index in the input_image
            input_image = input_image.clone()
            input_image[input_image == ignore_index] = 0
            # scatter to get the one-hot tensor
            result = torch.zeros(shape).to(input_image.device).scatter_(1, input_image, 1)
            # bring back the ignore_index in the result
            result[mask] = ignore_index
            return result
        else:
            # scatter to get the one-hot tensor
            return torch.zeros(shape).to(input_image.device).scatter_(1, input_image, 1)
