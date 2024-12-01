import torch.nn.functional as F
from torch import nn, cuda

class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input_tensor, target):
        weight = self._class_weights(input_tensor)
        return F.cross_entropy(input_tensor, target, weight=weight, ignore_index=self.ignore_index)

    def _class_weights(self, input_tensor):
        # normalize the input_tensor first
        input_tensor = F.softmax(input_tensor, dim=1)
        flattened = self._flatten(input_tensor)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = nominator / denominator
        return class_weights.detach()

    def _flatten(self, tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
        """
        # number of channels
        channels = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(channels, -1)

def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    weight = loss_config.pop('weight', None)

    if ignore_index is None:
        ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss

    if name == 'WeightedCrossEntropyLoss':
        loss =  WeightedCrossEntropyLoss(ignore_index=ignore_index)
    else:
        loss =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    if cuda.is_available():
        loss = loss.cuda()

    return loss
