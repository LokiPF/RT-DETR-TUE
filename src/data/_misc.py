"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata
from torch import Tensor 

# The three branches below keep the *detector* training path importable on torchvision
# 0.15.2, 0.16 and 0.17, where the v2 transforms and tv_tensors lived under different
# names. They are not the supported floor for this repository: the scene-uncertainty
# blur loader (`src/scene_uncertainty/dataset.py`) hands `SanitizeBoundingBoxes` a
# `labels_getter` that returns a tuple of per-annotation tensors, and that transform
# only accepts a tuple/list from torchvision 0.18.0 onwards. `requirements.txt`
# therefore declares `torchvision>=0.18.0`, and an environment built from it always
# takes the last branch; the earlier two are kept only for pre-existing detector-only
# installations.

if importlib.metadata.version('torchvision').startswith('0.15.2'):
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError(
        'Please make sure torchvision version >= 0.15.2 for the detector, '
        'or >= 0.18.0 for the scene uncertainty pipeline (see requirements.txt)'
    )



def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"
    
    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
       return Mask(tensor)

