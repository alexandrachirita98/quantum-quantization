"""pycls model builders, adapted to the evaluation pipeline's `model_classes` map.

pycls (https://github.com/facebookresearch/pycls) does not build models from constructor
arguments: `builders.build_model()` reads a **global** yacs config that `model_zoo.build_model`
populates by downloading the model's YAML. `evaluation.build_model`, on the other hand, expects
`model_classes[name](num_classes=...)`. `pycls_factory` bridges the two — it returns a
`num_classes -> nn.Module` callable that sets up the global config and hands back the network.

Valid names are the keys of pycls' model zoo: RegNetX-{200MF,400MF,600MF,800MF,1.6GF,3.2GF,4.0GF,
6.4GF,8.0GF,12GF,16GF,32GF}, the same set for RegNetY-*, plus ResNet-{50,101,152},
ResNeXt-{50,101,152} and EfficientNet-B{0..5}. `pycls.models.model_zoo.get_model_list()` prints them.
"""
import torch.nn as nn


def pycls_factory(zoo_name, pretrained=False):
    """Return a ``num_classes -> nn.Module`` builder for the pycls zoo entry ``zoo_name``.

    With ``pretrained=False`` the head is sized directly through the config. With
    ``pretrained=True`` the model must first be built with the checkpoint's own 1000-way head —
    pycls loads checkpoints strictly, so a resized head raises a shape mismatch — and the head is
    swapped for a fresh `num_classes` one afterwards.
    """
    def build(num_classes):
        # imported lazily so this module can be imported without pycls on the path
        from pycls.core.config import reset_cfg
        from pycls.models import model_zoo

        # model_zoo.build_model mutates the global cfg; reset first so repeat calls are independent
        reset_cfg()

        if pretrained:
            net = model_zoo.build_model(zoo_name, pretrained=True)
            net.head.fc = nn.Linear(net.head.fc.in_features, num_classes)
        else:
            net = model_zoo.build_model(zoo_name, cfg_list=['MODEL.NUM_CLASSES', int(num_classes)])
        return net

    return build
