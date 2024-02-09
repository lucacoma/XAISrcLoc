import torch
from zennit.canonizers import  SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from zennit.layer import Sum
from network_lib import ReSE2Block as RESE, SqueezeAndExcitationBlock as SQUEEZE
from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat, WSquare, Gamma, AlphaBeta
from zennit.composites import NameMapComposite
n_mic = 16

"""
Composite for LRP
N.B. Since the gradient is only overwritten by Rules, the gradient will be unchanged for layers without applicable rules. 
If layers should only pass their received gradient/relevance on, the :py:class:`~zennit.rules.Pass` 
rule should be used (which is done for all activations in all LRP composites,...) 
 the :py:class:`~zennit.rules.Norm` rule, which normalizes the gradient by output fraction, 
 is used for :py:class:`~zennit.layer.Sum` and :py:class:`~zennit.types.AvgPool` layers 

cfr 
https://github.com/chr5tphr/zennit/blob/60a2c088a29fb0a68ed63f596858d0b9becff374/docs/source/how-to/use-rules-composites-and-canonizers.rst#L357
"""
# LRP-COMPOSITE
name_map_loc_cnn = [(['conv1'], WSquare()),
            (['conv2'], Gamma()),
            (['conv3'], Gamma()),
            (['conv4'], Gamma()),
            (['conv5'], Gamma()),
            (['pool1'], Norm()),
            (['pool2'], Norm()),
            (['pool3'], Norm()),
            (['fc1'], Epsilon()),
            (['fc2'], Epsilon()),
            (['dr1'], Pass()),
            (['activation1'], Pass()),
            (['activation2'], Pass()),
            (['activation3'], Pass()),
            (['activation4'], Pass()),
            (['activation5'], Pass()),
            (['activation6'], Pass()),
            ]

composite_loc_cnn = NameMapComposite(
    name_map=name_map_loc_cnn,
)

# SAmpleCNN - Canonizer
class ReSE2BlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, RESE):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        x = self.BasicBlock(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x_se = self.SEBlock(x)
        x = torch.stack([x, x_se], dim=-1)
        x = self.canonizer_sum(x)
        x = self.activation(x)
        x = self.pool1(x)
        return x

class Mul(torch.nn.Module):
    '''Compute the sum along an axis.

    Parameters
    ----------
    dim : int
        Dimension over which to sum.
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        '''Computes the sum along a dimension.'''
        return torch.multiply(input, dim=self.dim)
# From https://github.com/frederikpahde/xai-canonization/blob/5147002d20aeb2eefbb2b45394afa0fc8734c0a0/quantus_evaluation/canonizers/efficientnet.py
class SignalOnlyGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x1,x2):
        return x1*x2

    @staticmethod
    def backward(ctx,grad_output):
        return torch.zeros_like(grad_output), grad_output

class SqueezeAndExcitationBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, SQUEEZE):
            attributes = {
                'forward': cls.forward.__get__(module),
                'fn_gate': SignalOnlyGate(),
            }
            return attributes

        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        x_s = self.globalAvgPooling(x)
        x_s = x_s.squeeze(-1)  # Remove time dimension
        x_s = self.dense1(x_s)
        x_s = self.activation1(x_s)
        x_s = self.dense2(x_s)
        x_s = self.activation2(x_s)
        x_s = x_s.unsqueeze(-1)  # Add again time dimension to perform excitation
        x_se = self.fn_gate.apply(x, x_s)
        return x_se

class SampleCNNCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ReSE2BlockCanonizer(),
            SqueezeAndExcitationBlockCanonizer(),
        ))




# Sample-CNN composite
from zennit.types import Convolution, Linear, AvgPool, Activation, BatchNorm
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.composites import layer_map_base

class EpsilonPlusWsquare(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`. This will be prepended to the ``first_map``
        defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(
        self, epsilon=1e-6, stabilizer=1e-6, layer_map=None, first_map=None, zero_params=None, canonizers=None
    ):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, Gamma(stabilizer=stabilizer, **rule_kwargs)),
            (Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
            (AvgPool, Norm()),
        ]
        first_map = first_map + [
            (Convolution, WSquare(stabilizer=stabilizer, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)
from zennit.composites import EpsilonPlusFlat
composite_sample_cnn = EpsilonPlusWsquare(canonizers=[SampleCNNCanonizer()])