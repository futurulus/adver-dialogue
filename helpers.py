# -*- coding: utf-8 -*-
import numpy as np  # NOQA: for doctest
import theano  # NOQA: for doctest
import theano.tensor as T
from collections import OrderedDict
from lasagne.layers import Layer
from theano.ifelse import ifelse
from theano.printing import Print


class ForgetSizeLayer(Layer):
    '''
    Workaround for lack of support for broadcasting in Lasagne merge layers.

    >>> from lasagne.layers import InputLayer, ElemwiseMergeLayer, dimshuffle
    >>> l_in = InputLayer((100, 20))
    >>> l_w = InputLayer((100, 20, 5))
    >>> l_broadcast = dimshuffle(l_in, (0, 1, 'x'))
    >>> l_forget = ForgetSizeLayer(l_broadcast, axis=2)
    >>> l_merge = ElemwiseMergeLayer((l_forget, l_w), T.mul)
    >>> l_merge.output_shape
    (100, 20, 5)

    https://github.com/Lasagne/Lasagne/issues/584#issuecomment-174573736
    '''
    def __init__(self, incoming, axis=-1, **kwargs):
        super(ForgetSizeLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return input

    def get_output_shape_for(self, input_shape, **kwargs):
        shape = list(input_shape)
        shape[self.axis] = None
        return tuple(shape)


class RepeatLayer(Layer):
    '''
    By Søren Sønderby.
    https://github.com/Lasagne/Lasagne/issues/362
    '''
    def __init__(self, incoming, num_repeats, axis=1, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.n = num_repeats
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[:self.axis]) + [self.n] + list(input_shape[self.axis:]))

    def get_output_for(self, input, **kwargs):
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = range(1, self.axis + 1) + [0] + range(self.axis + 1, input.ndim + 1)
        return stacked.dimshuffle(dim)


def batched_dot(x, y):
    '''
    Implements the Theano batched_dot function in a way that should be executable
    on the GPU. As of 0.7.0, the batched_dot function also doesn't compile
    when passed two 3D tensors for some reason--it gives the error:
      File ".../site-packages/theano/scan_module/scan.py", line 557, in scan
        scan_seqs = [seq[:actual_n_steps] for seq in scan_seqs]
      IndexError: failed to coerce slice entry of type TensorVariable to integer
    '''
    dims = range(x.ndim)
    assert x.ndim == y.ndim
    return T.sum(x.dimshuffle(*(dims + ['x'])) * y.dimshuffle(*(dims[:-2] + ['x'] + dims[-2:])),
                 axis=x.ndim - 1)


def apply_nan_suppression(updates, print_mode='all'):
    """Returns a modified update dictionary replacing updates containing
    non-finite values with no-op updates

    If any NaN or infinity values are found in the new_expression (second)
    half of an update, the update is replaced with the do-nothing update
    (shared_variable, shared_variable).

    This can be used to patch over the most intransigent, slippery instances
    of NaNs creeping into training, if they appear rarely and one is reasonably
    sure that the problem is not fundamental to the model.

    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions

    print_mode : str
        If ``'all'``, print a debugging message containing the name of the
        shared variable and its suppressed update value whenever a non-finite
        value is detected. If ``'shape'``, print only the name of the variable
        and the shape of the update value. If ``'none'``, suppress NaNs
        silently without printing anything.

    Returns
    -------
    OrderedDict
        A copy of `updates` with expressions containing non-finite values
        replaced by the original value.

    Examples
    --------
    >>> param = theano.shared(np.array([0., 0.], dtype=np.float32),
    ...                       name='param')
    >>> inc = T.fvector('inc')
    >>> updates = OrderedDict([(param, param + inc)])
    >>> safe_updates = apply_nan_suppression(updates)
    >>> func = theano.function([inc], safe_updates[param],
    ...                        updates=safe_updates)
    >>> func([1., 2.])
    array([ 1.,  2.], dtype=float32)
    >>> func([2., float('nan')])
    Warning: non-finite update suppressed for param: __str__ = [  3.  nan]
    array([ 1.,  2.], dtype=float32)
    """
    new_updates = OrderedDict([])

    for shared_variable, new_expression in updates.iteritems():
        isnan = T.isnan(new_expression).any() | T.isinf(new_expression).any()

        warning_msg = 'Warning: non-finite update suppressed for %s'
        if print_mode == 'all':
            suppressed = T.zeros_like(
                Print((warning_msg + ':') % shared_variable.name)(new_expression)
            )
        elif print_mode == 'shape':
            suppressed = T.zeros_like(
                Print((warning_msg + ':') % shared_variable.name,
                      attrs=('shape',))(new_expression)
            )
        elif print_mode == 'none' or print_mode is None:
            suppressed = T.zeros_like(new_expression)
        else:
            raise ValueError("print_mode must be one of 'all', 'shape', or 'none'")

        # For some reason, the ifelse needs to be used in a calculation, or the
        # Print gets optimized away. So we can't do
        #   suppressed = (zeros_like(Print('warning')(new_expression)) +
        #                 shared_variable)
        #   ifelse(isnan, suppressed, new_expression)
        new_updates[shared_variable] = shared_variable + ifelse(isnan, suppressed,
                                                                new_expression - shared_variable)

    return new_updates


def logit_softmax_nd(axis=1):
    '''
    A n-dimensional generalization of `theano.tensor.nnet.softmax`, in log space.

    Call this once to fill in the axis (optional) and return a nonlinearity function,
    which is then called again with its input to construct a computation graph:

        logit_softmax_nd(axis=2)(log_probs_3d)

    :param axis: the axis over which to compute the softmax.
    '''
    def logit_softmax_fn(logits):
        axis_last = logits.dimshuffle(range(axis) + range(axis + 1, logits.ndim) + [axis])
        logits_flattened = T.reshape(axis_last, (-1, T.shape(axis_last)[-1]))
        logits_shifted = logits_flattened - logits_flattened.max(axis=1, keepdims=True)
        logits_normalized = (logits_shifted -
                             T.log(T.sum(T.exp(logits_shifted), axis=1, keepdims=True)))
        return T.reshape(logits_normalized, logits.shape)

    return logit_softmax_fn
