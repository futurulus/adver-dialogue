import argparse
import lasagne
import numpy as np
import os
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as G
import time
import warnings
from collections import Sequence, OrderedDict
from lasagne.layers import get_output, get_all_params
from lasagne.updates import total_norm_constraint
from theano.compile import MonitorMode
from theano.printing import pydotprint

from helpers import apply_nan_suppression, profile
from stanza.monitoring import progress, summary
from stanza.research import config, iterators
from stanza.research.learner import Learner
from stanza.research.rng import get_rng


parser = config.get_options_parser()
parser.add_argument('--train_iters', type=int, default=10,
                    help='Number of iterations')
parser.add_argument('--train_epochs', type=int, default=100,
                    help='Number of epochs per iteration')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of examples per minibatch for training and evaluation')
parser.add_argument('--detect_nans', type=config.boolean, default=False,
                    help='If True, throw an error if a non-finite value is detected.')
parser.add_argument('--verbosity', type=int, default=4,
                    help='Amount of diagnostic output to produce. 0-1: only progress updates; '
                         '2-3: plus major experiment steps; '
                         '4-5: plus compilation and graph assembly steps; '
                         '6: plus parameter names for each function compilation; '
                         '7: plus verbose warnings; '
                         '8: plus shapes and types for each compiled function call; '
                         '9-10: plus vectorization of all datasets')
parser.add_argument('--graphviz', type=config.boolean, default=False,
                    help='If `True`, use theano.printing.pydotprint to visualize '
                         'function graphs.')
parser.add_argument('--nan_suppression', type=config.boolean, default=True,
                    help='If `True`, try to suppress NaNs in training.')
parser.add_argument('--validation_period', type=int, default=10000,
                    help='Number of minibatches to wait between running validation. Use 0 to '
                         'validate only at the end of every iteration. (Note: setting '
                         'validation_size to 0 disables validation.)')
parser.add_argument('--monitor_period', type=int, default=100,
                    help='Number of minibatches to wait between logging monitored tensors.')
parser.add_argument('--monitor_grads', type=config.boolean, default=False,
                    help='If `True`, return gradients for monitoring and write them to the '
                         'TensorBoard events file.')
parser.add_argument('--monitor_params', type=config.boolean, default=False,
                    help='If `True`, write parameter value histograms out to the '
                         'TensorBoard events file.')
parser.add_argument('--monitor_activations', type=config.boolean, default=False,
                    help='If `True`, write activation value histograms (outputs of named'
                         'layers) out to the TensorBoard events file.')
parser.add_argument('--grad_clipping', type=float, default=5.0,
                    help='The maximum absolute value of all gradients. This gradient '
                         'clipping is performed on the full gradient calculation, not '
                         'just the messages passing through the LSTM.')
parser.add_argument('--reset_optimizer_vars', type=config.boolean, default=True,
                    help='If True, reset variables that are not parameters (i.e. variables '
                         'used for the optimizer like Adagrad weights) between training on '
                         'different datasets. Only used if data_source has more than one value.')


NONLINEARITIES = {
    name: func
    for name, func in lasagne.nonlinearities.__dict__.iteritems()
    if name.islower() and not name.startswith('__')
}
del NONLINEARITIES['theano']

OPTIMIZERS = {
    name: func
    for name, func in lasagne.updates.__dict__.iteritems()
    if (name in lasagne.updates.__all__ and
        not name.startswith('apply_') and not name.endswith('_constraint'))
}

CELLS = {
    name[:-len('Layer')]: func
    for name, func in lasagne.layers.recurrent.__dict__.iteritems()
    if (name in lasagne.layers.recurrent.__all__ and name.endswith('Layer') and
        name != 'CustomRecurrentLayer')
}

rng = get_rng()
lasagne.random.set_rng(rng)


def detect_nan(i, node, fn):
    if not isinstance(node.op, (T.AllocEmpty, T.IncSubtensor,
                                G.GpuAllocEmpty, G.GpuIncSubtensor)):
        for output in fn.outputs:
            if (not isinstance(output[0], np.random.RandomState) and
                    not np.isfinite(output[0]).all()):
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                raise AssertionError


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.array(a)
    if len(a.shape) < 1:
        raise ValueError('scalar is not a valid probability distribution')
    elif len(a.shape) == 1:
        # Cast to higher resolution to try to get high-precision normalization
        a = np.exp(np.log(a) / temperature).astype(np.float64)
        a /= np.sum(a)
        return np.argmax(rng.multinomial(1, a, 1))
    else:
        return np.array([sample(s, temperature) for s in a])


class Unpicklable(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<%s removed in pickling>' % (self.name,)


class SimpleLasagneModel(object):
    def __init__(self, input_vars, target_vars, l_out, loss,
                 optimizer, learning_rate=0.001,
                 monitor_period=1, validation_period=1, id=None):
        if not isinstance(input_vars, Sequence):
            raise ValueError('input_vars should be a sequence, instead got %s' % (input_vars,))
        if not isinstance(target_vars, Sequence):
            raise ValueError('target_vars should be a sequence, instead got %s' % (input_vars,))

        self.get_options()

        self.input_vars = input_vars
        self.l_out = l_out
        self.loss = loss
        self.optimizer = optimizer
        self.id = id
        self.monitor_period = monitor_period
        self.validation_period = validation_period
        id_tag = (self.id + '/') if self.id else ''
        id_tag_log = (self.id + ': ') if self.id else ''

        if self.options.verbosity >= 6:
            output_model_structure(l_out)

        params = self.params()
        (monitored,
         train_loss_grads) = self.get_train_loss(target_vars, params)
        self.monitored_tags = monitored.keys()

        if self.options.grad_clipping:
            scaled_grads = total_norm_constraint(train_loss_grads, self.options.grad_clipping)
        else:
            scaled_grads = train_loss_grads

        updates = optimizer(scaled_grads, params, learning_rate=learning_rate)
        self.optimizer_vars = [var for var in updates if var not in params]
        if self.options.nan_suppression:
            # TODO: print_mode='all' somehow is always printing, even when
            # there are no NaNs. But tests are passing, even on GPU!
            updates = apply_nan_suppression(updates, print_mode='none')

        if self.options.detect_nans:
            mode = MonitorMode(post_func=detect_nan)
        else:
            mode = None

        if self.options.verbosity >= 2:
            print(id_tag_log + 'Compiling training function')
        params = input_vars + target_vars
        if self.options.verbosity >= 6:
            print('params = %s' % (params,))
        self.train_fn = theano.function(params, monitored.values(),
                                        updates=updates, mode=mode,
                                        name=id_tag + 'train', on_unused_input='warn')
        if self.options.run_dir and self.options.graphviz:
            self.visualize_graphs({'loss': monitored['loss']},
                                  out_dir=self.options.run_dir)

        test_prediction = get_output(l_out, deterministic=True)
        if self.options.verbosity >= 2:
            print(id_tag_log + 'Compiling prediction function')
        if self.options.verbosity >= 6:
            print('params = %s' % (input_vars,))
        self.predict_fn = theano.function(input_vars, test_prediction, mode=mode,
                                          name=id_tag + 'predict', on_unused_input='ignore')

        if self.options.run_dir and self.options.graphviz:
            self.visualize_graphs({'test_prediction': test_prediction},
                                  out_dir=self.options.run_dir)

    def visualize_graphs(self, monitored, out_dir):
        id_tag = (self.id + '.') if self.id else ''

        for tag, graph in monitored.iteritems():
            tag = tag.replace('/', '.')
            pydotprint(graph, outfile=os.path.join(out_dir, id_tag + tag + '.svg'),
                       format='svg', var_with_name_simple=True)

    def params(self):
        return get_all_params(self.l_out, trainable=True)

    def get_train_loss(self, target_vars, params):
        assert len(target_vars) == 1
        prediction = get_output(self.l_out)
        mean_loss = self.loss(prediction, target_vars[0]).mean()
        monitored = [('loss', mean_loss)]
        grads = T.grad(mean_loss, params)
        if self.options.monitor_grads:
            for p, grad in zip(params, grads):
                monitored.append(('grad/' + p.name, grad))
        if self.options.monitor_activations:
            for name, layer in get_named_layers(self.l_out).iteritems():
                monitored.append(('activation/' + name, get_output(layer)))
        return OrderedDict(monitored), grads

    @profile
    def fit(self, minibatches, num_epochs, summary_writer=None, step=0,
            validate_fn=lambda i: None):
        history = OrderedDict((tag, []) for tag in self.monitored_tags)
        id_tag = (self.id + '/') if self.id else ''
        params = self.params()
        step_offset = 0
        num_examples = 0
        time_history = 0
        history_start = time.time()

        progress.start_task('Epoch', num_epochs)
        for epoch in range(num_epochs):
            progress.progress(epoch)

            history_batch = OrderedDict((tag, []) for tag in self.monitored_tags)
            progress.start_task('Minibatch', len(minibatches))
            for i, batch in enumerate(minibatches):
                progress.progress(i)
                if self.options.verbosity >= 8:
                    print('types: %s' % ([type(v) for t in batch for v in t],))
                    print('shapes: %s' % ([v.shape for t in batch for v in t],))
                inputs, targets = batch
                num_examples += targets[0].shape[0]
                monitored = self.train_fn(*inputs + targets)
                for tag, value in zip(self.monitored_tags, monitored):
                    if self.options.verbosity >= 10:
                        print('%s: %s' % (tag, value))
                    history_batch[tag].append(value)

                if (step_offset + 1) % self.monitor_period == 0:
                    for tag, values in history_batch.items():
                        values_array = np.array([np.asarray(v) for v in values])
                        history[tag].append(values_array)
                        mean_values = np.mean(values_array, axis=0)
                        if len(mean_values.shape) == 0:
                            summary_writer.log_scalar(step + step_offset, tag, mean_values)
                        else:
                            summary_writer.log_histogram(step + step_offset, tag, mean_values)

                    if self.options.monitor_params:
                        for param in params:
                            val = param.get_value()
                            tag = 'param/' + param.name
                            if len(val.shape) == 0:
                                summary_writer.log_scalar(step + step_offset, tag, val)
                            else:
                                summary_writer.log_histogram(step + step_offset, tag, val)

                    history_end = time.time()
                    examples_per_sec = num_examples / (history_end - history_start)
                    num_examples = 0
                    summary_writer.log_scalar(step + step_offset,
                                              id_tag + 'examples_per_sec', examples_per_sec)
                    history_start = history_end

                if self.validation_period and (step_offset + 1) % self.validation_period == 0:
                    validate_iter = (step_offset + 1) / self.validation_period
                    validate_fn(validate_iter)

                step_offset += 1
            progress.end_task()
        progress.end_task()

        if not self.validation_period:
            validate_iter = step / step_offset
            validate_fn(validate_iter)

    def predict(self, Xs):
        if not isinstance(Xs, Sequence):
            raise ValueError('Xs should be a sequence, instead got %s' % (Xs,))
        id_tag_log = (self.id + ': ') if self.id else ''
        if self.options.verbosity >= 8:
            print(id_tag_log + 'predict shapes: %s' % [x.shape for x in Xs])
        return self.predict_fn(*Xs)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['loss'] = Unpicklable('loss')
        state['optimizer'] = Unpicklable('optimizer')
        state['l_out'] = Unpicklable('l_out')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.get_options()

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)

    def reset_optimizer(self):
        if not hasattr(self, 'optimizer_vars'):
            # Probably loaded from older pickle file, in which case the optimizer
            # will typically have been reset anyway (only real parameters are pickled)
            return

        for var in self.optimizer_vars:
            # Lasagne optimizer variables are nameless, as of 26 Aug 2016; most
            # real parameters have names.
            assert var.name is None, var.name
            val = var.get_value()
            var.set_value(np.zeros(val.shape, dtype=val.dtype))


def get_named_layers(layer, id_map=None):
    if id_map is None:
        id_map = {}

    if layer.name:
        id_map[layer.name] = layer
    if hasattr(layer, 'input_layers'):
        for inp in layer.input_layers:
            get_named_layers(inp, id_map)
    elif hasattr(layer, 'input_layer'):
        get_named_layers(layer.input_layer, id_map)

    return id_map


def output_model_structure(layer, indent=0):
    print('%s%s %s' % ('  ' * indent, layer.name, type(layer)))
    if hasattr(layer, 'input_layers'):
        for inp in layer.input_layers:
            output_model_structure(inp, indent=indent + 1)
    elif hasattr(layer, 'input_layer'):
        output_model_structure(layer.input_layer, indent=indent + 1)


class NeuralLearner(Learner):
    '''
    A base class for Lasagne-based learners.
    '''

    def __init__(self, id=None):
        super(NeuralLearner, self).__init__()
        self.id = id
        self.get_options()

    @profile
    def train(self, training_instances, validation_instances=None, metrics=None,
              keep_params=False):
        id_tag = (self.id + ': ') if self.id else ''

        if not hasattr(self, 'model'):
            self.init_vectorizers(training_instances)
        if not hasattr(self, 'model') or not keep_params:
            if self.options.verbosity >= 2:
                print(id_tag + 'Building model')
            if keep_params:
                warnings.warn("keep_params was passed, but the model hasn't been built; "
                              "initializing all parameters.")
            self.build_model()
        else:
            if not hasattr(self.options, 'reset_optimizer_vars') or \
                    self.options.reset_optimizer_vars:
                if self.options.verbosity >= 2:
                    print(id_tag + 'Resetting optimizer')
                self.model.reset_optimizer()

        if self.options.verbosity >= 2:
            print(id_tag + 'Training conditional model')
        if hasattr(self, 'writer'):
            writer = self.writer
        else:
            summary_path = config.get_file_path('losses.tfevents')
            if summary_path:
                writer = summary.SummaryWriter(summary_path)
            else:
                writer = None
            self.writer = writer

        if not hasattr(self, 'step_base'):
            self.step_base = 0

        minibatches = iterators.sized_imap(self.data_to_arrays,
                                           iterators.gen_batches(training_instances,
                                                                 self.options.batch_size))
        progress.start_task('Iteration', self.options.train_iters)
        for iteration in range(self.options.train_iters):
            progress.progress(iteration)
            self.model.fit(minibatches,  # batch_size=self.options.batch_size,
                           num_epochs=self.options.train_epochs,
                           summary_writer=writer,
                           step=self.step_base + iteration * self.options.train_epochs,
                           validate_fn=lambda i: self.validate_and_log(validation_instances,
                                                                       metrics, writer,
                                                                       iteration=i))

        self.step_base += self.options.train_iters * self.options.train_epochs
        writer.flush()
        progress.end_task()

    def validate_and_log(self, validation_instances, metrics, writer, iteration):
        validation_results = self.validate(validation_instances, metrics, iteration=iteration)
        if writer is not None:
            step = self.step_base + (iteration + 1) * self.options.validation_period
            self.on_iter_end(step, writer)
            for key, value in validation_results.iteritems():
                tag = 'val/' + key.split('.', 1)[1].replace('.', '/')
                writer.log_scalar(step, tag, value)

    def on_iter_end(self, step, writer):
        pass

    def params(self):
        return self.model.params()

    @property
    def num_params(self):
        if hasattr(self, 'quickpickle_numparams'):
            return self.quickpickle_numparams
        all_params = self.params()
        return sum(np.prod(p.get_value().shape) for p in all_params)

    def sample(self, inputs):
        return self.predict(inputs, random=True, verbosity=-6)

    def loss_out(self, input_vars=None, target_var=None):
        if input_vars is None:
            input_vars = self.model.input_vars
        if target_var is None:
            target_var = self.model.target_var
        pred = get_output(self.l_out, dict(zip(self.input_layers, input_vars)))
        return self.loss(pred, target_var)

    def __getstate__(self):
        if not hasattr(self, 'model'):
            raise RuntimeError("trying to pickle a model that hasn't been built yet")
        state = dict(self.__dict__)
        state['params_state'] = [p.get_value() for p in self.params()]
        state['model'] = Unpicklable('model')
        state['l_out'] = Unpicklable('l_out')
        state['input_layers'] = Unpicklable('input_layers')
        return state

    def __setstate__(self, state):
        self.unpickle(state)

    def unpickle(self, state, model_class=SimpleLasagneModel):
        self.__dict__.update(state)
        self.get_options()

        if 'quickpickle' in state and state['quickpickle']:
            return

        params_state = state['params_state']
        del self.params_state
        self.build_model(model_class)
        params = self.params()
        assert len(params) == len(params_state), '%d != %d' % (len(params), len(params_state))
        for p, value in zip(params, params_state):
            p.set_value(value)

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)

    def init_vectorizers(self, training_instances):
        '''
        Receives the entire training set. Override in subclasses to set up vocab indices, etc.
        '''
        pass

    def data_to_arrays(self, instances, test=False):
        '''
        Actually perform vectorization. Implement in subclasses.
        '''
        raise NotImplementedError

    def build_model(self, model_class=SimpleLasagneModel):
        '''
        Override in subclasses to specify the model structure. Should set `self.model`
        (an instance of `model_class`), `self.l_out` (a layer object of the NN library),
        and `self.input_layers` (a sequence of layer objects).
        '''
        raise NotImplementedError
