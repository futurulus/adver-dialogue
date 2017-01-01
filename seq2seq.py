# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as T
import warnings
from theano.tensor.nnet import crossentropy_categorical_1hot
from lasagne.layers import InputLayer, DropoutLayer, EmbeddingLayer
from lasagne.layers import ReshapeLayer, DenseLayer, get_output
from lasagne.layers.recurrent import Gate
from lasagne.init import Constant
from lasagne.nonlinearities import softmax

from stanza.monitoring import progress
from stanza.research import config, iterators

from tokenizers import TOKENIZERS
from vectorizers import SequenceVectorizer, strip_invalid_tokens
from neural import NeuralLearner, SimpleLasagneModel, sample
from neural import OPTIMIZERS, NONLINEARITIES, CELLS

parser = config.get_options_parser()
parser.add_argument('--cell_size', type=int, default=20,
                    help='The number of dimensions of all hidden layers and cells in '
                         'the speaker model. If 0 and using the AtomicSpeakerLearner, '
                         'remove all hidden layers and only train a linear classifier.')
parser.add_argument('--forget_bias', type=float, default=5.0,
                    help='The initial value of the forget gate bias in LSTM cells in '
                         'the speaker model. A positive initial forget gate bias '
                         'encourages the model to remember everything by default. '
                         'If cell is not LSTM, this value is ignored.')
parser.add_argument('--nonlinearity', choices=NONLINEARITIES.keys(), default='tanh',
                    help='The nonlinearity/activation function to use for dense and '
                         'recurrent layers in the speaker model.')
parser.add_argument('--cell', choices=CELLS.keys(), default='LSTM',
                    help='The recurrent cell to use for the speaker model.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='The dropout rate (probability of setting a value to zero). '
                         'Dropout will be disabled if nonpositive.')
parser.add_argument('--use_mask', type=config.boolean, default=True,
                    help='If `True`, use masking of sequence outputs in training.')
parser.add_argument('--use_input_mask', type=config.boolean, default=False,
                    help='If `True`, use masking of sequence inputs in training.')
parser.add_argument('--recurrent_layers', type=int, default=2,
                    help='The number of recurrent layers to pass the input through.')
parser.add_argument('--hidden_out_layers', type=int, default=0,
                    help='The number of dense layers to pass activations through '
                         'before the output.')
parser.add_argument('--eval_batch_size', type=int, default=16384,
                    help='The number of examples per batch for evaluating the speaker '
                         'model. Higher means faster but more memory usage. This should '
                         'not affect modeling accuracy.')
parser.add_argument('--beam_size', type=int, default=1,
                    help='The number of choices to keep in memory at each time step '
                         'during prediction. Only used for recurrent speakers.')
parser.add_argument('--optimizer', choices=OPTIMIZERS.keys(), default='rmsprop',
                    help='The optimization (update) algorithm to use for speaker training.')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='The learning rate to use for speaker training.')
parser.add_argument('--tokenizer', choices=TOKENIZERS.keys(), default='unigram',
                    help='The tokenization/preprocessing method to use for the speaker model.')
parser.add_argument('--unk_threshold', type=int, default=0,
                    help="The maximum number of occurrences of a token in the training data "
                         "before it's assigned a non-<unk> token index. 0 means nothing in "
                         "the training data is to be treated as unknown words; 1 means "
                         "single-occurrence words (hapax legomena) will be replaced with <unk>.")


class Seq2SeqLearner(NeuralLearner):
    '''
    An agent that uses a basic recurrent sequence-to-sequence model to generate dialogue
    responses.
    '''
    def __init__(self, id=None, context_len=1):
        super(Seq2SeqLearner, self).__init__(id=id)
        self.seq_vec = SequenceVectorizer(unk_threshold=self.options.unk_threshold)

    def predict(self, eval_instances, random=False, verbosity=0):
        result = []
        batches = iterators.iter_batches(eval_instances, self.options.eval_batch_size)
        num_batches = (len(eval_instances) - 1) // self.options.eval_batch_size + 1

        eos_index = self.seq_vec.vectorize(['</s>'])[0]

        tokenize, detokenize = TOKENIZERS[self.options.tokenizer]

        if self.options.verbosity + verbosity >= 2:
            print('Predicting')
        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Predict batch', num_batches)
        for batch_num, batch in enumerate(batches):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(batch_num)
            batch = list(batch)

            if self.options.use_input_mask:
                (x, xm, _p, mask), (_y,) = self._data_to_arrays(batch, test=True)
            else:
                (x, _p, mask), (_y,) = self._data_to_arrays(batch, test=True)
            assert mask.all()  # We shouldn't be masking anything in prediction

            beam_size = 1 if random else self.options.beam_size
            done = np.zeros((len(batch), beam_size), dtype=np.bool)
            beam = np.zeros((len(batch), beam_size, self.seq_vec.max_len),
                            dtype=np.int32)
            beam[:, :, 0] = self.seq_vec.vectorize(['<s>'])[0]
            beam_scores = np.log(np.zeros((len(batch), beam_size)))
            beam_scores[:, 0] = 0.0

            x = np.repeat(x, beam_size, axis=0)
            mask = np.repeat(mask, beam_size, axis=0)
            if self.options.use_input_mask:
                xm = np.repeat(xm, beam_size, axis=0)

            for length in range(1, self.seq_vec.max_len):
                if done.all():
                    break
                p = beam.reshape((beam.shape[0] * beam.shape[1], beam.shape[2]))[:, :-1]
                inputs = [x, xm, p, mask] if self.options.use_input_mask else [x, p, mask]
                probs = self.model.predict(inputs)
                if random:
                    indices = sample(probs[:, length - 1, :])
                    beam[:, 0, length] = indices
                    done = np.logical_or(done, indices == eos_index)
                else:
                    assert probs.shape[1] == p.shape[1], (probs.shape[1], p.shape[1])
                    assert probs.shape[2] == len(self.seq_vec.tokens), (probs.shape[2],
                                                                        len(self.seq_vec.tokens))
                    scores = np.log(probs)[:, length - 1, :].reshape((beam.shape[0], beam.shape[1],
                                                                      probs.shape[2]))
                    beam_search_step(scores, length, beam, beam_scores, done, eos_index)
            outputs = self.seq_vec.unvectorize_all(beam[:, 0, :])
            result.extend([detokenize(strip_invalid_tokens(o)) for o in outputs])
        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return result

    def score(self, eval_instances, verbosity=0):
        result = []
        batches = iterators.iter_batches(eval_instances, self.options.eval_batch_size)
        num_batches = (len(eval_instances) - 1) // self.options.eval_batch_size + 1

        if self.options.verbosity + verbosity >= 2:
            print('Scoring')
        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Score batch', num_batches)
        for batch_num, batch in enumerate(batches):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(batch_num)
            batch = list(batch)

            xs, (n,) = self._data_to_arrays(batch, test=False)
            if self.options.use_input_mask:
                mask = xs[3]
            else:
                mask = xs[2]

            probs = self.model.predict(xs)
            token_probs = probs[np.arange(probs.shape[0])[:, np.newaxis],
                                np.arange(probs.shape[1]), n]
            scores_arr = np.sum(np.log(token_probs) * mask, axis=1)
            scores = scores_arr.tolist()
            result.extend(scores)
        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return result

    def _data_to_arrays(self, training_instances, init_vectorizer=False, test=False):
        tokenize, detokenize = TOKENIZERS[self.options.tokenizer]

        if init_vectorizer:
            tokenized = [['<s>'] + tokenize(inst.output) + ['</s>']
                         for inst in training_instances]
            self.seq_vec.add_all(tokenized)
            unk_replaced = self.seq_vec.unk_replace_all(tokenized)
            config.dump(unk_replaced, 'unk_replaced.train.jsons', lines=True)

        inputs = []
        previous = []
        next_tokens = []
        if self.options.verbosity >= 9:
            print('%s _data_to_arrays:' % self.id)
        maxlen = self.seq_vec.max_len
        for i, inst in enumerate(training_instances):
            x_padded = pad_sequence(tokenize(inst.input), maxlen, left=True, pad='<s>')
            y_padded = (pad_sequence([], maxlen + 1, pad='</s>')
                        if test else
                        pad_sequence(tokenize(inst.output), maxlen + 1))
            prev = y_padded[:-1]
            next = y_padded[1:]
            if self.options.verbosity >= 9:
                print('%s, %s -> %s' % (repr(x_padded), repr(prev), repr(next)))
            inputs.append(x_padded)
            previous.append(prev)
            next_tokens.append(next)

        X = np.zeros((len(inputs), self.seq_vec.max_len - 1), dtype=np.int32)
        P = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        mask = np.zeros((len(previous), self.seq_vec.max_len - 1), dtype=np.int32)
        N = np.zeros((len(next_tokens), self.seq_vec.max_len - 1), dtype=np.int32)
        for i, (inp, prev, next) in enumerate(zip(inputs, previous, next_tokens)):
            if len(inp) > X.shape[1]:
                inp = inp[:X.shape[1]]
            if len(prev) > P.shape[1]:
                prev = prev[:P.shape[1]]
            if len(next) > N.shape[1]:
                next = next[:N.shape[1]]
            X[i, :len(inp)] = self.seq_vec.vectorize(inp)
            P[i, :len(prev)] = self.seq_vec.vectorize(prev)
            N[i, :len(next)] = self.seq_vec.vectorize(next)
            for t, token in enumerate(next):
                mask[i, t] = (token != '<MASK>')

        if self.options.verbosity >= 9:
            print('X: %s' % (repr(X),))
            print('P: %s' % (repr(P),))
            print('mask: %s' % (repr(mask),))
            print('N: %s' % (repr(N),))
        return [X, P, mask], [N]

    def _build_model(self, model_class=SimpleLasagneModel):
        id_tag = (self.id + '/') if self.id else ''

        input_vars = [
            T.imatrix(id_tag + 'input'),
            T.imatrix(id_tag + 'previous'),
            T.imatrix(id_tag + 'mask')
        ]
        target_var = T.imatrix(id_tag + 'targets')

        self.l_out, self.input_layers = self._get_l_out(input_vars)
        self.model = model_class(input_vars, [target_var], self.l_out, id=self.id,
                                 loss=self.masked_loss(input_vars),
                                 optimizer=OPTIMIZERS[self.options.optimizer],
                                 learning_rate=self.options.learning_rate)

    def _get_l_out(self, input_vars):
        check_options(self.options)
        id_tag = (self.id + '/') if self.id else ''

        input_var, prev_output_var, mask_var = input_vars

        l_in = InputLayer(shape=(None, self.seq_vec.max_len), input_var=input_var,
                          name=id_tag + 'desc_input')
        l_in_embed = EmbeddingLayer(l_in, input_size=len(self.seq_vec.tokens),
                                    output_size=self.options.cell_size,
                                    name=id_tag + 'desc_embed')
        l_enc_drop = l_in_embed

        cell = CELLS[self.options.cell]
        cell_kwargs = {
            'grad_clipping': self.options.grad_clipping,
            'num_units': self.options.cell_size,
        }
        if self.options.cell == 'LSTM':
            cell_kwargs['forgetgate'] = Gate(b=Constant(self.options.forget_bias))
        if self.options.cell != 'GRU':
            cell_kwargs['nonlinearity'] = NONLINEARITIES[self.options.nonlinearity]

        for i in range(1, self.options.recurrent_layers):
            l_enc = cell(l_enc_drop, name=id_tag + 'enc%d' % i, **cell_kwargs)
            if self.options.dropout > 0.0:
                l_enc_drop = DropoutLayer(l_enc, p=self.options.dropout,
                                          name=id_tag + 'enc%d_drop' % i)
            else:
                l_enc_drop = l_enc
        l_encoded = cell(l_enc_drop, only_return_final=True,
                         name=id_tag + 'enc%d' % self.options.recurrent_layers,
                         **cell_kwargs)

        l_prev_out = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                                input_var=prev_output_var,
                                name=id_tag + 'prev_input')
        l_prev_embed = EmbeddingLayer(l_prev_out, input_size=len(self.seq_vec.tokens),
                                      output_size=self.options.cell_size,
                                      name=id_tag + 'prev_embed')
        l_dec_drop = l_prev_embed
        l_mask_in = InputLayer(shape=(None, self.seq_vec.max_len - 1),
                               input_var=mask_var, name=id_tag + 'mask_input')

        cell_kwargs['mask_input'] = (l_mask_in if self.options.use_mask else None)
        cell_kwargs['hid_init'] = l_encoded

        for i in range(1, self.options.recurrent_layers):
            l_dec = cell(l_dec_drop, name=id_tag + 'dec%d' % i, **cell_kwargs)
            if self.options.dropout > 0.0:
                l_dec_drop = DropoutLayer(l_dec, p=self.options.dropout,
                                          name=id_tag + 'dec%d_drop' % i)
            else:
                l_dec_drop = l_dec
        l_dec = cell(l_dec_drop,
                     name=id_tag + 'dec%d' % self.options.recurrent_layers,
                     **cell_kwargs)
        l_shape = ReshapeLayer(l_dec, (-1, self.options.cell_size),
                               name=id_tag + 'reshape')
        l_hidden_out = l_shape
        for i in range(1, self.options.hidden_out_layers + 1):
            l_hidden_out = DenseLayer(
                l_hidden_out, num_units=self.options.cell_size,
                nonlinearity=NONLINEARITIES[self.options.nonlinearity],
                name=id_tag + 'hidden_out%d' % i)
        l_softmax = DenseLayer(l_hidden_out, num_units=len(self.seq_vec.tokens),
                               nonlinearity=softmax, name=id_tag + 'softmax')
        l_out = ReshapeLayer(l_softmax, (-1, self.seq_vec.max_len - 1, len(self.seq_vec.tokens)),
                             name=id_tag + 'out')

        return l_out, [l_in, l_prev_out, l_mask_in]

    def loss_out(self, input_vars=None, target_var=None):
        if input_vars is None:
            input_vars = self.model.input_vars
        if target_var is None:
            target_var = self.model.target_var
        pred = get_output(self.l_out, dict(zip(self.input_layers, input_vars)))
        loss = self.masked_loss(input_vars)
        return loss(pred, target_var)

    def masked_loss(self, input_vars):
        return masked_seq_crossentropy(input_vars[-1])

    def sample_prior_smooth(self, num_samples):
        return self.prior_smooth.sample(num_samples)


def check_options(options):
    if options.recurrent_layers and not options.grad_clipping:
        warnings.warn('Norm-constraint gradient clipping is disabled for a recurrent model. '
                      'This will likely lead to exploding gradients.')
    if options.recurrent_layers and options.grad_clipping > 6.0:
        warnings.warn('Gradient clipping norm is unusually high (%s). '
                      'This could lead to exploding gradients.' % options.grad_clipping)
    if options.nonlinearity == 'rectify':
        warnings.warn('Using ReLU as the output nonlinearity for a recurrent unit. This may '
                      'be a source of NaNs in the gradient.')


def crossentropy_categorical_1hot_nd(coding_dist, true_idx):
    '''
    A n-dimensional generalization of `theano.tensor.nnet.crossentropy_categorical`.

    :param coding_dist: a float tensor with the last dimension equal to the number of categories
    :param true_idx: an integer tensor with one fewer dimension than `coding_dist`, giving the
                     indices of the true targets
    '''
    if coding_dist.ndim != true_idx.ndim + 1:
        raise ValueError('`coding_dist` must have one more dimension that `true_idx` '
                         '(got %s and %s)' % (coding_dist.type, true_idx.type))
    coding_flattened = T.reshape(coding_dist, (-1, T.shape(coding_dist)[-1]))
    scores_flattened = crossentropy_categorical_1hot(coding_flattened, true_idx.flatten())
    return T.reshape(scores_flattened, true_idx.shape)


def masked_seq_crossentropy(mask):
    '''
    Return a loss function for sequence models.

    :param mask: a 2-D int tensor (num_examples x max_length) with 1 in valid token locations
        and 0 in locations that should be masked out

    The returned function will have the following parameters and return type:

    :param coding_dist: a 3-D float tensor (num_examples x max_length x num_token_types)
        of log probabilities assigned to each token
    :param true_idx: a 2-D int tensor (num_examples x max_length) of true token indices
    :return: a 1-D float tensor of per-example cross-entropy values
    '''
    def msxe_loss(coding_dist, true_idx):
        mask_float = T.cast(mask, 'float32')
        return (crossentropy_categorical_1hot_nd(coding_dist, true_idx) * mask_float).sum(axis=1)

    return msxe_loss


def beam_search_step(scores, length, beam, beam_scores, done, eos_index):
    '''
    Perform one step of beam search, given the matrix of probabilities
    for each possible following token.

    Modifies `beam`, `beam_scores`, and `done` *in place*.

    :param scores: Scores (log probabilities, up to a constant) assigned by the
        model to each token for each sequence on the various beams.
    :type scores: float ndarray, shape `(batch_size, beam_size, vocab_size)`
    :param int length: Current length of already predicted sequences.
        Should equal the axis-1 index in `beam` where the next
        predicted tokens will be populated.
    :param beam: Token indices for the top-k sequences predicted for each
        example.
    :type beam: int ndarray, shape `(batch_size, beam_size, max_seq_len)`
    :param beam_scores: log probabilities assigned to current candidate sequences
    :type beam_scores: float ndarray, shape `(batch_size, beam_size)`
    :param done: Mask of beam entries that have reached the &lt;/s&gt; token
    :type done: boolean ndarray, shape `(batch_size, beam_size)`

    As an example, suppose the distribution represented by the model is:

        'a cat': 0.375,
        'cat': 0.25,
        'cat a': 0.125,
        'cat cat': 0.125,
        'a': 0.0625,
        '': 0.03125,
        'a a': 0.03125,

    >>> a_cat,   cat, cat_a, cat_cat,    a,    null,     a_a = \\
    ... [0.375, 0.25, 0.125, 0.125, 0.0625, 0.03125, 0.03125]

    >>> vec = SequenceVectorizer(); vec.add(['<s>', 'a', 'cat', '</s>'])
    >>> vec.vectorize(['<s>', 'a', 'cat', '</s>'])
    array([1, 2, 3, 4], dtype=int32)
    >>> eos_index = vec.vectorize(['</s>'])[0]

    Initialize the beam. Note that -inf should be the initial score
    for all but one item on each beam; if all scores start at 0,
    the beam will be saturated with duplicates of the greedy choice.

    >>> batch_size = 1; beam_size = 2; max_seq_len = 3
    >>> beam = np.zeros((batch_size, beam_size, max_seq_len), dtype=np.int)
    >>> beam_scores = np.log(np.zeros((batch_size, beam_size)))
    >>> beam_scores[:, 0] = 0.0
    >>> done = np.zeros((batch_size, beam_size), dtype=np.bool)

    >>> next_scores = np.log([[[0.0, 0.0,
    ...                         a_cat + a + a_a,
    ...                         cat + cat_cat + cat_a,
    ...                         null]] * 2])
    >>> beam_search_step(next_scores, 0, beam, beam_scores, done, eos_index)
    >>> beam
    array([[[3, 0, 0],
            [2, 0, 0]]])
    >>> np.exp(beam_scores).round(5)
    array([[ 0.5    ,  0.46875]])
    >>> done
    array([[False, False]], dtype=bool)

    Note that 'cat' is the greedy first choice, but 'a cat' will end up
    with a higher score.

    >>> next_scores = np.log([[[0.0, 0.0, cat_a / 0.5, cat_cat / 0.5, cat / 0.5],
    ...                        [0.0, 0.0, a_a / 0.46875, a_cat / 0.46875, a / 0.46875]]])
    >>> beam_search_step(next_scores, 1, beam, beam_scores, done, eos_index)
    >>> beam
    array([[[2, 3, 0],
            [3, 4, 0]]])
    >>> np.exp(beam_scores).round(3)
    array([[ 0.375,  0.25 ]])
    >>> done
    array([[False,  True]], dtype=bool)

    The best sequences have been identified; the score for 'cat' stays constant
    after it reaches the end-of-sentence token, and the beam is padded with
    end-of-sentence tokens regardless of the returned scores.

    >>> next_scores = np.log([[[0.0, 0.0, 0.0, 0.0, 1.0],
    ...                        [0.0, 0.0, 0.25, 0.5, 0.25]]])
    >>> beam_search_step(next_scores, 2, beam, beam_scores, done, eos_index)
    >>> beam
    array([[[2, 3, 4],
            [3, 4, 4]]])
    >>> np.exp(beam_scores).round(3)
    array([[ 0.375,  0.25 ]])
    >>> done
    array([[ True,  True]], dtype=bool)
    '''
    assert len(scores.shape) == 3, scores.shape
    batch_size, beam_size, vocab_size = scores.shape
    assert len(beam.shape) == 3, beam.shape
    assert beam.shape[:2] == (batch_size, beam_size), \
        '%s != (%s, %s, *)' % (beam.shape, batch_size, beam_size)
    max_seq_len = beam.shape[2]
    assert beam_scores.shape == (batch_size, beam_size), \
        '%s != %s' % (beam_scores.shape, (batch_size, beam_size))
    assert done.shape == (batch_size, beam_size), \
        '%s != %s' % (done.shape, (batch_size, beam_size))

    # Compute updated scores
    new_scores = (scores * ~done[:, :, np.newaxis] +
                  beam_scores[:, :, np.newaxis]).reshape((batch_size, beam_size * vocab_size))
    # Get indices of top k scores
    topk = np.argsort(-new_scores)[:, :beam_size]
    # Transform into previous beam indices and new token indices
    rows, new_indices = np.unravel_index(topk, (beam_size, vocab_size))
    assert rows.shape == (batch_size, beam_size), \
        '%s != %s' % (rows.shape, (batch_size, beam_size))
    assert new_indices.shape == (batch_size, beam_size), \
        '%s != %s' % (new_indices.shape, (batch_size, beam_size))

    # Extract best pre-existing rows
    beam[:, :, :] = beam[np.arange(batch_size)[:, np.newaxis], rows, :]
    assert beam.shape == (batch_size, beam_size, max_seq_len), \
        '%s != %s' % (beam.shape, (batch_size, beam_size, max_seq_len))
    # Append new token indices
    beam[:, :, length] = new_indices
    # Update beam scores
    beam_scores[:, :] = new_scores[np.arange(batch_size)[:, np.newaxis], topk]
    # Get previous done status and update it with
    # which rows have newly reached </s>
    done[:, :] = done[np.arange(batch_size)[:, np.newaxis], rows] | (new_indices == eos_index)
    # Pad already-finished sequences with </s>
    beam[done, length] = eos_index


def pad_sequence(sequence, length, left=False, pad='<MASK>'):
    if left:
        return ([pad] * (length - 2 - len(sequence)) + ['<s>'] + sequence + ['</s>'])[-length:]
    else:
        return (['<s>'] + sequence + ['</s>'] + [pad] * (length - 2 - len(sequence)))[:length]
