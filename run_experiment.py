#!/usr/bin/env python
from stanza.research import config
config.redirect_output()

from stanza.cluster import pick_gpu
parser = config.get_options_parser()
parser.add_argument('--device', default=None,
                    help='The device to use in Theano ("cpu" or "gpu[0-n]"). If None, '
                         'pick a free-ish device automatically.')
options, extras = parser.parse_known_args()
if '-h' in extras or '--help' in extras:
    # If user is just asking for the options, don't scare them
    # by saying we're picking a GPU...
    pick_gpu.bind_theano('cpu')
else:
    pick_gpu.bind_theano(options.device)

import datetime
from itertools import islice

from stanza.monitoring import progress
from stanza.research import evaluate, output, iterators

import metrics
import learners
import datasets
from helpers import profile

parser = config.get_options_parser()
parser.add_argument('--learner', default='Seq2Seq', choices=learners.LEARNERS.keys(),
                    help='The name of the model to use in the experiment.')
parser.add_argument('--load', metavar='MODEL_FILE', default=None,
                    help='If provided, skip training and instead load a pretrained model '
                         'from the specified path. If None or an empty string, train a '
                         'new model.')
parser.add_argument('--train_size', type=int, default=None,
                    help='The number of examples to use in training. If None, use the '
                         'whole training set.')
parser.add_argument('--validation_size', type=int, default=None,
                    help="The number of examples to use in validation. If None, use the "
                         "whole validation set. If 0 (or if the data_source doesn't have "
                         "a validation set), validation will be skipped.")
parser.add_argument('--eval_size', type=int, default=None,
                    help='The number of examples to use in testing. '
                         'If None, use the whole dev/test set.')
parser.add_argument('--data_source', default='opensub_dev', choices=datasets.SOURCES.keys(),
                    help='The type of data to use.')
parser.add_argument('--metrics', default=['accuracy', 'perplexity', 'log_likelihood_bits',
                                          'token_perplexity_micro'],
                    choices=metrics.METRICS.keys(),
                    help='The evaluation metrics to report for the experiment.')
parser.add_argument('--output_train_data', type=config.boolean, default=False,
                    help='If True, write out the training dataset (after cutting down to '
                         '`train_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--output_eval_data', type=config.boolean, default=False,
                    help='If True, write out the evaluation dataset (after cutting down to '
                         '`eval_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--progress_tick', type=int, default=10,
                    help='The number of seconds between logging progress updates.')


@profile
def main():
    options = config.options()

    progress.set_resolution(datetime.timedelta(seconds=options.progress_tick))

    source = datasets.SOURCES[options.data_source]
    SG = iterators.SizedGenerator

    train_data = SG(lambda: islice(source.train_data(), 0, options.train_size), length=None)
    if not hasattr(options, 'verbosity') or options.verbosity >= 4:
        print('Training set size: {}'.format(len(train_data)))

    validation_data = None
    if source.validation_data is not None:
        validation_data = SG(lambda: islice(source.validation_data(), 0, options.train_size),
                             length=None)
        if not hasattr(options, 'verbosity') or options.verbosity >= 4:
            print('Validation set size: {}'.format(len(validation_data)))

    eval_data = SG(lambda: islice(source.eval_data(), 0, options.train_size), length=None)
    if not hasattr(options, 'verbosity') or options.verbosity >= 4:
        print('Eval set size: {}'.format(len(eval_data)))

    learner = learners.new(options.learner)

    m = [metrics.METRICS[m] for m in options.metrics]

    if options.load:
        with open(options.load, 'rb') as infile:
            learner.load(infile)
    else:
        learner.train(train_data, validation_data, metrics=m)
        model_path = config.get_file_path('model.pkl')
        if model_path:
            with open(model_path, 'wb') as outfile:
                learner.dump(outfile)

        train_results = evaluate.evaluate(learner, train_data, metrics=m, split_id='train',
                                          write_data=options.output_train_data)
        output.output_results(train_results, 'train')

    eval_results = evaluate.evaluate(learner, eval_data, metrics=m, split_id='eval',
                                     write_data=options.output_eval_data)
    output.output_results(eval_results, 'eval')


if __name__ == '__main__':
    main()
