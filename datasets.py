import gzip
import json
from collections import namedtuple
from itertools import islice

from stanza.research import config
from stanza.research.instance import Instance
from stanza.research.rng import get_rng


rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--train_data_file', type=str, default=None,
                    help='Path to a json file to use as the training dataset. Ignored if '
                         'not using the `file` data source.')
parser.add_argument('--valid_data_file', type=str, default=None,
                    help='Path to a json file to use as the validation dataset. Ignored if '
                         'not using the `file` data source.')
parser.add_argument('--test_data_file', type=str, default=None,
                    help='Path to a json file to use as the evaluation dataset. Ignored if '
                         'not using the `file` data source.')


def instances_from_json_file(filename):
    if not filename:
        return
    openfunc = gzip.open if filename.endswith('.gz') else open
    with openfunc(filename, 'r') as infile:
        for line in infile:
            yield Instance(**json.loads(line.strip()))


def json_file_train():
    options = config.options()
    return instances_from_json_file(options.train_data_file)


def json_file_valid():
    options = config.options()
    return instances_from_json_file(options.valid_data_file)


def json_file_test():
    options = config.options()
    return instances_from_json_file(options.test_data_file)


def lines_to_instances(lines, context_len=2):
    context = [u''] * context_len
    for line in lines:
        try:
            line = line.decode('utf-8').strip()
        except UnicodeDecodeError:
            print(line)
            raise

        yield Instance(input=u' ~ '.join(context), output=line)

        del context[0]
        context.append(line)


def instances_from_file(data_file, context_len=2, start=None, end=None):
    with open(data_file, 'r') as infile:
        for inst in lines_to_instances(islice(infile, start, end), context_len=context_len):
            yield inst


OPENSUB_SIZE = 88134110
OPENSUB_VALIDATION_SIZE = 10000
OPENSUB_TRAIN_SIZE = int(OPENSUB_SIZE * 0.8) - OPENSUB_VALIDATION_SIZE


def opensub_train():
    return instances_from_json_file('data/opensub/opensub_train_shuffled.jsons.gz')


def opensub_train_inorder():
    return instances_from_file('data/opensub/2012.data', context_len=2,
                               start=0, end=OPENSUB_TRAIN_SIZE)


def opensub_valid():
    return instances_from_file('data/opensub/2012.data', context_len=2,
                               start=OPENSUB_TRAIN_SIZE,
                               end=OPENSUB_TRAIN_SIZE + OPENSUB_VALIDATION_SIZE)


def opensub_test():
    return instances_from_file('data/opensub/2012.data', context_len=2,
                               start=OPENSUB_TRAIN_SIZE, end=None)


def opensub5k_train():
    return instances_from_file('data/opensub/2012_5k.data', context_len=2,
                               start=0, end=2500)


def opensub5k_valid():
    return instances_from_file('data/opensub/2012_5k.data', context_len=2,
                               start=2500, end=3000)


def opensub5k_test():
    return instances_from_file('data/opensub/2012_5k.data', context_len=2,
                               start=3000, end=None)


def opensub50k_train():
    return instances_from_file('data/opensub/2012_50k.data', context_len=2,
                               start=0, end=39000)


def opensub50k_valid():
    return instances_from_file('data/opensub/2012_50k.data', context_len=2,
                               start=39000, end=40000)


def opensub50k_test():
    return instances_from_file('data/opensub/2012_50k.data', context_len=2,
                               start=40000, end=None)


def toy_train():
    TOY_DIALOGUES_TRAIN = [
        'How are you?',
        'Great, how are you?',
        "I'm good.",
    ]
    return lines_to_instances(TOY_DIALOGUES_TRAIN)


def toy_test():
    TOY_DIALOGUES_TEST = [
        [
            'How are you?',
            'Great, how are you?',
            'Great, how are you?',
            "I'm good.",
        ],
        [
            'How are you?',
            "I'm good.",
        ],
    ]
    return (
        inst
        for d in TOY_DIALOGUES_TEST
        for inst in lines_to_instances(d)
    )


DataSource = namedtuple('DataSource', ['train_data', 'validation_data', 'eval_data'])

SOURCES = {
    'file': DataSource(json_file_train, json_file_valid, json_file_test),
    'toy': DataSource(toy_train, None, toy_test),
    '5k': DataSource(opensub5k_train, opensub5k_valid, opensub5k_test),
    '50k': DataSource(opensub50k_train, opensub50k_valid, opensub50k_test),
    'opensub': DataSource(opensub_train, opensub_valid, opensub_test),
}
