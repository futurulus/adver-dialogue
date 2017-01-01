from collections import namedtuple
from itertools import islice

from stanza.research.instance import Instance
from stanza.research.rng import get_rng


rng = get_rng()


def dialogues_to_instances(dialogues, context_len=2):
    return [Instance(input=' '.join(dialogue[max(0, i - context_len):i]),
                     output=dialogue[i])
            for dialogue in dialogues
            for i in range(len(dialogue))]


def dialogue_batches(data_file, dialogue_size=None, context_len=2, start=None, end=None):
    insts = []
    with open(data_file, 'r') as infile:
        batch = []
        for line in islice(infile, start, end):
            batch.append(line)
            if len(batch) >= dialogue_size:
                insts.extend(dialogues_to_instances([batch], context_len=context_len))
                batch = []
        insts.extend(dialogues_to_instances([batch], context_len=context_len))
    return insts


OPENSUB_SIZE = 88134110
OPENSUB_TRAIN_SIZE = int(OPENSUB_SIZE * 0.8)


def opensub_train():
    return dialogue_batches('data/opensub/2012.data',
                            dialogue_size=1000,
                            context_len=2,
                            start=0, end=OPENSUB_TRAIN_SIZE)


def opensub_test():
    return dialogue_batches('data/opensub/2012.data',
                            dialogue_size=1000,
                            context_len=2,
                            start=OPENSUB_TRAIN_SIZE, end=None)


def opensub5k_train():
    return dialogue_batches('data/opensub/2012_5k.data',
                            dialogue_size=500,
                            context_len=2,
                            start=0, end=2500)


def opensub5k_test():
    return dialogue_batches('data/opensub/2012_5k.data',
                            dialogue_size=500,
                            context_len=2,
                            start=2500, end=None)


def toy_train():
    TOY_DIALOGUES_TRAIN = [
        [
            'How are you?',
            'Great, how are you?',
            "I'm good.",
        ],
    ]
    return dialogues_to_instances(TOY_DIALOGUES_TRAIN)


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
    return dialogues_to_instances(TOY_DIALOGUES_TEST)


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'toy': DataSource(toy_train, toy_test),
    '5k': DataSource(opensub5k_train, opensub5k_test),
    'opensub': DataSource(opensub_train, opensub_test),
}
