from collections import namedtuple

from stanza.research.instance import Instance
from stanza.research.rng import get_rng


rng = get_rng()


def dialogues_to_instances(dialogues):
    return [Instance(input=' '.join(dialogue[max(0, i - 2):i]),
                     output=dialogue[i])
            for dialogue in dialogues
            for i in range(len(dialogue))]


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
}
