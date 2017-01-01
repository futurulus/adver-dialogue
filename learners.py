from seq2seq import Seq2SeqLearner


def new(key):
    '''
    Construct a new learner with the class named by `key`. A list
    of available learners is in the dictionary `LEARNERS`.
    '''
    return LEARNERS[key]()


LEARNERS = {
    'Seq2Seq': Seq2SeqLearner,
}
