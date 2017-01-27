import json
import sys

from datasets import opensub_train


if __name__ == '__main__':
    for i, inst in enumerate(opensub_train()):
        print json.dumps(inst.__dict__)
        if i % 100000 == 0:
            print >>sys.stderr, i
