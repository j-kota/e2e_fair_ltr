import numpy as np
import json
import os

try:
    import cPickle as myPickle
except ImportError:
    import pickle as myPickle


def serialize(obj, path, in_json=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            myPickle.dump(obj, file)


def unserialize(path, form=None):
    if form is None:
        form = os.path.basename(path).split(".")[-1]
    if form == "npy":
        return np.load(path)
    elif form == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return myPickle.load(file)

def read_rank_dataset(path):
    with open(path) as file:
        for line in file:
            label, line = line.strip().split(' ', maxsplit=1)
            label = float(label)
            line = dict(list(map(lambda x: x.split(':'), line.split())))
            qid = int(line['qid'])
            features = {int(idx): float(value) for idx, value in line.items() if idx.isdigit()}
            cost = 1.0
            if 'cost' in line:
                cost = float(line['cost'])
            yield label, qid, features, cost