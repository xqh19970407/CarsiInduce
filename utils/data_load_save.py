import json


def load_json(path):
    with open(path, 'r') as f:
        dict_data = json.load(f)
    return dict_data