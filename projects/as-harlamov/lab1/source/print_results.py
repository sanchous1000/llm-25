import json
from itertools import groupby


if __name__ == '__main__':
    with open('../results/ollama_experiments.json') as f:
        data = json.load(f)

    for model, items in groupby(data, key=lambda x: x['model']):
        print('МОДЕЛЬ', model)
        for item in items:
            print(item['response'])
        print()
