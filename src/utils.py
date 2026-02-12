import json

def save_metrics(metrics, path='metrics.json'):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)