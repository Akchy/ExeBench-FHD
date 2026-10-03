import json
import os


def atomic_write_json(path, obj, indent=2):
    """Write JSON via a temp file and os.replace so an interrupted run never
    leaves a truncated file behind."""
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp_path, path)
