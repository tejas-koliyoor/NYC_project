import os
import json
import re


def test_online_keys_are_iso_z():
    path = os.path.join('features', 'online', 'online_2025-03.json')
    assert os.path.exists(path), 'Run materialization first.'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pat = re.compile(r'^\d+_\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z_\d+_\d+$')
    for k in data.keys():
        assert pat.match(k), f'Bad key format: {k}'