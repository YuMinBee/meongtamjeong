"""Read-only availability check; never log credentials or download data bodies."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
from datetime import datetime, timezone

import requests


def probe(url):
    try:
        with requests.head(url, timeout=25, allow_redirects=True) as response:
            return {
                'url': url,
                'status': response.status_code,
                'content_type': response.headers.get('Content-Type'),
                'content_length': response.headers.get('Content-Length'),
                'etag': response.headers.get('ETag'),
            }
    except requests.RequestException as exc:
        return {'url': url, 'error_type': type(exc).__name__}


def main():
    urls = [
        'https://www.kaggle.com/api/v1/competitions/data/list/petfinder-adoption-prediction',
        'https://storage.googleapis.com/petfinder_dataset/train.csv',
        'https://storage.googleapis.com/petfinder_dataset/train_images.zip',
        'https://storage.googleapis.com/petfinder_dataset/test.csv',
        'https://storage.googleapis.com/petfinder_dataset/test_images.zip',
    ]
    with ThreadPoolExecutor(max_workers=5) as pool:
        probes = list(pool.map(probe, urls))
    report = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'HEAD only; no dataset downloaded and no training run',
        'kaggle_legacy_credentials_present': (Path.home()/'.kaggle/kaggle.json').is_file(),
        'kaggle_token_present': bool(os.getenv('KAGGLE_API_TOKEN')) or (Path.home()/'.kaggle/access_token').is_file(),
        'distribution_source': 'https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/pet_finder/pet_finder_dataset_builder.py',
        'rules_url': 'https://www.kaggle.com/competitions/petfinder-adoption-prediction/rules',
        'rules_acceptance': 'not verified in this session',
        'probes': probes,
    }
    target = Path('D:/meongtamjeong_research/petfinder_access')
    target.mkdir(exist_ok=True, parents=True)
    (target/'availability.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
