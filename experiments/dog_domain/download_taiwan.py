"""Resume image downloads from the frozen official Taiwan adoption snapshot."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app.heldout_image_evaluation import ImageDownloadError, SafePublicImageDownloader
from experiments.composed_retrieval.download import sha256
from experiments.dog_domain.notice_extension import OUT as KOREAN_OUT, pixel_hash, read, write

OUT = KOREAN_OUT / 'taiwan'


def fetch(row, retry_failures=False):
    animal_id = str(row['animal_id'])
    if not animal_id.isdigit():
        raise ValueError('Expected a numeric official animal ID')
    image_path = OUT / 'images' / f'{animal_id}.png'
    record_path = OUT / 'downloads' / f'{animal_id}.json'
    url = row['album_file'].strip()
    if record_path.exists():
        cached = read(record_path)
        if cached['source_url'] == url:
            if cached['status'] == 'ok' and image_path.exists() and sha256(image_path) == cached['file_sha256']:
                return cached
            if cached['status'] != 'ok' and not retry_failures:
                return cached
    record = {'animal_id': animal_id, 'source_url': url,
              'downloaded_at': datetime.now(timezone.utc).isoformat()}
    downloader = SafePublicImageDownloader(timeout=15, retries=1,
                                           allowed_hosts=['www.pet.gov.tw', 'asms.coa.gov.tw'])
    try:
        downloaded = downloader.fetch(url)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = image_path.with_suffix('.part')
        downloaded.image.save(temporary, format='PNG')
        temporary.replace(image_path)
        record.update(status='ok', image_path=str(image_path.relative_to(OUT)),
                      final_url=downloaded.final_url, width=downloaded.width, height=downloaded.height,
                      source_bytes=downloaded.byte_count, stored_bytes=image_path.stat().st_size,
                      payload_sha256=downloaded.payload_sha256, pixel_sha256=pixel_hash(downloaded.image),
                      file_sha256=sha256(image_path))
    except ImageDownloadError as exc:
        record.update(status='failed', reason=str(exc))
    finally:
        downloader.close()
    write(record_path, record)
    return record


def run(retry_failures=False):
    source = read(OUT / 'source.json')
    assert source['sha256'] == sha256(OUT / 'notices.json')
    rows = [r for r in read(OUT / 'notices.json') if r['animal_kind'] == '\u72d7']
    selected = sorted([r for r in rows if str(r.get('album_file') or '').strip()], key=lambda r: str(r['animal_id']))
    assert len({str(r['animal_id']) for r in selected}) == len(selected)
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(lambda r: fetch(r, retry_failures), selected), 1):
            results.append(result)
            if i % 250 == 0:
                print('processed', i, '/', len(selected), dict(Counter(r['status'] for r in results)), flush=True)
    record_by_id = {str(r['animal_id']): r for r in selected}
    good = [r for r in results if r['status'] == 'ok']
    duplicates = {}
    records = []
    for result in good:
        row = record_by_id[result['animal_id']]
        duplicates.setdefault(result['pixel_sha256'], []).append(result['animal_id'])
        records.append({**result, 'shelter_id': str(row['animal_shelter_pkid']),
                        'color_original': row['animal_colour'], 'size_original': row['animal_bodytype'],
                        'age_original': row['animal_age'], 'breed_original': row['animal_Variety'],
                        'description_original': row['animal_remark'], 'caption_original': row['animal_caption'],
                        'source_update': row['animal_update']})
    duplicates = {h: ids for h, ids in duplicates.items() if len(ids) > 1}
    # Keep all source records; provide a deterministic de-duplicated view separately.
    seen, unique = set(), []
    for record in records:
        if record['pixel_sha256'] not in seen:
            unique.append(record)
            seen.add(record['pixel_sha256'])
    write(OUT / 'image_records.json', records)
    write(OUT / 'unique_image_records.json', unique)
    write(OUT / 'exact_duplicate_groups.json', duplicates)
    write(OUT / 'failed_downloads.json', [r for r in results if r['status'] != 'ok'])
    report = {'status': 'completed', 'source_sha256': source['sha256'], 'source_page': source['source_page'],
              'license': source['license'], 'dog_notices': len(rows), 'no_image_url': len(rows)-len(selected),
              'attempted': len(selected), 'downloaded': len(good), 'failed': len(selected)-len(good),
              'failure_reasons': dict(Counter(r['reason'] for r in results if r['status'] != 'ok')),
              'exact_duplicate_groups': len(duplicates), 'unique_images': len(unique),
              'duplicate_extra_records': len(good)-len(unique), 'stored_bytes': sum(r['stored_bytes'] for r in good),
              'unique_with_color_and_size': sum(bool(r['color_original'] and r['size_original']) for r in unique),
              'unique_with_description': sum(bool(str(r['description_original'] or '').strip()) for r in unique),
              'shelters': len({r['shelter_id'] for r in unique}),
              'image_records_sha256': sha256(OUT / 'image_records.json'),
              'unique_records_sha256': sha256(OUT / 'unique_image_records.json'),
              'completed_at': datetime.now(timezone.utc).isoformat(),
              'storage': 'Decoded RGB pixels saved losslessly as PNG; source bytes are hashed, not retained.',
              'limitations': ['Exact pixel duplicates only; near copies and multiple-animal photos not manually audited.',
                              'Single photo per notice; no same-image self-retrieval evaluation.',
                              'No training, evaluation, or raw-photo publication performed.']}
    write(OUT / 'download_report.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-failures', action='store_true')
    run(parser.parse_args().retry_failures)
