"""Audit the downloaded competition archive without extracting or logging descriptions."""
import argparse
import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(
        'D:/meongtamjeong_research/petfinder/petfinder-adoption-prediction.zip'))
    args = parser.parse_args()
    report = {'archive': str(args.archive), 'checked_utc': datetime.now(timezone.utc).isoformat()}
    with ZipFile(args.archive) as archive:
        members = archive.infolist()
        print('Checking all ZIP members (CRC)...', flush=True)
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Archive integrity failure: {bad}')
        report.update(crc_valid=True, archive_bytes=args.archive.stat().st_size,
                      uncompressed_bytes=sum(m.file_size for m in members), files=len(members))
        splits = {}
        for split in ('train', 'test'):
            photo_counts = Counter(
                Path(m.filename).stem.rsplit('-', 1)[0] for m in members
                if m.filename.startswith(f'{split}_images/') and m.filename.lower().endswith('.jpg'))
            with archive.open(f'{split}/{split}.csv') as handle:
                rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig')))
            dogs = [r for r in rows if r['Type'] == '1']
            paired = [r for r in dogs if photo_counts[r['PetID']] and r.get('Description', '').strip()]
            single = [r for r in paired if r.get('Quantity') == '1']
            splits[split] = {
                'all_listings': len(rows), 'unique_pet_ids': len({r['PetID'] for r in rows}),
                'dog_listings': len(dogs),
                'dog_photos': sum(photo_counts[r['PetID']] for r in dogs),
                'dogs_with_photo': sum(bool(photo_counts[r['PetID']]) for r in dogs),
                'dogs_with_description': sum(bool(r.get('Description', '').strip()) for r in dogs),
                'dogs_with_photo_and_description': len(paired),
                'single_dog_listings_with_photo_and_description': len(single),
                'single_dog_listings_with_description_and_multiple_photos': sum(photo_counts[r['PetID']] >= 2 for r in single),
                'single_dog_listings_with_description_and_photo_and_color_size': sum(
                    r.get('Color1', '0') not in ('', '0') and r.get('MaturitySize', '0') not in ('', '0') for r in single),
                'dog_rescuers': len({r['RescuerID'] for r in dogs if r.get('RescuerID')}),
                'columns': list(rows[0]) if rows else [],
            }
        report['splits'] = splits
    report['scope'] = ('Inventory only. No training, splitting, duplicate-image audit, '
                       'description accuracy validation, or image decoding performed.')
    target = args.archive.parent / 'inventory.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Report saved: {target}')


if __name__ == '__main__':
    main()
