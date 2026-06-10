import os, shutil, random
from pathlib import Path

random.seed(42)
SRC = Path('data_raw')
DST_TRAIN = Path('data/train')
DST_TEST = Path('data/test')
TEST_FRAC = 0.30

SUPERCLASSES = ['Glioma', 'Meningioma', 'NORMAL', 'Neurocitoma',
                'Outros Tipos de Lesões', 'Schwannoma']

for cls in SUPERCLASSES:
    (DST_TRAIN / cls).mkdir(parents=True, exist_ok=True)
    (DST_TEST / cls).mkdir(parents=True, exist_ok=True)

counts = {cls: {'train': 0, 'test': 0} for cls in SUPERCLASSES}
unmatched = []

for folder in sorted(SRC.iterdir()):
    if not folder.is_dir():
        continue
    matched = next((cls for cls in SUPERCLASSES if folder.name.startswith(cls)), None)
    if matched is None:
        unmatched.append(folder.name)
        continue
    files = sorted(f for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    random.shuffle(files)
    n_test = int(len(files) * TEST_FRAC)
    contrast_tag = folder.name.split()[-1].replace('+', 'plus')
    for split_name, split_files in (('test', files[:n_test]), ('train', files[n_test:])):
        for f in split_files:
            dst = (DST_TEST if split_name == 'test' else DST_TRAIN) / matched / f'{contrast_tag}_{f.name}'
            shutil.copy2(f, dst)
            counts[matched][split_name] += 1

print('\nDONE')
print(f'{"Class":40s} {"train":>8s} {"test":>8s}')
for cls in SUPERCLASSES:
    print(f'{cls:40s} {counts[cls]["train"]:>8d} {counts[cls]["test"]:>8d}')
total_train = sum(c['train'] for c in counts.values())
total_test = sum(c['test'] for c in counts.values())
print(f'{"TOTAL":40s} {total_train:>8d} {total_test:>8d}')
if unmatched:
    print(f'\nUNMATCHED FOLDERS: {unmatched}')
