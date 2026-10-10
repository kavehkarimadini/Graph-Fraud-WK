"""Helpers for the Colab / Jupyter experiment notebook."""
from __future__ import print_function

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

GRAPHCONSIS_FILES = {
    'yelpchi': 'YelpChi.mat',
    'yelp': 'YelpChi.mat',
    'amazon': 'Amazon.mat',
    'tfinance': 'tfinance',
    'tsocial': 'tsocial',
    'elliptic': 'elliptic',
    'elliptic_bitcoin': 'elliptic',
    'elliptic_bitcoin_dataset': 'elliptic',
    'comp': 'comp',
    'comp.dgl': 'comp',
    'fdcompcn': 'comp',
}

DATASET_ALIASES = {
    'YelpChi': 'yelpchi',
    'yelpchi': 'yelpchi',
    'yelp': 'yelpchi',
    'Amazon': 'amazon',
    'amazon': 'amazon',
    'tfinance': 'tfinance',
    'T-Finance': 'tfinance',
    'tsocial': 'tsocial',
    'T-Social': 'tsocial',
    'elliptic': 'elliptic',
    'Elliptic': 'elliptic',
    'elliptic_bitcoin': 'elliptic',
    'elliptic_bitcoin_dataset': 'elliptic',
    'comp': 'comp',
    'comp.dgl': 'comp',
    'fdcompcn': 'comp',
    'FDCompCN': 'comp',
}

NEEDED_FILES = {
    'yelpchi': 'YelpChi.mat',
    'amazon': 'Amazon.mat',
    'tfinance': 'tfinance',
    'tsocial': 'tsocial',
}

# Official zip name -> extracted file that loaders expect under dataset/
DATASET_ZIPS = {
    'YelpChi.zip': 'YelpChi.mat',
    'Amazon.zip': 'Amazon.mat',
    'tfinance.zip': 'tfinance',
    'tsocial.zip': 'tsocial',
}


def in_colab():
    return 'google.colab' in sys.modules


def repo_root():
    here = Path(__file__).resolve().parent.parent
    marker = here / 'algorithms' / 'GraphConsis' / 'supervised_train.py'
    if marker.is_file():
        return here
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / 'algorithms' / 'GraphConsis' / 'supervised_train.py').is_file():
            return p
    return cwd


def dataset_dir(root=None):
    root = Path(root) if root else repo_root()
    path = root / 'dataset'
    path.mkdir(parents=True, exist_ok=True)
    return path


def notebook_runs_dir(root=None):
    root = Path(root) if root else repo_root()
    path = root / 'notebook_runs'
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_dataset(name):
    key = str(name).strip()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    return key.lower().replace('.mat', '')


def unzip_if_needed(directory):
    """Extract known dataset zips in ``directory`` into that same folder."""
    directory = Path(directory)
    for zpath in sorted(directory.glob('*.zip')):
        extract_dataset_zip(zpath, directory)


def missing_dataset_files(directory=None, root=None):
    ddir = Path(directory) if directory else dataset_dir(root)
    missing = []
    for fname in NEEDED_FILES.values():
        path = ddir / fname
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(fname)
    return missing


def missing_datasets(root=None):
    ddir = dataset_dir(root)
    unzip_if_needed(ddir)
    return missing_dataset_files(ddir)


def ensure_drive_mounted():
    """Mount Google Drive on Colab if it is not already mounted."""
    mydrive = Path('/content/drive/MyDrive')
    if mydrive.is_dir() or not in_colab():
        return mydrive if mydrive.is_dir() else None
    from google.colab import drive
    drive.mount('/content/drive')
    if not mydrive.is_dir():
        raise FileNotFoundError('Google Drive did not mount at /content/drive/MyDrive')
    return mydrive


def resolve_drive_path(user_path):
    """Resolve a Drive-relative or absolute folder/file path.

    Examples (after Drive is mounted):
      ``datasets``                  -> ``/content/drive/MyDrive/datasets``
      ``MyDrive/datasets``          -> ``/content/drive/MyDrive/datasets``
      ``/content/drive/MyDrive/datasets``
    """
    raw = str(user_path).strip().replace('\\', '/')
    if not raw:
        raise ValueError('Empty dataset path')
    p = Path(raw).expanduser()
    if p.exists():
        return p.resolve()

    ensure_drive_mounted()
    rel = raw.lstrip('/')
    candidates = []
    drive_root = Path('/content/drive')
    mydrive = drive_root / 'MyDrive'
    if rel.lower().startswith('content/drive/'):
        candidates.append(Path('/') / rel)
    if rel.lower().startswith('mydrive/'):
        candidates.append(drive_root / rel)
        candidates.append(mydrive / rel[len('MyDrive/'):])
    else:
        candidates.extend([mydrive / rel, drive_root / rel, Path.cwd() / rel])
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        'Dataset path not found: {!r}. Mount Drive and pass a folder relative '
        'to MyDrive (e.g. "GraphFraud/zips") or an absolute /content/drive/... path. '
        'Tried: {}'.format(user_path, [str(c) for c in candidates]))


def _find_named_file(root, filename):
    """Case-insensitive search for a file name under root (not recursive for speed first)."""
    root = Path(root)
    if root.is_file() and root.name.lower() == filename.lower():
        return root
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_file() and child.name.lower() == filename.lower():
            return child
    for child in root.rglob('*'):
        if child.is_file() and child.name.lower() == filename.lower():
            return child
    return None


def extract_dataset_zip(zpath, dest):
    """Extract known graph files from a zip into dest, flattening nested folders."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {v.lower(): v for v in NEEDED_FILES.values()}
    wanted.update({k.lower().replace('.zip', ''): v for k, v in DATASET_ZIPS.items()})
    extracted = []
    with zipfile.ZipFile(zpath, 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = Path(info.filename).name
            if base.lower() not in wanted and Path(zpath).stem.lower() not in wanted:
                continue
            if base.lower() in wanted:
                out_name = wanted[base.lower()]
            else:
                out_name = wanted[Path(zpath).stem.lower()]
            target = dest / out_name
            if target.is_file() and target.stat().st_size > 0:
                print('Already present:', target.name)
                extracted.append(target)
                continue
            print('Extracting {} ({:.1f} MB) from {} -> {}'.format(
                info.filename, info.file_size / 1e6, Path(zpath).name, target))
            tmp = target.with_suffix(target.suffix + '.partial')
            with zf.open(info, 'r') as src, open(tmp, 'wb') as out:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            tmp.replace(target)
            extracted.append(target)
    if not extracted:
        print('No known dataset files inside', zpath)
    return extracted


def _looks_like_repo(path):
    return (Path(path) / 'algorithms' / 'GraphConsis' / 'supervised_train.py').is_file()


def find_repo_root(start):
    start = Path(start)
    if _looks_like_repo(start):
        return start.resolve()
    if start.is_dir():
        for child in sorted(start.iterdir()):
            if child.is_dir() and _looks_like_repo(child):
                return child.resolve()
        marker = 'algorithms/GraphConsis/supervised_train.py'
        for hit in start.rglob('supervised_train.py'):
            if hit.parent.name == 'GraphConsis':
                root = hit.parents[2]
                if _looks_like_repo(root):
                    return root.resolve()
    return None


def unpack_archive(zpath, dest):
    """Unzip ``zpath`` into ``dest`` (uses system unzip on Colab when available)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zpath = Path(zpath)
    unzip_bin = shutil.which('unzip')
    if unzip_bin:
        print('Unzipping {} -> {} (this can take several minutes)'.format(zpath, dest))
        subprocess.check_call([unzip_bin, '-o', '-q', str(zpath), '-d', str(dest)])
        return dest
    print('Unzipping {} -> {} with zipfile'.format(zpath, dest))
    skip_parts = ('.venv/', '__pycache__/', '.git/')
    with zipfile.ZipFile(zpath, 'r') as zf:
        for info in zf.infolist():
            name = info.filename.replace('\\', '/')
            if any(part in name for part in skip_parts):
                continue
            zf.extract(info, dest)
    return dest


def unpack_repo_from_drive(drive_zip, dest=None):
    """Unzip a full-repo archive from Drive onto local disk and return the repo root.

    Zip the ``Graph-Fraud-WK`` folder (include ``dataset/``, exclude ``.venv``).
    ``drive_zip`` is relative to MyDrive, e.g. ``Graph-Fraud-WK.zip``.
    """
    src = resolve_drive_path(drive_zip)
    if not src.is_file():
        raise FileNotFoundError('Repo zip is not a file: {}'.format(src))
    dest = Path(dest) if dest else Path('/content') if in_colab() else (Path.cwd() / '_unpacked_repo')
    dest.mkdir(parents=True, exist_ok=True)
    print('Repo zip:', src)
    unpack_archive(src, dest)
    root = find_repo_root(dest)
    if root is None:
        raise FileNotFoundError(
            'Unzipped {}, but did not find algorithms/GraphConsis/. '
            'Zip the Graph-Fraud-WK folder itself (the one that contains algorithms/).'.format(src))
    ddir = root / 'dataset'
    if ddir.is_dir():
        unzip_if_needed(ddir)
    print('Repo root:', root)
    return root


def prepare_datasets_from_drive(drive_path, dest=None):
    """Unzip / copy YelpChi, Amazon, tfinance, tsocial from a Drive folder into dataset/.

    ``drive_path`` is typically a folder relative to MyDrive that contains the
    official zips (or already-extracted files). Extraction writes into the
    local repo ``dataset/`` so training does not read from Drive.
    """
    dest = Path(dest) if dest else dataset_dir()
    dest.mkdir(parents=True, exist_ok=True)
    src = resolve_drive_path(drive_path)
    print('Drive dataset source:', src)
    print('Extract / copy into:', dest.resolve())

    if src.is_file() and src.suffix.lower() == '.zip':
        extract_dataset_zip(src, dest)
    else:
        for zip_name, out_name in DATASET_ZIPS.items():
            target = dest / out_name
            if target.is_file() and target.stat().st_size > 0:
                print('Already present:', out_name)
                continue
            zpath = _find_named_file(src, zip_name)
            if zpath is not None:
                extract_dataset_zip(zpath, dest)
                continue
            extracted = _find_named_file(src, out_name)
            if extracted is not None:
                print('Copying', extracted, '->', target)
                shutil.copy2(extracted, target)
            else:
                print('Not found in Drive folder:', zip_name, 'or', out_name)

    missing = missing_dataset_files(dest)
    if missing:
        print('Still missing under dataset/:', missing)
    else:
        print('All required dataset files are in', dest.resolve())
    return missing


def make_run_tag(**kwargs):
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    parts = [stamp]
    if 'epochs' in kwargs:
        parts.append('ep{}'.format(kwargs['epochs']))
    if 'learning_rate' in kwargs:
        lr = str(kwargs['learning_rate']).replace('.', 'p')
        parts.append('lr{}'.format(lr))
    if 'hid_dim' in kwargs:
        parts.append('hid{}'.format(kwargs['hid_dim']))
    if 'order' in kwargs:
        parts.append('ord{}'.format(kwargs['order']))
    if 'dim_1' in kwargs:
        parts.append('dim{}'.format(kwargs['dim_1']))
    return '_'.join(parts)


def zip_run_folder(out_dir):
    """Zip a run folder next to it: ``.../seed123_tag.zip``."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError('Run folder not found: {}'.format(out_dir))
    archive = shutil.make_archive(str(out_dir), 'zip', root_dir=str(out_dir.parent),
                                  base_dir=out_dir.name)
    print('Zipped run to:', os.path.abspath(archive))
    return archive


def offer_download(zip_path):
    if not in_colab():
        print('Local run — zip is at', os.path.abspath(zip_path))
        return zip_path
    from google.colab import files
    files.download(zip_path)
    return zip_path


def zip_and_download(out_dir):
    if not out_dir:
        print('Nothing to zip (no_save?).')
        return None
    zip_path = zip_run_folder(out_dir)
    return offer_download(zip_path)


def _graphconsis_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    file_name = GRAPHCONSIS_FILES[ds]
    test_size = 0.2 if ds == 'yelpchi' else 0.4
    return dict(
        model='graphsage_mean',
        learning_rate=0.01,
        train_prefix=str(ddir) + os.sep,
        file_name=file_name,
        epochs=50 if ds != 'tsocial' else 20,
        dropout=0.0,
        weight_decay=0.0,
        dim_1=128,
        batch_size=512 if ds != 'tsocial' else 256,
        sigmoid=True,
        context_dim=0,
        gpu='0',
        seed=123,
        samples_1=25,
        samples_2=10,
        save_dir=str(notebook_runs_dir(root)),
        test_size=test_size,
        train_ratio=0.4,
        early_stop=25,
        no_save=False,
        run_tag=None,
        dataset_str='elliptic' if ds == 'elliptic' else os.path.splitext(file_name)[0],
    )


def _bwgnn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    hid_dim, order, epochs = (10, 5, 100) if ds == 'tsocial' else (64, 2, 100)
    test_size = 0.2 if ds == 'yelpchi' else 0.4
    return dict(
        dataset=ds,
        train_ratio=0.4,
        early_stop=25,
        hid_dim=hid_dim,
        order=order,
        epochs=epochs,
        learning_rate=0.01,
        weight_decay=0.0,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=test_size,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_graphconsis(dataset='yelpchi', download=True, **overrides):
    """Train GraphConsis, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.GraphConsis.supervised_train import train

    params = _graphconsis_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            dim_1=params['dim_1'])
    print('GraphConsis', params['file_name'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def run_bwgnn(dataset='tfinance', download=True, **overrides):
    """Train BWGNN, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.BWGNN.BWGNN_main import train

    params = _bwgnn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['hid_dim'], order=params['order'])
    print('BWGNN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _splitgnn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    from algorithms.SplitGNN.SplitGNN_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['comp'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        use_file_masks=False,
        intra_dim=paper['intra_dim'],
        C=paper['C'],
        K=paper['K'],
        gamma=paper['gamma'],
        dropout=paper['dropout'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=paper['learning_rate'],
        weight_decay=paper['weight_decay'],
        gpu='0',
        seed=30,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_splitgnn(dataset='comp', download=True, **overrides):
    """Train SplitGNN, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.SplitGNN.SplitGNN_main import train

    params = _splitgnn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'])
    print('SplitGNN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _dgagnn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.DGAGNN.DGAGNN_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        n_hidden=paper['n_hidden'],
        n_head=paper['n_head'],
        n_bins=paper['n_bins'],
        z=paper['z'],
        dropout=paper['dropout'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=0.01,
        weight_decay=5e-4,
        no_bin_encoding=False,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_dgagnn(dataset='yelpchi', download=True, **overrides):
    """Train DGA-GNN, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.DGAGNN.DGAGNN_main import train

    params = _dgagnn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'])
    print('DGA-GNN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _glhn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.GLHN.GLHN_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        hid_dim=paper['hid_dim'],
        n_layers=paper['n_layers'],
        tau=0.1,
        gamma1=paper['gamma1'],
        gamma2=paper['gamma2'],
        graph_type=paper['graph_type'],
        filter_pair='laplacian',
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=0.01,
        weight_decay=1e-5,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_glhn(dataset='yelpchi', download=True, **overrides):
    """Train GLHN, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.GLHN.GLHN_main import train

    params = _glhn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['hid_dim'])
    print('GLHN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _bsl_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.BSL.BSL_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        hid_dim=paper['hid_dim'],
        alpha=0.4,
        beta=0.8,
        batch_size=256,
        paper_split=False,
        label_ratio=paper['label_ratio'],
        unlabeled_ratio=paper['unlabeled_ratio'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=0.01,
        weight_decay=1e-5,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_bsl(dataset='yelpchi', download=True, **overrides):
    """Train BSL, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.BSL.BSL_main import train

    params = _bsl_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['hid_dim'])
    print('BSL', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _caregnn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.CAREGNN.CAREGNN_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        inter='GNN',
        emb_size=64,
        batch_size=paper['batch_size'],
        eval_batch_size=4096,
        lambda_1=2.0,
        weight_decay=1e-3,
        step_size=0.02,
        init_threshold=0.5,
        under_sample=1.0,
        no_rl=False,
        feat_norm=paper['feat_norm'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=0.01,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_caregnn(dataset='yelpchi', download=True, **overrides):
    """Train CARE-GNN, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.CAREGNN.CAREGNN_main import train

    params = _caregnn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['emb_size'])
    print('CARE-GNN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _h2fdetector_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.H2FDetector.H2FDetector_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        intra_dim=8,
        head=4,
        n_layer=2,
        dropout=0.1,
        gamma1=paper['gamma1'],
        gamma2=paper['gamma2'],
        feat_norm=paper['feat_norm'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=0.1,
        weight_decay=5e-5,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_h2fdetector(dataset='yelpchi', download=True, **overrides):
    """Train H2-FDetector, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.H2FDetector.H2FDetector_main import train

    params = _h2fdetector_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['intra_dim'] * params['head'])
    print('H2-FDetector', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _fraudre_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.FRAUDRE.FRAUDRE_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        embed_dim=64,
        n_layers=2,
        hidden_dim=64,
        max_neighbors=10,
        batch_size=paper['batch_size'],
        eval_batch_size=4096,
        weight_decay=1e-4,
        feat_norm=paper['feat_norm'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=paper['learning_rate'],
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_fraudre(dataset='yelpchi', download=True, **overrides):
    """Train FRAUDRE, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.FRAUDRE.FRAUDRE_main import train

    params = _fraudre_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['embed_dim'])
    print('FRAUDRE', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _pcgnn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.PCGNN.PCGNN_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        emb_size=64,
        rho=0.5,
        alpha=2.0,
        threshold=0.5,
        thres=paper['thres'],
        batch_size=paper['batch_size'],
        eval_batch_size=1024,
        weight_decay=paper['weight_decay'],
        feat_norm=paper['feat_norm'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=paper['learning_rate'],
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_pcgnn(dataset='yelpchi', download=True, **overrides):
    """Train PC-GNN, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.PCGNN.PCGNN_main import train

    params = _pcgnn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['emb_size'])
    print('PC-GNN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _sparsegad_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.SparseGAD.SparseGAD_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        hidden=paper['hidden'],
        nlayers=2,
        dropout=paper['dropout'],
        dropout_adj=paper['dropout_adj'],
        k=10,
        update_adj_epoch=paper['update_adj_epoch'],
        feat_norm=paper['feat_norm'],
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=0.01,
        weight_decay=5e-4,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_sparsegad(dataset='yelpchi', download=True, **overrides):
    """Train SparseGAD, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.SparseGAD.SparseGAD_main import train

    params = _sparsegad_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['hidden'])
    print('SparseGAD', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _amnet_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.AMNet.AMNet_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        relation=paper['relation'],
        hidden=paper['hidden'],
        order=paper['order'],
        n_filters=2,
        dropout=0.3,
        beta=paper['beta'],
        feat_norm='standard',
        loss_weight='none',
        select_metric='auroc',
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=paper['learning_rate'],
        lr_filter=5e-2,
        weight_decay=paper['weight_decay'],
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_amnet(dataset='yelpchi', download=True, **overrides):
    """Train AMNet, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.AMNet.AMNet_main import train

    params = _amnet_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['hidden'])
    print('AMNet', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _ghrn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.GHRN.GHRN_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        homo=1,
        hid_dim=paper['hid_dim'],
        order=paper['order'],
        del_ratio=paper['del_ratio'],
        adj_type='sym',
        self_loop=paper['self_loop'],
        feat_norm=paper['feat_norm'],
        epochs=100,
        early_stop=25,
        learning_rate=0.01,
        weight_decay=0.0,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_ghrn(dataset='yelpchi', download=True, **overrides):
    """Train GHRN (BWGNN, edge pruning, BWGNN), save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.GHRN.GHRN_main import train

    params = _ghrn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['hid_dim'])
    print('GHRN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _ggcn_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    from algorithms.GGCN.GGCN_main import DATASET_DEFAULTS
    paper = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    return dict(
        dataset=ds,
        train_ratio=0.4,
        relation='all',
        nlayers=paper['nlayers'],
        hidden=paper['hidden'],
        dropout=paper['dropout'],
        decay_rate=1.0,
        exponent=3.0,
        no_degree=False,
        no_sign=False,
        no_decay=False,
        use_bn=False,
        use_ln=False,
        scale_init=0.5,
        deg_intercept_init=0.5,
        feat_norm='standard',
        loss_weight='none',
        epochs=paper['epochs'],
        early_stop=paper['early_stop'],
        learning_rate=0.01,
        weight_decay=paper['weight_decay'],
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
        dataset_str=ds,
    )


def run_ggcn(dataset='yelpchi', download=True, **overrides):
    """Train GGCN, save artifacts, zip the run folder, download on Colab."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.GGCN.GGCN_main import train

    params = _ggcn_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['hidden'])
    print('GGCN', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def _tam_defaults(dataset, root):
    ddir = dataset_dir(root)
    ds = normalize_dataset(dataset)
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds == 'yelp':
        ds = 'yelpchi'
    return dict(
        dataset=ds,
        train_ratio=0.4,
        relation='all',
        embedding_dim=128,
        cutting=None,
        n_tree=None,
        lamda=0.0,
        minmax_loss=None,
        feat_norm=None,
        epochs=500,
        learning_rate=1e-5,
        weight_decay=0.0,
        gpu='0',
        seed=123,
        data_prefix=str(ddir) + os.sep,
        save_dir=str(notebook_runs_dir(root)),
        test_size=None,
        no_save=False,
        run_tag=None,
    )


def run_tam(dataset='yelpchi', download=True, **overrides):
    """Train TAM (unsupervised), save artifacts, zip the run folder, download on Colab.

    ``cutting`` / ``n_tree`` / ``minmax_loss`` / ``feat_norm`` left as ``None`` follow
    the per-dataset and graph-type defaults of ``TAM_main``.
    """
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.TAM.TAM_main import train

    params = _tam_defaults(dataset, root)
    params.update(overrides)
    if not params.get('run_tag'):
        params['run_tag'] = make_run_tag(
            epochs=params['epochs'], learning_rate=params['learning_rate'],
            hid_dim=params['embedding_dim'])
    print('TAM', params['dataset'], '->', params['save_dir'])
    metrics, out_dir = train(argparse.Namespace(**params))
    zip_path = zip_and_download(out_dir) if download else (zip_run_folder(out_dir) if out_dir else None)
    return {'metrics': metrics, 'out_dir': out_dir, 'zip_path': zip_path, 'params': params}


def zip_all_notebook_runs(root=None, download=True):
    """Zip the entire notebook_runs/ tree (every model/dataset/run)."""
    runs = notebook_runs_dir(root)
    archive = shutil.make_archive(str(runs), 'zip', root_dir=str(runs.parent),
                                  base_dir=runs.name)
    print('Zipped all notebook runs to:', os.path.abspath(archive))
    if download:
        offer_download(archive)
    return archive
