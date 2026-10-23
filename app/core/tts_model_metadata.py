"""Pinned download inventory and size cache; fetch functions run in a worker.

The cache describes the *complete* download, not remaining bytes or installed
files. No model weights are downloaded here.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile

from app.core.tts_catalog import MODEL_REVISIONS, MODELS

# Fixed resources fetched by _prepare_gpt_assets (download bytes, not extracted size).
GPT_AUXILIARY_BYTES = 588856634 + 131266198 + 896069 + 2526731 + 1539115
GPT_AUXILIARY_IDENTITY = [
    "G2PWModel.zip:0c47645e02a7bc3688d7b263b0042c81e3cd82cd:46292be0374a49308069233cd5c147ae4c41806558e4781a2467a31a4d8099da",
    "lid.176.bin:i0bvGpDpJiHbCgE2pdfmiT6e8rhcCJdD:131266198",
]



def _identity(model_id):
    from app.core.tts_runtime import _model_repos
    identity = [[repo, MODEL_REVISIONS[repo], folder, sorted(selected) if selected is not None else None]
                for repo, folder, selected in _model_repos(model_id)]
    if model_id == "gpt-sovits":
        from app.core.tts_runtime import _NLTK_REVISION, _NLTK_ASSETS
        identity += [GPT_AUXILIARY_IDENTITY, [_NLTK_REVISION, [list(row) for row in _NLTK_ASSETS]]]
    return identity


def _cache_path(model_id, cache_dir=None):
    from app.core.tts_runtime import ROOT
    directory = Path(cache_dir) if cache_dir is not None else ROOT / 'AppData/tts/_cache/model-metadata'
    identity = hashlib.sha256(json.dumps(_identity(model_id), sort_keys=True).encode()).hexdigest()
    return directory / (model_id + '-' + identity + '.json')


def build_model_inventory(model_id, fetch_json=None, cancel=None):
    """Return exactly the downloader's required fixed-revision file inventory."""
    from app.core.tts_runtime import _model_repos, _required_files, _json_url, _check, RuntimeSetupError
    if model_id == 'edge-tts':
        return []
    fetch_json = fetch_json or _json_url
    inventory = []
    for repo, folder, selected in _model_repos(model_id):
        _check(cancel)
        revision = MODEL_REVISIONS[repo]
        meta = fetch_json('https://huggingface.co/api/models/' + repo + '/revision/' + revision + '?blobs=true')
        if meta.get('sha') != revision:
            raise RuntimeSetupError('模型版本与固定清单不一致')
        siblings = meta['siblings']
        for row in siblings:
            name = row['rfilename']
            if selected is not None and name not in selected:
                continue
            if selected is None and (name.startswith('.') or name.endswith(('.png', '.jpg', '.mp4', '.md'))):
                continue
            if model_id == 'indextts2' and selected is None and name.startswith('qwen'):
                continue
            relative = 'semantic_codec_model.safetensors' if name == 'semantic_codec/model.safetensors' else name
            relative = (PurePosixPath(folder) / relative).as_posix()
            if PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts or '\\' in relative or ':' in relative:
                raise RuntimeSetupError('模型清单包含非法路径')
            lfs = row.get('lfs') or {}
            size = row.get('size', lfs.get('size'))
            sha256, blob_id = lfs.get('sha256'), row.get('blobId')
            if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not (sha256 or blob_id):
                raise RuntimeSetupError('模型清单缺少文件身份：' + name)
            inventory.append(dict(repo=repo, revision=revision, name=name, relative=relative,
                                  sha256=sha256, blob_id=blob_id, size=size))
        if selected is not None and not selected.issubset({row['rfilename'] for row in siblings}):
            raise RuntimeSetupError('必要模型文件不存在：' + repo)
    paths = [row['relative'] for row in inventory]
    if len(paths) != len(set(paths)) or not _required_files(model_id).issubset(paths):
        raise RuntimeSetupError('模型下载清单不完整')
    return inventory


def get_cached_model_bytes(model_id, cache_dir=None):
    """No network: revision-keyed disk cache, then verified catalog snapshot."""
    if model_id == 'edge-tts':
        return 0
    identity = _identity(model_id)
    try:
        value = json.loads(_cache_path(model_id, cache_dir).read_text('utf8'))
        total = value['bytes']
        if value.get('schema') == 1 and value.get('identity') == identity and type(total) is int and total > 0:
            return total
    except (OSError, ValueError, TypeError, KeyError):
        pass
    entry = MODELS[model_id]
    # Snapshots are valid only for the exact complete recipe, including aux repos.
    if entry.get('download_identity') == identity:
        return entry.get('download_bytes')
    return None


def fetch_model_bytes(model_id, cache_dir=None, fetch_json=None, cancel=None):
    """Network operation: call from a background worker, not the GUI thread."""
    if model_id == 'edge-tts':
        return 0
    inventory = build_model_inventory(model_id, fetch_json=fetch_json, cancel=cancel)
    total = sum(row['size'] for row in inventory) + (GPT_AUXILIARY_BYTES if model_id == 'gpt-sovits' else 0)
    path = _cache_path(model_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=1, identity=_identity(model_id), bytes=total)
    # Atomic replace permits independent dialogs/workers without partial reads.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(payload, stream)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return total


def format_model_size(total):
    return f'{total / 1e9:.2f} GB'
