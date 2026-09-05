"""Read homogeneous DGL v2 ``save_graphs`` binaries without importing DGL.

Official DGL wheels do not cover Python 3.13 or recent Windows+CUDA combos, and
``pip install dgl`` then falls back to the ancient 0.1.3 sdist whose
``libdgl.dll`` fails to load. T-Finance / T-Social are still DGL v2 files
(magic ``kDGLSerializeMagic``); this module walks NDArray blobs in that format.

NDArray on-disk layout (DGL ``SaveDGLArray``):
  uint64 magic (kDGLNDArrayMagic), uint64 reserved,
  DGLContext (int32 device_type, int32 device_id),
  int32 ndim, DGLDataType (uint8 code, uint8 bits, uint16 lanes),
  int64 shape[ndim], int64 data_byte_size, payload.
"""
from __future__ import print_function

import os
import struct

import numpy as np

K_DGL_SERIALIZE_MAGIC = 0xDD2E4FF046B4A13F
K_DGL_NDARRAY_MAGIC = 0xDD5E40F096B4A13F
_NDARRAY_MAGIC_BYTES = struct.pack('<Q', K_DGL_NDARRAY_MAGIC)

# DGLDataType.code: 0=int, 1=uint, 2=float
_DTYPE = {
    (0, 8): np.int8,
    (0, 16): np.int16,
    (0, 32): np.int32,
    (0, 64): np.int64,
    (1, 8): np.uint8,
    (1, 16): np.uint16,
    (1, 32): np.uint32,
    (1, 64): np.uint64,
    (2, 16): np.float16,
    (2, 32): np.float32,
    (2, 64): np.float64,
}


def _numpy_dtype(code, bits, lanes):
    if lanes != 1:
        return None
    return _DTYPE.get((int(code), int(bits)))


def iter_dgl_ndarrays(path):
    """Yield ``(shape, numpy_dtype, payload_offset, nbytes)`` for each NDArray."""
    import mmap

    path = os.path.abspath(path)
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        header = f.read(32)
        if len(header) < 32:
            raise ValueError('File too small to be a DGL graph: {}'.format(path))
        magic, _version, _graph_type, ngraph = struct.unpack_from('<QQQQ', header, 0)
        if magic != K_DGL_SERIALIZE_MAGIC:
            raise ValueError(
                'Not a DGL serialized graph (magic={:#x}): {}'.format(magic, path))
        if int(ngraph) < 1:
            raise ValueError('DGL file contains no graphs: {}'.format(path))
        mm_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            recs = list(_scan_ndarrays(mm_obj, file_size))
        finally:
            mm_obj.close()
    for rec in recs:
        yield rec


def _scan_ndarrays(mm, file_size):
    start = 0
    while True:
        i = mm.find(_NDARRAY_MAGIC_BYTES, start)
        if i < 0:
            break
        parsed = _parse_ndarray_header(mm, i, file_size)
        if parsed is None:
            start = i + 1
            continue
        shape, dt, data_off, nbytes = parsed
        yield shape, dt, data_off, nbytes
        start = data_off + max(nbytes, 0)


def _parse_ndarray_header(mm, off, file_size):
    if off + 32 > file_size:
        return None
    try:
        ndim = struct.unpack_from('<i', mm, off + 24)[0]
        code, bits, lanes = struct.unpack_from('<BBH', mm, off + 28)
    except struct.error:
        return None
    if ndim < 0 or ndim > 8:
        return None
    dt = _numpy_dtype(code, bits, lanes)
    if dt is None:
        return None
    shape_off = off + 32
    nbytes_off = shape_off + 8 * ndim
    if nbytes_off + 8 > file_size:
        return None
    if ndim:
        shape = struct.unpack_from('<' + 'q' * ndim, mm, shape_off)
    else:
        shape = ()
    if any(s < 0 for s in shape):
        return None
    nbytes = struct.unpack_from('<q', mm, nbytes_off)[0]
    expected = int(np.dtype(dt).itemsize)
    for s in shape:
        expected *= int(s)
    if nbytes != expected:
        return None
    data_off = nbytes_off + 8
    if data_off + nbytes > file_size:
        return None
    return shape, dt, data_off, nbytes


def _read_ndarray(path, shape, dt, data_off, nbytes):
    if nbytes == 0 or int(np.prod(shape)) == 0:
        return np.zeros(shape, dtype=dt)
    count = int(np.prod(shape))
    with open(path, 'rb') as f:
        f.seek(data_off)
        arr = np.fromfile(f, dtype=dt, count=count)
    if arr.size != count:
        raise IOError('Short read for NDArray at offset {} in {}'.format(data_off, path))
    return arr.reshape(shape)


def _is_integer_dtype(dt):
    return np.issubdtype(dt, np.integer)


def _is_float_dtype(dt):
    return np.issubdtype(dt, np.floating)


def _same_meta(a, b):
    return a is not None and b is not None and a[2] == b[2] and a[0] == b[0]


def _int_minmax(path, data_off, length, dt):
    arr = np.memmap(path, dtype=dt, mode='r', offset=data_off, shape=(int(length),))
    try:
        return int(arr.min()), int(arr.max())
    finally:
        del arr


def _is_feature_matrix(shape, dt):
    """Node features: 2-D float, or 2-D int that is not a 1/2-col one-hot."""
    if len(shape) != 2:
        return False
    cols = int(shape[1])
    if cols < 1 or cols > 512:
        return False
    if _is_float_dtype(dt):
        return True
    return _is_integer_dtype(dt) and cols > 2


def load_dgl_v2_homo_arrays(path):
    """Parse a homogeneous DGL v2 graph into numpy ``(features, labels, src, dst)``.

    Features are the largest 2-D float matrix, or a 2-D int matrix (T-Social).
    Labels are a 1-D/2-D integer array with the same number of rows (2-D is
    argmax'd). Edges are COO src/dst, or CSR indptr + indices.
    """
    path = os.path.abspath(path)
    metas = list(iter_dgl_ndarrays(path))
    if not metas:
        raise ValueError('No NDArray blobs found in {}'.format(path))

    feat_meta = None
    for shape, dt, data_off, nbytes in metas:
        if not _is_feature_matrix(shape, dt):
            continue
        # Prefer float (T-Finance); otherwise take the largest 2-D block (T-Social).
        better = feat_meta is None or int(np.prod(shape)) > int(np.prod(feat_meta[0]))
        if feat_meta is not None and _is_float_dtype(feat_meta[1]) and not _is_float_dtype(dt):
            better = False
        if _is_float_dtype(dt) and feat_meta is not None and not _is_float_dtype(feat_meta[1]):
            better = True
        if better:
            feat_meta = (shape, dt, data_off, nbytes)
    if feat_meta is None:
        raise ValueError('No 2-D feature matrix in {}'.format(path))
    features = _read_ndarray(path, *feat_meta)
    n = int(features.shape[0])

    lab2, lab1_cands = None, []
    for shape, dt, data_off, nbytes in metas:
        if _same_meta((shape, dt, data_off, nbytes), feat_meta):
            continue
        if not _is_integer_dtype(dt) or not shape or int(shape[0]) != n:
            continue
        if len(shape) == 2 and 1 <= int(shape[1]) <= 2:
            if lab2 is None or int(shape[1]) < int(lab2[0][1]):
                lab2 = (shape, dt, data_off, nbytes)
        elif len(shape) == 1:
            lab1_cands.append((shape, dt, data_off, nbytes))
    if lab2 is not None:
        raw = _read_ndarray(path, *lab2)
        labels = np.argmax(raw, axis=1).astype(np.int64) if raw.shape[1] > 1 else raw.reshape(-1).astype(np.int64)
    elif lab1_cands:
        # Prefer a class-id vector (few distinct values) over node ids / arange.
        best, best_nunique = None, None
        for meta in lab1_cands:
            arr = _read_ndarray(path, *meta).reshape(-1)
            nunique = int(np.unique(arr).size)
            if nunique < 2:
                continue
            if best is None or nunique < best_nunique:
                best, best_nunique = arr, nunique
        if best is None:
            best = _read_ndarray(path, *lab1_cands[0]).reshape(-1)
        labels = best.astype(np.int64)
    else:
        raise ValueError('No label array with {} rows in {}'.format(n, path))

    node_index_arrs = []
    indptr_meta = None
    for shape, dt, data_off, nbytes in metas:
        if len(shape) != 1 or not _is_integer_dtype(dt):
            continue
        length = int(shape[0])
        if length == n + 1:
            indptr_meta = (shape, dt, data_off, nbytes)
            continue
        if length <= n:
            continue
        mn, mx = _int_minmax(path, data_off, length, dt)
        if mn >= 0 and mx < n:
            node_index_arrs.append((length, shape, dt, data_off, nbytes))

    src = dst = None
    by_len = {}
    for rec in node_index_arrs:
        by_len.setdefault(rec[0], []).append(rec[1:])
    for length in sorted(by_len, reverse=True):
        if len(by_len[length]) >= 2:
            a, b = by_len[length][0], by_len[length][1]
            src = _read_ndarray(path, *a).astype(np.int64, copy=False)
            dst = _read_ndarray(path, *b).astype(np.int64, copy=False)
            break
    if src is None and indptr_meta is not None and node_index_arrs:
        indices_meta = max(node_index_arrs, key=lambda r: r[0])[1:]
        indptr = _read_ndarray(path, *indptr_meta).astype(np.int64, copy=False)
        dst = _read_ndarray(path, *indices_meta).astype(np.int64, copy=False)
        if int(indptr[-1]) == dst.size and indptr.size == n + 1:
            src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    if src is None or dst is None:
        raise ValueError('Could not find COO src/dst arrays in {}'.format(path))
    if src.size != dst.size or int(src.min()) < 0 or int(dst.min()) < 0 or int(src.max()) >= n or int(dst.max()) >= n:
        raise ValueError('COO indices out of range for {} nodes in {}'.format(n, path))

    features = np.ascontiguousarray(features, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    return features, labels, src, dst


# FDCompCN / SplitGNN: C-I-C, C-S-C, C-P-C after the homogeneous union.
_HETERO_REL_NAMES = ('cic', 'csc', 'cpc')


def load_dgl_v2_hetero_arrays(path):
    """Parse a multi-relation DGL v2 heterograph (e.g. ``dataset/comp.dgl``).

    Returns a dict with:

    - ``features`` ``(N, F)`` float32
    - ``labels`` ``(N,)`` int64 (class ids; uint8 train/valid/test masks are not used)
    - ``relations`` list of ``(name, src, dst)`` — largest COO pair is ``homo``
    - optional ``train_mask`` / ``valid_mask`` / ``test_mask`` uint8 ``(N,)``
    - optional ``homo_edge_label`` int64 in ``{-1, 1}`` and ``homo_edge_train_mask``
    """
    path = os.path.abspath(path)
    metas = list(iter_dgl_ndarrays(path))
    if not metas:
        raise ValueError('No NDArray blobs found in {}'.format(path))

    feat_meta = None
    for shape, dt, data_off, nbytes in metas:
        if not _is_feature_matrix(shape, dt):
            continue
        better = feat_meta is None or int(np.prod(shape)) > int(np.prod(feat_meta[0]))
        if feat_meta is not None and _is_float_dtype(feat_meta[1]) and not _is_float_dtype(dt):
            better = False
        if _is_float_dtype(dt) and feat_meta is not None and not _is_float_dtype(feat_meta[1]):
            better = True
        if better:
            feat_meta = (shape, dt, data_off, nbytes)
    if feat_meta is None:
        raise ValueError('No 2-D feature matrix in {}'.format(path))
    features = np.ascontiguousarray(_read_ndarray(path, *feat_meta), dtype=np.float32)
    n = int(features.shape[0])

    labels = None
    labels_nunique = None
    mask_metas = []
    for shape, dt, data_off, nbytes in metas:
        if _same_meta((shape, dt, data_off, nbytes), feat_meta):
            continue
        if len(shape) != 1 or int(shape[0]) != n or not _is_integer_dtype(dt):
            continue
        if np.dtype(dt) == np.uint8:
            mask_metas.append((shape, dt, data_off, nbytes))
            continue
        arr = _read_ndarray(path, shape, dt, data_off, nbytes).reshape(-1)
        nunique = int(np.unique(arr).size)
        if nunique < 2:
            continue
        if labels is None or nunique < labels_nunique:
            labels, labels_nunique = arr.astype(np.int64, copy=False), nunique
    if labels is None:
        raise ValueError('No 1-D integer label array with {} rows in {}'.format(n, path))
    labels = np.ascontiguousarray(labels, dtype=np.int64)

    train_mask = valid_mask = test_mask = None
    if len(mask_metas) >= 3:
        train_mask = np.ascontiguousarray(_read_ndarray(path, *mask_metas[0]), dtype=np.uint8)
        valid_mask = np.ascontiguousarray(_read_ndarray(path, *mask_metas[1]), dtype=np.uint8)
        test_mask = np.ascontiguousarray(_read_ndarray(path, *mask_metas[2]), dtype=np.uint8)
    elif len(mask_metas) == 1:
        train_mask = np.ascontiguousarray(_read_ndarray(path, *mask_metas[0]), dtype=np.uint8)

    pairs = []
    i = 0
    while i < len(metas) - 1:
        s1, dt1, off1, nb1 = metas[i]
        s2, dt2, off2, nb2 = metas[i + 1]
        if (len(s1) == 1 and len(s2) == 1 and int(s1[0]) == int(s2[0])
                and int(s1[0]) >= 32 and int(s1[0]) != n
                and _is_integer_dtype(dt1) and _is_integer_dtype(dt2)
                and np.dtype(dt1) == np.dtype(dt2) and np.dtype(dt1) != np.uint8):
            mn1, mx1 = _int_minmax(path, off1, s1[0], dt1)
            mn2, mx2 = _int_minmax(path, off2, s2[0], dt2)
            if mn1 >= 0 and mn2 >= 0 and mx1 < n and mx2 < n:
                src = np.ascontiguousarray(_read_ndarray(path, s1, dt1, off1, nb1), dtype=np.int64)
                dst = np.ascontiguousarray(_read_ndarray(path, s2, dt2, off2, nb2), dtype=np.int64)
                pairs.append((src, dst))
                i += 2
                continue
        i += 1
    if not pairs:
        raise ValueError('No COO relation pairs found in {}'.format(path))

    sizes = [p[0].size for p in pairs]
    homo_i = int(np.argmax(sizes))
    relations = [('homo', pairs[homo_i][0], pairs[homo_i][1])]
    extra_i = 0
    for idx, (src, dst) in enumerate(pairs):
        if idx == homo_i:
            continue
        name = _HETERO_REL_NAMES[extra_i] if extra_i < len(_HETERO_REL_NAMES) else 'rel{}'.format(extra_i)
        relations.append((name, src, dst))
        extra_i += 1

    homo_e = int(relations[0][1].size)
    homo_edge_label = None
    homo_edge_train_mask = None
    for shape, dt, data_off, nbytes in metas:
        if len(shape) != 1 or int(shape[0]) != homo_e:
            continue
        if _is_integer_dtype(dt) and np.dtype(dt) != np.uint8:
            arr = _read_ndarray(path, shape, dt, data_off, nbytes).reshape(-1)
            uniq = set(int(v) for v in np.unique(arr))
            if uniq <= {-1, 0, 1} and -1 in uniq:
                homo_edge_label = np.ascontiguousarray(arr, dtype=np.int64)
        elif np.dtype(dt) == np.uint8:
            homo_edge_train_mask = np.ascontiguousarray(
                _read_ndarray(path, shape, dt, data_off, nbytes), dtype=np.uint8)

    return {
        'features': features,
        'labels': labels,
        'relations': relations,
        'train_mask': train_mask,
        'valid_mask': valid_mask,
        'test_mask': test_mask,
        'homo_edge_label': homo_edge_label,
        'homo_edge_train_mask': homo_edge_train_mask,
    }
