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


def load_dgl_v2_homo_arrays(path):
    """Parse a homogeneous DGL v2 graph into numpy ``(features, labels, src, dst)``.

    Features are the largest 2-D float array. Labels are a 1-D/2-D integer array
    with the same number of rows (2-D is argmax'd). Edges are the two longest
    1-D integer arrays of equal length whose values index ``[0, n)``.
    """
    path = os.path.abspath(path)
    metas = list(iter_dgl_ndarrays(path))
    if not metas:
        raise ValueError('No NDArray blobs found in {}'.format(path))

    feat_meta = None
    for shape, dt, data_off, nbytes in metas:
        if len(shape) == 2 and _is_float_dtype(dt):
            if feat_meta is None or int(np.prod(shape)) > int(np.prod(feat_meta[0])):
                feat_meta = (shape, dt, data_off, nbytes)
    if feat_meta is None:
        raise ValueError('No 2-D float feature matrix in {}'.format(path))
    features = _read_ndarray(path, *feat_meta)
    n = int(features.shape[0])

    lab2, lab1 = None, None
    for shape, dt, data_off, nbytes in metas:
        if not _is_integer_dtype(dt) or not shape or int(shape[0]) != n:
            continue
        if len(shape) == 2 and 1 <= int(shape[1]) <= 32:
            if lab2 is None or int(shape[1]) < int(lab2[0][1]):
                lab2 = (shape, dt, data_off, nbytes)
        elif len(shape) == 1:
            lab1 = (shape, dt, data_off, nbytes)
    if lab2 is not None:
        raw = _read_ndarray(path, *lab2)
        labels = np.argmax(raw, axis=1).astype(np.int64) if raw.shape[1] > 1 else raw.reshape(-1).astype(np.int64)
    elif lab1 is not None:
        labels = _read_ndarray(path, *lab1).reshape(-1).astype(np.int64)
    else:
        raise ValueError('No label array with {} rows in {}'.format(n, path))

    by_len = {}
    for shape, dt, data_off, nbytes in metas:
        if len(shape) != 1 or not _is_integer_dtype(dt):
            continue
        length = int(shape[0])
        if length <= n:
            continue
        by_len.setdefault(length, []).append((shape, dt, data_off, nbytes))
    pair = None
    for length in sorted(by_len, reverse=True):
        if len(by_len[length]) >= 2:
            pair = by_len[length][:2]
            break
    if pair is None:
        raise ValueError('Could not find COO src/dst arrays in {}'.format(path))
    src = _read_ndarray(path, *pair[0]).astype(np.int64, copy=False)
    dst = _read_ndarray(path, *pair[1]).astype(np.int64, copy=False)
    if int(src.min()) < 0 or int(dst.min()) < 0 or int(src.max()) >= n or int(dst.max()) >= n:
        raise ValueError('COO indices out of range for {} nodes in {}'.format(n, path))

    features = np.ascontiguousarray(features, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    return features, labels, src, dst
