"""Minimal FITS reader for POLARIS output — no astropy dependency.

FITS is simple enough to parse directly, and this machine has no astropy: the
header is a sequence of 80-character cards padded to 2880-byte blocks and ending
at an `END` card, then the data follows as big-endian binary, also padded to
2880. POLARIS gzips its detector files.

`read_fits(path)` -> (data, header). Multi-HDU files return the primary HDU;
`read_fits_all` returns every HDU.
"""

import gzip
import os

import numpy as np

_BITPIX = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}


def _open(path):
    return (gzip.open(path, "rb") if path.endswith((".gz", ".gzip"))
            else open(path, "rb"))


def _parse_header(fh):
    """Read 2880-byte blocks of 80-char cards until END. Returns dict or None."""
    hdr = {}
    while True:
        block = fh.read(2880)
        if not block or len(block) < 2880:
            return None if not hdr else hdr
        for i in range(0, 2880, 80):
            card = block[i:i + 80].decode("ascii", "replace")
            key = card[:8].strip()
            if key == "END":
                return hdr
            if not key or card[8:10] != "= ":
                continue                      # COMMENT / HISTORY / blank
            val = card[10:].split("/")[0].strip()
            if val.startswith("'"):
                val = val.strip().strip("'").strip()
            elif val in ("T", "F"):
                val = (val == "T")
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val.replace("D", "E"))
                    except ValueError:
                        pass
            hdr[key] = val


def _read_data(fh, hdr):
    naxis = int(hdr.get("NAXIS", 0))
    if naxis == 0:
        return None
    dims = [int(hdr[f"NAXIS{i}"]) for i in range(1, naxis + 1)]
    n = 1
    for d in dims:
        n *= d
    bitpix = int(hdr["BITPIX"])
    dt = np.dtype(_BITPIX[bitpix])
    nbytes = n * dt.itemsize
    raw = fh.read(nbytes)
    if len(raw) < nbytes:
        raise ValueError(f"truncated FITS data ({len(raw)} of {nbytes} bytes)")
    arr = np.frombuffer(raw, dtype=dt).astype(np.float64
                                              if bitpix < 0 else np.int64)
    # FITS axis order is fastest-first, numpy's is slowest-first
    arr = arr.reshape(dims[::-1])
    scale, zero = hdr.get("BSCALE", 1), hdr.get("BZERO", 0)
    if scale != 1 or zero != 0:
        arr = arr * scale + zero
    pad = (-nbytes) % 2880
    if pad:
        fh.read(pad)
    return arr


def read_fits_all(path):
    """Every HDU as a list of (data, header)."""
    out = []
    with _open(path) as fh:
        while True:
            hdr = _parse_header(fh)
            if not hdr:
                break
            out.append((_read_data(fh, hdr), hdr))
            if not hdr.get("EXTEND", False) and len(out) == 1:
                # still try for extensions: POLARIS writes a single HDU, but a
                # trailing block means more data
                pass
    return out


def read_fits(path):
    """The primary HDU as (data, header)."""
    hdus = read_fits_all(path)
    if not hdus:
        raise ValueError(f"no HDU found in {path}")
    return hdus[0]


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(f"=== {p}  ({os.path.getsize(p) / 1e6:.2f} MB)")
        for i, (data, hdr) in enumerate(read_fits_all(p)):
            shape = None if data is None else data.shape
            print(f"  HDU {i}: shape {shape} "
                  f"NAXIS={hdr.get('NAXIS')} BITPIX={hdr.get('BITPIX')}")
            for k, v in hdr.items():
                if k not in ("SIMPLE", "EXTEND"):
                    print(f"     {k:10s} = {v}")
            if data is not None:
                d = np.asarray(data)
                print(f"     data: min {d.min():.4e} max {d.max():.4e} "
                      f"nonzero {int(np.count_nonzero(d))}/{d.size}")
