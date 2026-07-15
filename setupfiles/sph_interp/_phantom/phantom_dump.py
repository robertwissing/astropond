"""Write a Phantom full-dump binary IC from SPH particles.

Phantom uses a tagged Fortran-unformatted dump format (src/main/utils_dumpfiles.f90
+ readwrite_dumps.f90). This writer reproduces that on-disk layout for a single
hydro block (gas particles), which is what the suite needs as an IC. Layout
(each Fortran record = int32 byte-count, payload, int32 byte-count):

  1. global magic : int1(60769,i4), real(int2)(60878.0,f8), int2(60878,i4),
                    iversion(1,i4), int1o(690706,i4)
  2. fileid       : char*100 (must start with 'F' -> full dump, double precision)
  3. header group : the 8 datatypes in order
                    [i_int, i_int1, i_int2, i_int4, i_int8, i_real, i_real4,
                     i_real8]; each is a count(i4) record, then (if count>0) a
                    tags record (count x char*16) and a values record.
                      i_int  : nparttot, ntypes, npartoftype, nblocks, nptmass,
                               ndustlarge, ndustsmall
                      i_real : time, gamma, hfact, massoftype, udist, umass,
                               utime, umagfd
  4. nblockarrays : i4 (= narraylengths*nblocks = 1 here)
  5. block header : number(npart,i8), nums(8 x i4) = [0,0,0,0,0, 7, 1, 0]
  6. arrays (grouped by datatype, each = tag(char*16) record + data record):
       i_real  (f8): x, y, z, vx, vy, vz, u
       i_real4 (f4): h

MHD (Bxyz) is not written here (hydro IC); add an MHD block (extra i_real4 arrays
Bx/By/Bz + a separate block) once validated against a Phantom built with MHD=yes.

VALIDATION: `selftest_roundtrip` writes a dump and reads it back with Phantom's
own scripts/readPhantomDump.read_dump, asserting the particle arrays + header
round-trip. This proves the byte layout; FINAL acceptance (required-tag coverage)
still needs a check against a built Phantom (phantomsetup/moddump).
"""
import struct

import numpy as np

# Magic constants from utils_dumpfiles.f90.
INT1 = 60769
INT2 = 60878
INT1O = 690706
IVERSION = 1
LENTAG = 16
LENID = 100


def _record(payload):
    """Wrap a payload as a Fortran unformatted record (int32 len markers)."""
    n = len(payload)
    return struct.pack("<i", n) + payload + struct.pack("<i", n)


def _tag(s):
    """A char*16 tag, space-padded/truncated (Fortran fixed-length string)."""
    b = s.encode("ascii")[:LENTAG]
    return b + b" " * (LENTAG - len(b))


def _header_group(tags, vals, dtype):
    """One datatype's header sub-block: count(i4) record + tags + values."""
    return (_record(struct.pack("<i", len(tags)))
            + _record(b"".join(_tag(t) for t in tags))
            + _record(np.asarray(vals, dtype=dtype).tobytes()))


def _header_block(int_tags, int_vals, int8_tags, int8_vals, real_tags, real_vals):
    """The 8-datatype header group.

    Phantom writes the particle counts (nparttot, ntypes, npartoftype) in BOTH the
    default-int (i_int) AND int8 (i_int8) groups, and its tagged-read path
    (unfill_header) re-extracts nparttot/ntypes/npartoftype as kind=8 from i_int8.
    If they are absent there, the kind=8 extract leaves nparttot=0, Phantom
    allocates a zero-length particle array and the subsequent particle read
    overruns the heap (double-free/segfault). So i_int8 MUST be populated too.
    """
    out = b""
    # order: i_int, i_int1, i_int2, i_int4, i_int8, i_real, i_real4, i_real8
    for idx in range(8):
        if idx == 0 and int_tags:                       # i_int (int32)
            out += _header_group(int_tags, int_vals, "<i4")
        elif idx == 4 and int8_tags:                    # i_int8 (int64)
            out += _header_group(int8_tags, int8_vals, "<i8")
        elif idx == 5 and real_tags:                    # i_real (float64)
            out += _header_group(real_tags, real_vals, "<f8")
        else:
            out += _record(struct.pack("<i", 0))        # empty datatype
    return out


def write_phantom_fulldump(path, pos, vel, u, h, massoftype, time=0.0,
                           gamma=5.0 / 3.0, hfact=1.2,
                           udist=1.0, umass=1.0, utime=1.0, umagfd=1.0,
                           box=None, Bxyz=None, fileid=None):
    """Write a Phantom hydro full-dump IC.

    pos (N,3), vel (N,3), u (N,), h (N,): gas particle arrays.
    massoftype : the (equal) gas particle mass.
    box        : (3,) periodic period per axis, centred on the origin. REQUIRED for
                 a Phantom built with -DPERIODIC (our orstang/sedov SETUPs): the
                 reader's unfill_rheader FATALs ("dump does not contain boundary
                 positions but we are using periodic boundaries") unless the header
                 carries xmin..zmax. Written as [-box/2, +box/2]. None = omit (only
                 safe for a non-periodic Phantom build).
    The rest are header scalars (code units default to 1).
    """
    pos = np.ascontiguousarray(pos, dtype="<f8")
    vel = np.ascontiguousarray(vel, dtype="<f8")
    u = np.ascontiguousarray(u, dtype="<f8")
    h = np.ascontiguousarray(h, dtype="<f4")
    n = pos.shape[0]
    mhd = Bxyz is not None
    if mhd:
        Bxyz = np.ascontiguousarray(Bxyz, dtype="<f8").reshape(n, 3)
    if fileid is None:
        fileid = "FT:Phantom IC from tipsy (sph_interp)"

    int_tags = ["nparttot", "ntypes", "npartoftype", "nblocks", "nptmass",
                "ndustlarge", "ndustsmall"]
    int_vals = [n, 1, n, 1, 0, 0, 0]
    # Phantom re-reads these as kind=8 in its tagged-read path; must be present here
    # too or it allocates 0 particles and overruns on the particle read.
    int8_tags = ["nparttot", "ntypes", "npartoftype"]
    int8_vals = [n, 1, n]
    real_tags = ["time", "gamma", "hfact", "massoftype",
                 "udist", "umass", "utime", "umagfd"]
    real_vals = [time, gamma, hfact, massoftype, udist, umass, utime, umagfd]
    if box is not None:
        bx, by, bz = (float(v) for v in np.asarray(box, float).reshape(3))
        real_tags += ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
        real_vals += [-bx / 2, bx / 2, -by / 2, by / 2, -bz / 2, bz / 2]

    with open(path, "wb") as f:
        # 1. global magic record
        f.write(_record(struct.pack("<i", INT1) + struct.pack("<d", float(INT2))
                        + struct.pack("<i", INT2) + struct.pack("<i", IVERSION)
                        + struct.pack("<i", INT1O)))
        # 2. fileid (char*100)
        fid = fileid.encode("ascii")[:LENID]
        f.write(_record(fid + b" " * (LENID - len(fid))))
        # 3. header group
        f.write(_header_block(int_tags, int_vals, int8_tags, int8_vals,
                              real_tags, real_vals))
        # 4-6. blocks. Phantom uses narraylengths = 2 (hydro) or 4 (mhd): block 1 =
        # gas particles, block 2 = sinks (empty here), and for MHD block 3 (empty) +
        # block 4 = magnetic field. The real Fortran reader walks all narraylengths
        # blocks, so a single-block dump (narraylengths=1) desyncs/segfaults it; this
        # matches the canonical phantomsetup layout (verified by byte-diff). Each
        # datatype index: [int,i1,i2,i4,i8,real,real4,real8].
        #
        # Block headers for ALL blocks are written first (number + nums), THEN the
        # array data per block -- the order read_block_header/read_phantom_arrays use.
        # iorig (i8 particle IDs), alpha/divv/psi/divB/curlB are optional (the reader
        # has 'not found' fallbacks); we write iorig + h and, for MHD, Bx,By,Bz.
        narraylengths = 4 if mhd else 2
        nums1 = [0, 0, 0, 0, 1, 7, 3, 0]      # block 1: iorig(i8); x..u(real); h,alpha,divv(real4)
        nums_empty = [0, 0, 0, 0, 0, 0, 0, 0]  # empty sink/intermediate blocks
        # block 4 matches the canonical phantomsetup_mhd layout exactly: Bx,By,Bz,psi
        # (real) + divB,curlBx,curlBy,curlBz (real4). psi is the div-B cleaning scalar;
        # divB/curlB are diagnostics. All zero here (an IC has no cleaning history),
        # but writing the full set avoids the segfault seen when only Bx,By,Bz are
        # present and the MHD reader's Bevol/cleaning arrays are left partly unfilled.
        nums4 = [0, 0, 0, 0, 0, 4, 4, 0]      # block 4: Bx,By,Bz,psi(real); divB,curlBx,curlBy,curlBz(real4)
        block_hdrs = [(n, nums1), (0, nums_empty)]
        if mhd:
            block_hdrs += [(0, nums_empty), (n, nums4)]
        # 4. nblockarrays (= narraylengths * nblocks, nblocks=1)
        f.write(_record(struct.pack("<i", narraylengths)))
        # 5. all block headers
        for number, nums in block_hdrs:
            f.write(_record(struct.pack("<q", number)
                            + np.asarray(nums, dtype="<i4").tobytes()))
        # 6. array data, per block, in datatype order

        def _arr(tag, col, dt):
            f.write(_record(_tag(tag)))
            f.write(_record(np.ascontiguousarray(col, dtype=dt).tobytes()))

        # block 1: iorig(i8); x,y,z,vx,vy,vz,u(real f8); h,alpha,divv(real4 f4)
        _arr("iorig", np.arange(1, n + 1, dtype="<i8"), "<i8")
        for tag, col in (("x", pos[:, 0]), ("y", pos[:, 1]), ("z", pos[:, 2]),
                         ("vx", vel[:, 0]), ("vy", vel[:, 1]), ("vz", vel[:, 2]),
                         ("u", u)):
            _arr(tag, col, "<f8")
        _arr("h", h, "<f4")
        _arr("alpha", np.ones(n), "<f4")
        _arr("divv", np.zeros(n), "<f4")
        # block 2 (sinks) empty -> no arrays. MHD: block 3 empty, block 4 = full
        # magnetic set (Bx,By,Bz,psi real; divB,curlBx,curlBy,curlBz real4), all
        # zero except B -- matches the canonical layout so the reader's Bevol/cleaning
        # arrays are fully populated.
        if mhd:
            z8 = np.zeros(n)
            for tag, col in (("Bx", Bxyz[:, 0]), ("By", Bxyz[:, 1]),
                             ("Bz", Bxyz[:, 2]), ("psi", z8)):
                _arr(tag, col, "<f8")
            for tag in ("divB", "curlBx", "curlBy", "curlBz"):
                _arr(tag, z8, "<f4")
    return path


def selftest_roundtrip(path, pos, vel, u, h, massoftype, **hdr):
    """Write a dump then read it back with Phantom's own readPhantomDump and
    assert the particle arrays + key header values round-trip. Returns the dump
    dict. Raises AssertionError on mismatch."""
    import os
    import sys
    write_phantom_fulldump(path, pos, vel, u, h, massoftype, **hdr)
    # import the repo's reader (scripts/readPhantomDump.py)
    here = os.path.dirname(os.path.abspath(__file__))
    scripts = os.path.normpath(os.path.join(
        here, "..", "..", "..", "codes", "phantom", "phantom", "scripts"))
    sys.path.insert(0, scripts)
    import readPhantomDump as rpd
    d = rpd.read_dump(path)
    data = d["blocks"][0]["data"]
    assert np.allclose(data["x"], pos[:, 0]), "x mismatch"
    assert np.allclose(data["y"], pos[:, 1]), "y mismatch"
    assert np.allclose(data["z"], pos[:, 2]), "z mismatch"
    assert np.allclose(data["vx"], vel[:, 0]), "vx mismatch"
    assert np.allclose(data["u"], u), "u mismatch"
    assert np.allclose(np.asarray(data["h"], float), h, rtol=1e-6), "h mismatch"
    q = d["quantities"]
    assert abs(q["gamma"] - hdr.get("gamma", 5.0 / 3.0)) < 1e-6, "gamma mismatch"
    assert abs(q["massoftype"] - massoftype) < 1e-12, "massoftype mismatch"
    return d
