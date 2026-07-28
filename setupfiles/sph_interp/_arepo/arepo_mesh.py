"""AREPO Voronoi mesh backend helper for sph_interp.

Runs AREPO (codes/arepo) as a SHELL-OUT mesher: write the generator points as an
HDF5 snapshot, run `Arepo param.txt 14 0` (RestartFlag 14 = build the Voronoi mesh
and dump it, no simulation), then parse the dumped face geometry into the package's
VoronoiGrid CSR. The mesher runs as a subprocess so a crash on pathological input
cannot take down the interpreter.

AREPO is a periodic moving-mesh code, so meshing is done in its native fully-closed
regime (every cell tiles the torus -> exact volume). Two modes (see `tessellate`):
  * legacy (periodic=None): PERIODIC over a CUBIC box (BoxSize = max domain extent);
    used by targets.py's ghost-IMAGE periodic path, which pads then meshes.
  * native (periodic=(3,)): PERIODIC over the PHYSICAL non-cubic box, per-axis
    periods baked in via compile-time LONG_Y/LONG_Z of a per-aspect binary
    (Arepo_asp<LY>_<LZ>). No ghost padding -- a thin/anisotropic box meshes
    correctly (the cubic clip put the short-axis period wrong -> malformed cells).
The dump format and the reconstruction quirks are documented inline; see also memory
`arepo-mesh-backend`.

Dump files (RestartFlag 14, little-endian, in OutputDir):
  tess_000_coordinates.dat : [i32 nVert][f32 x3*nVert]  vertex coords (single prec)
  tess_000_indices.dat     : [i32 nEntries] then per face [i32 cellID][i32 len]
                             [i32 x len vertexIdx]; cellID = generator ParticleID.
  tess_000_normals.dat     : (unused here; faces are rebuilt via ConvexHull)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

from .arepo_param import write_mesh_param

_HERE = os.path.dirname(__file__)
# The public checkout lives below codes/arepo/AREPO and build_arepo.sh leaves
# executables inside that checkout. Keeping `AREPO/` and `Arepo` beside one
# another is impossible on the default case-insensitive macOS filesystem.
_AREPO_ROOT = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "codes", "arepo"))
_AREPO_DIR = os.path.join(_AREPO_ROOT, "AREPO")
_AREPO_EXE = os.path.join(_AREPO_DIR, "Arepo")

# Non-cubic PERIODIC boxes need AREPO's per-axis periods baked in at COMPILE time
# via LONG_Y/LONG_Z (boxSize_Y = BoxSize*LONG_Y; LONG_* default 1 = cubic). We
# select/build a per-aspect binary Arepo_asp<LY>_<LZ>, matching runtest.sh's naming
# EXACTLY (tag = "%.4f_%.4f" % (LY, LZ) with '.'->'p') so the two share the cache.
_CUBIC_TOL = 5e-4                      # |LONG-1| below this -> use the base cubic binary

# --- parallel cell reconstruction -----------------------------------------
# scipy builds without open_memstream make EVERY ConvexHull() call create and
# unlink two temp files; with many workers those serialize on the shared /tmp
# directory lock (measured ceiling ~3.6 cores no matter the worker count), so
# each pool worker gets its own tempdir on /dev/shm. With that, the hull loop
# scales ~linearly in physical cores (200k cells: 45s serial -> 2.4s at 64).
_RECON_CHUNK = 2000                    # cells per pool task
_RECON_MIN_PARALLEL = 8000             # below this, pool startup beats the gain
_recon_data = None                     # (coord_s, cv0, cv1, gen0, dom_lo); set
                                       # before forking so workers inherit it


def _physical_cores():
    """Available cores / SMT siblings (pool workers gain nothing from SMT)."""
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 1
    try:
        with open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") as f:
            smt = max(1, len(f.read().strip().replace("-", ",").split(",")))
    except OSError:
        smt = 1
    return max(1, avail // smt)


def _recon_init(shmroot):
    import tempfile as _tf
    d = os.path.join(shmroot, "w%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    _tf.tempdir = d


def _recon_chunk(ab):
    """Rebuild cells [a, b): per-cell ConvexHull of its unwrapped vertices."""
    from scipy.spatial import ConvexHull
    coord_s, cv0, cv1, gen0, dom_lo = _recon_data
    a, b = ab
    vols = np.empty(b - a); rads = np.empty(b - a)
    nface = np.empty(b - a, np.int64)
    tris_list = []
    for row in range(a, b):
        cv = coord_s[cv0[row]:cv1[row]]                  # this cell's vertices
        d = cv - gen0[row]
        rads[row - a] = np.sqrt(float((d * d).sum(1).max()))
        # ConvexHull tolerates duplicate input points (it ignores coincident
        # ones), so skip a per-cell np.unique. Voronoi cell is convex -> hull
        # == cell: exact volume + clean outward-oriented triangles.
        hull = ConvexHull(cv)
        vols[row - a] = hull.volume
        tris = cv[hull.simplices] + dom_lo               # original frame
        nface[row - a] = tris.shape[0]
        tris_list.append(tris.reshape(-1, 3))
    return vols, rads, nface, np.concatenate(tris_list, axis=0)


def _reconstruct_cells(ncell, n_workers):
    """Run _recon_chunk over all cells, with a fork pool when it pays off.
    Chunk results are assembled in cell order, so the output is identical to
    the serial loop bit for bit."""
    import multiprocessing as mp
    import tempfile as _tf
    if ncell == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int64), np.zeros((0, 3))
    chunks = [(a, min(a + _RECON_CHUNK, ncell))
              for a in range(0, ncell, _RECON_CHUNK)]
    nw = _physical_cores() if (n_workers is None or n_workers <= 0) else int(n_workers)
    nw = min(nw, len(chunks))
    if nw > 1 and ncell >= _RECON_MIN_PARALLEL:
        shmbase = "/dev/shm" if os.path.isdir("/dev/shm") else None
        shmroot = _tf.mkdtemp(prefix="arepo_hull_", dir=shmbase)
        try:
            with mp.get_context("fork").Pool(nw, _recon_init, (shmroot,)) as pool:
                out = pool.map(_recon_chunk, chunks, chunksize=1)
        except OSError:                                  # fork/pool unavailable
            out = [_recon_chunk(ab) for ab in chunks]
        finally:
            shutil.rmtree(shmroot, ignore_errors=True)
    else:
        out = [_recon_chunk(ab) for ab in chunks]
    volume = np.concatenate([o[0] for o in out])
    cell_rad = np.concatenate([o[1] for o in out])
    nface = np.concatenate([o[2] for o in out])
    fvxyz = np.concatenate([o[3] for o in out], axis=0)
    return volume, cell_rad, nface, fvxyz


def has_arepo():
    return os.path.isfile(_AREPO_EXE) and os.access(_AREPO_EXE, os.X_OK)


def _missing_shared_libraries(exe):
    """Absolute shared-library dependencies that no longer exist.

    Homebrew keeps the stable ``opt`` directory but changes a library's ABI
    filename on upgrades (for example HDF5 ``310`` -> ``320``).  A previously
    built AREPO then still looks executable while dyld cannot launch it.
    """
    if not os.path.isfile(exe):
        return []
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["otool", "-L", exe], capture_output=True, text=True,
                timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return []
        missing = []
        for line in result.stdout.splitlines()[1:]:
            library = line.strip().split(" (compatibility version", 1)[0]
            # Current macOS keeps Apple libraries in the dyld shared cache, so
            # /usr/lib entries can launch even though stat() cannot see them.
            apple_library = library.startswith(("/usr/lib/", "/System/"))
            if (os.path.isabs(library) and not apple_library
                    and not os.path.exists(library)):
                missing.append(library)
        return missing
    if sys.platform.startswith("linux"):
        try:
            result = subprocess.run(
                ["ldd", exe], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [line.strip().split(" => not found", 1)[0]
                for line in result.stdout.splitlines()
                if " => not found" in line]
    return []


def _run_arepo_build(exe, config, build_dir, reason):
    """Build/relink one AREPO executable and verify its runtime dependencies."""
    helper = os.path.join(_AREPO_ROOT, "build_arepo.sh")
    if not os.path.isfile(helper):
        raise RuntimeError(
            f"AREPO executable is unusable ({reason}), and build helper "
            f"{helper} is missing")
    env = os.environ.copy()
    env.update(AREPO_CONFIG=os.path.basename(config),
               AREPO_BUILD_DIR=build_dir,
               AREPO_EXEC=os.path.basename(exe),
               AREPO_REBUILD="1",
               JOBS=str(os.cpu_count() or 4))
    result = subprocess.run(
        [helper], cwd=_AREPO_ROOT, env=env, capture_output=True, text=True)
    missing = _missing_shared_libraries(exe)
    if (result.returncode != 0 or not os.path.isfile(exe)
            or not os.access(exe, os.X_OK) or missing):
        detail = (result.stderr or result.stdout)[-2000:]
        if missing:
            detail += "\nstill missing: " + ", ".join(missing)
        raise RuntimeError(
            f"failed to rebuild AREPO ({reason}):\n{detail.strip()}")
    print(f"  rebuilt AREPO ({reason}) -> {exe}")
    return exe


def _aspect_tag(ly, lz):
    """Aspect tag matching runtest.sh: '%.4f_%.4f' % (LY, LZ) with '.'->'p'."""
    return ("%.4f_%.4f" % (ly, lz)).replace(".", "p")


def _ensure_arepo_binary(ly, lz):
    """Return the path to an AREPO binary whose compile-time LONG_Y/LONG_Z match the
    box aspect (ly=box_y/box_x, lz=box_z/box_x). Cubic (both ~1) -> the base `Arepo`.
    Otherwise `Arepo_asp<tag>`; built on demand (cp Config.sh + append LONG_Y/LONG_Z,
    then make) exactly as runtest.sh does, and cached by tag."""
    if abs(ly - 1.0) < _CUBIC_TOL and abs(lz - 1.0) < _CUBIC_TOL:
        if not has_arepo():
            raise RuntimeError("backend='arepo' needs the built Arepo executable "
                               "at %s" % _AREPO_EXE)
        missing = _missing_shared_libraries(_AREPO_EXE)
        if missing:
            config = os.path.join(_AREPO_DIR, "Config.sh")
            return _run_arepo_build(
                _AREPO_EXE, config, "build",
                "missing " + ", ".join(missing))
        return _AREPO_EXE
    tag = _aspect_tag(ly, lz)
    exe = os.path.join(_AREPO_DIR, "Arepo_asp%s" % tag)
    base_cfg = os.path.join(_AREPO_DIR, "Config.sh")
    if not os.path.isfile(base_cfg):
        raise RuntimeError("cannot build Arepo_asp%s: %s missing" % (tag, base_cfg))
    cfg = os.path.join(_AREPO_DIR, "Config_asp%s.sh" % tag)
    missing = (_missing_shared_libraries(exe)
               if os.path.isfile(exe) and os.access(exe, os.X_OK) else [])
    if os.path.isfile(exe) and os.access(exe, os.X_OK) and not missing:
        return exe
    if not os.path.isfile(cfg):
        with open(base_cfg) as f:
            cfg_txt = f.read()
        with open(cfg, "w") as f:
            f.write(cfg_txt)
            f.write("\nLONG_Y=%.10g\nLONG_Z=%.10g\n" % (ly, lz))
    reason = ("missing " + ", ".join(missing) if missing
              else "aspect-ratio binary not built")
    return _run_arepo_build(
        exe, cfg, "build_asp%s" % tag, reason)


def _write_snapshot(path, pts, boxsize):
    """Write generator points as an AREPO type-0 (gas) HDF5 snapshot (== IC format).
    The Voronoi tessellation depends only on Coordinates; hydro fields are trivial."""
    import h5py
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    n = pts.shape[0]
    f = h5py.File(path, "w")
    h = f.create_group("Header"); p0 = f.create_group("PartType0")
    npart = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
    h.attrs.create("NumPart_ThisFile", npart)
    h.attrs.create("NumPart_Total", npart)
    h.attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype=np.int32))
    h.attrs.create("MassTable", np.zeros(6, dtype=np.float64))
    h.attrs.create("Time", 0.0); h.attrs.create("Redshift", 0.0)
    h.attrs.create("BoxSize", np.float64(boxsize))
    h.attrs.create("NumFilesPerSnapshot", 1)
    h.attrs.create("Omega0", 0.0); h.attrs.create("OmegaB", 0.0)
    h.attrs.create("OmegaLambda", 0.0); h.attrs.create("HubbleParam", 1.0)
    for fl in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"):
        h.attrs.create(fl, 0)
    h.attrs.create("Flag_DoublePrecision", 1)
    p0.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))
    p0.create_dataset("Coordinates", data=pts)
    p0.create_dataset("Masses", data=np.ones(n, dtype=np.float64))
    p0.create_dataset("Velocities", data=np.zeros((n, 3), dtype=np.float64))
    p0.create_dataset("InternalEnergy", data=np.ones(n, dtype=np.float64))
    f.close()


def _read_coordinates(path):
    with open(path, "rb") as f:
        nvert = int(np.fromfile(f, np.int32, 1)[0])
        xyz = np.fromfile(f, np.float32, nvert * 3).reshape(nvert, 3).astype(np.float64)
    return xyz


def _read_indices(path):
    buf = np.fromfile(path, np.int32)
    nentries = int(buf[0]); pos = 1; end = 1 + nentries
    faces = []
    while pos < end:
        cid = int(buf[pos]); length = int(buf[pos + 1]); pos += 2
        faces.append((cid, buf[pos:pos + length].astype(np.int64))); pos += length
    return faces


def tessellate(pts, dom_lo, dom_hi, n_threads=1, keep=False, periodic=None,
               n_workers=-1):
    """Run AREPO on `pts` (within [dom_lo, dom_hi]) and return the mesh as a dict:
    gen (Ncell,3), volume, cell_rad, cf_start, fv_start, fvx/fvy/fvz (absolute,
    in the ORIGINAL frame). Coords are shifted by -dom_lo for the run and shifted
    back on parse.

    `periodic`: None/all-False -> the legacy CUBIC path (BoxSize = max extent),
    used by the ghost-image periodic path in targets.py where the padded set is
    meshed effectively non-periodically. A (3,) with any True -> the NATIVE
    per-axis PERIODIC path: AREPO meshes the PHYSICAL non-cubic box directly, with
    boxSize_Y/Z set by the compile-time LONG_Y/LONG_Z of a per-aspect binary
    (Arepo_asp<LY>_<LZ>, built on demand). No ghost-generator padding -- AREPO's
    own per-axis wrap produces the true periodic cells (which the per-face min-image
    unwrap below reconstructs around each generator).

    Reconstruction: AREPO's raw per-cell face lists are not always cyclically
    ordered and carry occasional extra near-coplanar entries, so fan-triangulating
    them overcounts. Voronoi cells are convex, so each cell is rebuilt as the
    CONVEX HULL of its (periodic-unwrapped) vertices -> exact volume + clean
    outward-oriented triangular faces. The per-cell hulls run on a fork pool of
    `n_workers` (<=0 -> all physical cores; serial below _RECON_MIN_PARALLEL
    cells), output identical to the serial loop -- see _reconstruct_cells."""
    dom_lo = np.asarray(dom_lo, float); dom_hi = np.asarray(dom_hi, float)
    per = (False, False, False) if periodic is None else tuple(bool(x) for x in periodic)
    native = any(per)
    if native:
        # PER-AXIS physical box; boxSize_X = BoxSize, boxSize_Y/Z via LONG_Y/LONG_Z.
        box = np.asarray(dom_hi - dom_lo, float)
        box_x = float(box[0])
        ly, lz = float(box[1] / box_x), float(box[2] / box_x)
        exe = _ensure_arepo_binary(ly, lz)
        boxsize = box_x
    else:
        # LEGACY cubic clip (ghost-image path meshes the padded set here).
        box = np.full(3, float(np.max(dom_hi - dom_lo)))
        exe = _ensure_arepo_binary(1.0, 1.0)
        boxsize = float(box[0])
    shifted = np.ascontiguousarray(pts, dtype=np.float64) - dom_lo
    # keep strictly inside (0, box_axis) -- AREPO rejects points on/over the face
    eps = 1e-9 * box
    shifted = np.clip(shifted, eps, box - eps)

    td = tempfile.mkdtemp(prefix="arepo_mesh_")
    try:
        outdir = os.path.join(td, "output"); os.makedirs(outdir)
        # Complete mode-14 param file (timestepping irrelevant -- AREPO meshes and
        # exits). MaxMemSize must cover AREPO's mesh-dump buffers, dominated by
        # VertexCoordinates = 1000*NumGas*3 floats (~0.012*N MB); scale it with N
        # (generous headroom) so large meshes don't OOM ("Not enough memory ...
        # for VertexCoordinates").
        n_pts = shifted.shape[0]
        mem_mb = max(8000, int(0.05 * n_pts))
        write_mesh_param(os.path.join(td, "param.txt"), boxsize=boxsize, maxmem=mem_mb)
        _write_snapshot(os.path.join(outdir, "snap_000.hdf5"), shifted, boxsize)
        cmd = [exe, "param.txt", "14", "0"]
        if n_threads and n_threads > 1:
            cmd = ["mpirun", "-np", str(int(n_threads))] + cmd
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True)
        base = os.path.join(outdir, "tess_000")
        if r.returncode != 0 or not os.path.isfile(base + "_coordinates.dat"):
            raise RuntimeError("arepo mesh dump failed (rc=%d): %s"
                               % (r.returncode, r.stderr[-1200:]))
        verts = _read_coordinates(base + "_coordinates.dat")
        faces = _read_indices(base + "_indices.dat")
    finally:
        if not keep:
            shutil.rmtree(td, ignore_errors=True)

    # --- VECTORIZED preprocessing (was ~12s of per-cell Python glue) ----------
    # Build flat CSR arrays in numpy. AREPO sorts faces by cellID PER RANK
    # (sort_faces_by_ID), so a serial dump is globally sorted -- but an MPI dump
    # CONCATENATES each rank's sorted block, so faces are only piecewise sorted
    # (N blocks for N ranks). We therefore group vertices by a per-vertex cell id
    # with a stable sort below, which is order-independent (serial == identity).
    fcell = np.fromiter((c for c, _ in faces), dtype=np.int64, count=len(faces))
    flens = np.fromiter((v.shape[0] for _, v in faces), dtype=np.int64, count=len(faces))
    fstart = np.zeros(len(faces) + 1, np.int64); fstart[1:] = np.cumsum(flens)
    allidx = (np.concatenate([v for _, v in faces]) if faces
              else np.zeros(0, np.int64))
    fcoord = verts[allidx]                                   # (totfv,3) gathered
    # per-face generator (cellID = ParticleID = shifted index + 1)
    face_g = shifted[fcell - 1]                              # (nface,3)
    # Unwrap EACH vertex to the minimum image around its cell generator. A face
    # crossing a periodic boundary can contain raw vertices on opposite stored
    # images, so shifting the whole face from its arithmetic centroid chooses the
    # wrong image on thin boxes. This vertex-wise form is unambiguous while every
    # cell spans less than half a period (enforced for sparse meshes by
    # VoronoiGrid's periodic background). `box` is per-axis for the native path.
    vertex_g = np.repeat(face_g, flens, axis=0)
    delta = fcoord - vertex_g
    fcoord_uw = fcoord - box * np.round(delta / box)

    # group vertices by CELL ID (a Voronoi cell is the convex hull of ALL its
    # vertices, so gathering them across however many rank-blocks reconstructs it)
    vcell = np.repeat(fcell, flens)                          # (totfv,) cell id per vertex
    order = np.argsort(vcell, kind="stable")                 # stable: serial == identity
    vcell_s = vcell[order]
    coord_s = fcoord_uw[order]
    uniq, cv0 = np.unique(vcell_s, return_index=True)        # first vertex of each cell
    ncell = uniq.shape[0]
    cv1 = np.append(cv0[1:], vcell_s.shape[0])               # one past last vertex
    gen = shifted[uniq - 1] + dom_lo                         # original frame

    global _recon_data
    _recon_data = (coord_s, cv0, cv1, gen - dom_lo, dom_lo)
    try:
        volume, cell_rad, nface, fvxyz = _reconstruct_cells(ncell, n_workers)
    finally:
        _recon_data = None
    nvert = np.full(int(nface.sum()), 3, np.int64)
    tface = nvert.shape[0]
    cf_start = np.zeros(ncell + 1, np.int64); cf_start[1:] = np.cumsum(nface)
    fv_start = np.zeros(tface + 1, np.int64); fv_start[1:] = np.cumsum(nvert)
    return dict(gen=gen, volume=volume, cell_rad=cell_rad, cf_start=cf_start,
                fv_start=fv_start, fvx=np.ascontiguousarray(fvxyz[:, 0]),
                fvy=np.ascontiguousarray(fvxyz[:, 1]),
                fvz=np.ascontiguousarray(fvxyz[:, 2]))
