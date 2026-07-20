"""Target density fields: table lookup or arbitrary Python callable.

Coordinate mappings follow the C direction codes in pkdUpdateDensity
(pkd.c): 1=x, 2=y, 3=z, 4=spherical r, 5=cylindrical r, 6=signed x.
A 1D table/callable is evaluated at the mapped coordinate; a 2D table at a
pair of mapped coordinates; a 3D table at (x, y, z) directly.

THE table format is .npz and SELF-DESCRIBING (CLEANUP_PLAN C6, 2026-07-17):
a TableDensity saves as axes + values + its coordinate mapping in the
header, so a loaded table knows its own dimensionality, extent and
coordinate system - no --direction flag at the point of use. AMR tables
(amrtable.AMRTable) share the .npz container with their own block layout;
reconstruct.load_table() sniffs and loads either.

The densitytable_xdr format of the C icgenerator (3-line ascii header
dim/n/R, then XDR float arrays, each prefixed by a u32 length) is LEGACY:
`from_xdr` remains only as the reader for the one-shot converter

    python -m glassgen.density old_densitytable_xdr new_table.npz \
        --direction r_sph

(numpy big-endian dtypes rather than xdrlib, removed in Python 3.13).
"""
import numpy as np
from numba import njit, prange

MAPPINGS = ('x', 'y', 'z', 'r_sph', 'r_cyl', 'signed_x')


def map_coord(pos, mapping):
    if mapping == 'x':
        return np.abs(pos[:, 0])
    if mapping == 'y':
        return np.abs(pos[:, 1])
    if mapping == 'z':
        return np.abs(pos[:, 2])
    if mapping == 'r_sph':
        return np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    if mapping == 'r_cyl':
        return np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    if mapping == 'signed_x':
        return pos[:, 0].copy()
    raise ValueError(f"unknown mapping '{mapping}', expected one of {MAPPINGS}")


@njit(cache=True, parallel=True)
def _interp1d(axis, values, x, out):
    n = axis.shape[0]
    x0 = axis[0]
    dx = axis[1] - axis[0]
    for i in prange(x.shape[0]):
        t = (x[i] - x0) / dx
        if t <= 0.0:
            out[i] = values[0]
        elif t >= n - 1:
            out[i] = values[n - 1]
        else:
            j = int(t)
            f = t - j
            out[i] = values[j] * (1.0 - f) + values[j + 1] * f


@njit(cache=True, parallel=True)
def _interp2d(ax0, ax1, values, x, y, out):
    n0 = ax0.shape[0]
    n1 = ax1.shape[0]
    for i in prange(x.shape[0]):
        t0 = (x[i] - ax0[0]) / (ax0[1] - ax0[0])
        t1 = (y[i] - ax1[0]) / (ax1[1] - ax1[0])
        if t0 < 0.0:
            t0 = 0.0
        elif t0 > n0 - 1:
            t0 = n0 - 1.0
        if t1 < 0.0:
            t1 = 0.0
        elif t1 > n1 - 1:
            t1 = n1 - 1.0
        j0 = min(int(t0), n0 - 2)
        j1 = min(int(t1), n1 - 2)
        f0 = t0 - j0
        f1 = t1 - j1
        out[i] = (values[j0, j1] * (1 - f0) * (1 - f1)
                  + values[j0, j1 + 1] * (1 - f0) * f1
                  + values[j0 + 1, j1] * f0 * (1 - f1)
                  + values[j0 + 1, j1 + 1] * f0 * f1)


@njit(cache=True, parallel=True)
def _interp3d(ax0, ax1, ax2, values, pos, out):
    n0, n1, n2 = ax0.shape[0], ax1.shape[0], ax2.shape[0]
    for i in prange(pos.shape[0]):
        t0 = (pos[i, 0] - ax0[0]) / (ax0[1] - ax0[0])
        t1 = (pos[i, 1] - ax1[0]) / (ax1[1] - ax1[0])
        t2 = (pos[i, 2] - ax2[0]) / (ax2[1] - ax2[0])
        if t0 < 0.0:
            t0 = 0.0
        elif t0 > n0 - 1:
            t0 = n0 - 1.0
        if t1 < 0.0:
            t1 = 0.0
        elif t1 > n1 - 1:
            t1 = n1 - 1.0
        if t2 < 0.0:
            t2 = 0.0
        elif t2 > n2 - 1:
            t2 = n2 - 1.0
        j0 = min(int(t0), n0 - 2)
        j1 = min(int(t1), n1 - 2)
        j2 = min(int(t2), n2 - 2)
        f0, f1, f2 = t0 - j0, t1 - j1, t2 - j2
        c00 = values[j0, j1, j2] * (1 - f2) + values[j0, j1, j2 + 1] * f2
        c01 = values[j0, j1 + 1, j2] * (1 - f2) + values[j0, j1 + 1, j2 + 1] * f2
        c10 = values[j0 + 1, j1, j2] * (1 - f2) + values[j0 + 1, j1, j2 + 1] * f2
        c11 = values[j0 + 1, j1 + 1, j2] * (1 - f2) + values[j0 + 1, j1 + 1, j2 + 1] * f2
        c0 = c00 * (1 - f1) + c01 * f1
        c1 = c10 * (1 - f1) + c11 * f1
        out[i] = c0 * (1 - f0) + c1 * f0


class DensityField:
    """Interface: rho0(pos) maps (N,3) positions to (N,) target densities."""

    def rho0(self, pos):
        raise NotImplementedError

    def rho0_h(self, pos, h):
        """Kernel-scale-aware target readout: the target as seen by a particle
        with smoothing length ``h`` (support radius 2h, glassgen convention).
        Default = the point read (scale-free fields); overridden by
        :class:`KernelMatchedDensity` with the kernel average. The relax
        evaluates the pseudo-pressure target through this."""
        return self.rho0(pos)


class TableDensity(DensityField):
    def __init__(self, axes, values, mapping='r_sph'):
        values = np.ascontiguousarray(values, dtype=np.float64)
        self.axes = tuple(np.ascontiguousarray(a, dtype=np.float64)
                          for a in (axes if isinstance(axes, (tuple, list)) else (axes,)))
        self.values = values
        self.dim = len(self.axes)
        if values.ndim != self.dim:
            raise ValueError(f"{self.dim} axes but values has ndim {values.ndim}")
        if self.dim == 1:
            self.mapping = mapping
        elif self.dim == 2:
            if not (isinstance(mapping, (tuple, list)) and len(mapping) == 2):
                raise ValueError("2D table needs a pair of mappings, e.g. ('r_cyl','z')")
            self.mapping = tuple(mapping)
        else:
            self.mapping = None  # 3D tables are indexed by (x, y, z)

    def save(self, path):
        """Write the self-describing .npz: ax0..ax{dim-1} + values + the
        coordinate mapping (1D/2D; a 3D table is indexed by (x,y,z) and
        carries none). Loads back via TableDensity.load / load_table."""
        meta = {'values': self.values}
        for i, a in enumerate(self.axes):
            meta[f'ax{i}'] = a
        if self.dim == 1:
            meta['mapping'] = np.array([self.mapping])
        elif self.dim == 2:
            meta['mapping'] = np.array(list(self.mapping))
        np.savez(path, **meta)

    @classmethod
    def load(cls, path):
        """Load a TableDensity .npz written by save(). Dimensionality comes
        from the values array; 1D/2D tables carry their mapping in the
        header (legacy 3D saves without one load fine - 3D is (x,y,z))."""
        with np.load(path) as d:
            values = d['values']
            dim = values.ndim
            axes = tuple(d[f'ax{i}'] for i in range(dim))
            if 'mapping' in d.files:
                maps = [str(s) for s in d['mapping']]
                mapping = maps[0] if dim == 1 else tuple(maps)
            elif dim == 3:
                mapping = None
            else:
                raise ValueError(
                    f'{path}: a {dim}D table needs its coordinate mapping in '
                    'the header - re-save it, or convert the original xdr '
                    'with python -m glassgen.density')
        return cls(axes, values, mapping=mapping)

    @classmethod
    def from_xdr(cls, filename, mapping='r_sph'):
        """LEGACY reader for the C generator's densitytable_xdr format -
        kept for the one-shot converter (python -m glassgen.density); the
        tools consume self-describing .npz tables only."""
        with open(filename, 'rb') as f:
            dim = int(f.readline())
            n = int(f.readline())
            float(f.readline())  # R, domain size; axes carry the actual grid
            data = f.read()

        be_u4 = np.dtype('>u4')
        be_f4 = np.dtype('>f4')
        off = 0

        def unpack_array():
            nonlocal off
            count = int(np.frombuffer(data, be_u4, count=1, offset=off)[0])
            off += 4
            arr = np.frombuffer(data, be_f4, count=count, offset=off)
            off += 4 * count
            return arr.astype(np.float64)

        axes = tuple(unpack_array() for _ in range(dim))
        for ax in axes:
            if len(ax) != n:
                raise ValueError(f"axis length {len(ax)} != header n {n}")
        values = unpack_array().reshape((n,) * dim)
        return cls(axes, values, mapping=mapping)

    def rho0(self, pos):
        out = np.empty(len(pos))
        if self.dim == 1:
            _interp1d(self.axes[0], self.values, map_coord(pos, self.mapping), out)
        elif self.dim == 2:
            _interp2d(self.axes[0], self.axes[1], self.values,
                      map_coord(pos, self.mapping[0]),
                      map_coord(pos, self.mapping[1]), out)
        else:
            _interp3d(self.axes[0], self.axes[1], self.axes[2], self.values,
                      np.ascontiguousarray(pos), out)
        return out


class CallableDensity(DensityField):
    """Wrap a callable. With a mapping, fn receives the (N,) mapped
    coordinate; with mapping=None, fn receives the (N,3) positions."""

    def __init__(self, fn, mapping=None):
        self.fn = fn
        self.mapping = mapping

    def rho0(self, pos):
        if self.mapping is None:
            rho = self.fn(pos)
        else:
            rho = self.fn(map_coord(pos, self.mapping))
        return np.asarray(rho, dtype=np.float64)


def _km_stencil(nshell, ndir=26):
    """Fixed kernel-average quadrature stencil in UNIT-SUPPORT coordinates
    (u = r / (2h), |u| <= 1): ``nshell`` radial midpoints x ``ndir``
    SYMMETRIZED-FIBONACCI directions (a golden-angle spiral on the upper
    half-sphere plus its antipodes: near-uniform coverage with NO direction
    lying in a coordinate plane - a cube-lattice stencil puts 8 of 26
    directions exactly in the z=0 plane, so an AXIS-ALIGNED thin slab (the
    galaxy disk) reads ~8/26 of its peak instead of its true kernel fraction;
    antipodal symmetry keeps a LINEAR field exact at the centre). Successive
    shells are rotated about z by the golden angle to decorrelate. Radial
    weights follow the kernel-volume measure u^2 * w_wc2(q=2u), normalised to
    sum 1 (a CONSTANT field is exact). Returns (offsets (K,3), weights (K,))."""
    from .kernels import w_wc2
    nhalf = max(ndir // 2, 3)
    ga = np.pi * (3.0 - np.sqrt(5.0))                 # golden angle
    i = np.arange(nhalf)
    z = (i + 0.5) / nhalf                             # upper half-sphere
    s = np.sqrt(1.0 - z * z)
    phi = i * ga + 0.4                                # offset: avoid phi=0 axis
    half = np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)
    dirs = np.concatenate([half, -half], axis=0)
    u = (np.arange(nshell) + 0.5) / nshell            # radial midpoints
    wr = u * u * np.array([w_wc2(float(4.0 * uu * uu)) for uu in u])
    offsets = np.empty((nshell * len(dirs), 3))
    for si, uu in enumerate(u):                       # golden-rotate per shell
        rot = si * ga
        c, sn = np.cos(rot), np.sin(rot)
        rd = dirs @ np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]).T
        offsets[si * len(dirs):(si + 1) * len(dirs)] = uu * rd
    weights = np.repeat(wr / (wr.sum() * len(dirs)), len(dirs))
    return np.ascontiguousarray(offsets), np.ascontiguousarray(weights)


class KernelMatchedDensity(DensityField):
    """Kernel-matched target readout (RECONSTRUCT_PLAN.md task M4).

    ``rho0_h(pos, h)`` returns the ``base`` field averaged over each
    particle's OWN Wendland-C2 kernel (support radius 2h_i, the glassgen
    convention): a fixed unit-support stencil is scaled by 2h_i per particle
    and the base is read at the K offset points,

        rho0_i = sum_q w_q * base(x_i + 2 h_i * u_q),    sum_q w_q = 1.

    The particle's SPH density IS a kernel average at that same h_i, so the
    pseudo-pressure ratio rho/rho0 compares like with like: where the base is
    smooth on the kernel scale nothing changes (constant exact, linear exact
    by stencil symmetry, quadratic bias ~ the SPH estimator's own); where the
    base is SHARPER than the kernel (the razor-thin galaxy disk in z) the
    ratio -> 1 and the relax converges to the best N-particle representation
    instead of blowing the feature open (the 2026-07-10 hole). Cost: K
    (= 26*nshell) extra table lookups per particle-iteration.

    ``rho0(pos)`` stays the base point read - occupancy cuts, sampling and
    coarse-count criteria keep their sharp view of the target.

    hmode ('particle' default | 'target', 2026-07-13 user proposal): where
    the averaging scale comes from.
      'particle' - the h the relax passes (each particle's own, from its
        realized neighbours). State-DEPENDENT: a drained region's h grows,
        the target peak smooths further, the drained state self-sustains.
      'target' - h_t(x) from the TARGET density at the point, the h a
        particle of mass ``m`` would have AT equilibrium:
        2h_t = (3*nsmooth*m/(4*pi*rho_target))**(1/3). The applied field
        becomes a FIXED function of position (the target as seen by the
        equilibrium ensemble): a drained cusp keeps facing the full sharp
        demand (restoring), and the scale is self-regularizing (h_t grows
        where the target is faint, so off-plane stragglers still read a
        ball average ~ their own SPH estimate - no M4-hole asymmetry).
        The h argument of rho0_h is ignored. Needs ``m`` (= M_gas/N;
        coarse cascade stages with heavier particles still read the
        final-N scale - envelope builds run uncascaded on sharp fields).
      'hybrid' (the M17 verdict arm) - target-h ONLY where the point read
        is resolved signal (>= rho_edge, the occupied disk - where the
        drained-state feedback bites), particle-h in the scaffold (where
        the pure target-h arm firmed up the smear-band demand and piled
        31% of the mass into the wings; the state-dependent ball average
        is what keeps strays soft out there). Needs m and rho_edge."""

    smooth_readout = True

    def __init__(self, base, nshell=2, hmode='particle', m=None, nsmooth=64,
                 rho_edge=None, h_scale=1.0):
        if hmode not in ('particle', 'target', 'hybrid'):
            raise ValueError(f"unknown hmode {hmode!r} "
                             "(use 'particle', 'target' or 'hybrid')")
        if hmode in ('target', 'hybrid') and m is None:
            raise ValueError(f"hmode={hmode!r} needs the particle mass m "
                             "(the equilibrium-h scale is (3*nsmooth*m/"
                             "(4*pi*rho_target))**(1/3))")
        if hmode == 'hybrid' and rho_edge is None:
            raise ValueError("hmode='hybrid' needs rho_edge (the "
                             "signal/scaffold gate)")
        self.base = base
        self.hmode = hmode
        self.m = None if m is None else float(m)
        self.nsmooth = int(nsmooth)
        self.rho_edge = None if rho_edge is None else float(rho_edge)
        # h_scale (2026-07-13 user experiment): read the target at a
        # FRACTION of the particle's h - sharper demand than the full
        # kernel average (less drained-state softening) without the
        # state-independent target-h's off-plane shelf. Applies to the
        # 'particle' scale (and the hybrid's particle side).
        self.h_scale = float(h_scale)
        self.offsets, self.weights = _km_stencil(int(nshell))

    def rho0(self, pos):
        base = self.base
        return base.rho0(pos) if hasattr(base, 'rho0') else \
            np.asarray(base(pos), dtype=np.float64)

    def rho0_h(self, pos, h, chunk=4_000_000):
        # BATCHED stencil evaluation: one read of K*rows points per block
        # instead of K separate reads of len(pos) points. The readout chain
        # (occupancy + base/shell table queries, numba-parallel) is launch-
        # dominated on per-offset calls - the 2026-07-10 galaxy timing probe
        # measured this loop at 94% of the whole relax iteration; batching
        # cuts the K numba launches (and the numpy temp churn around them)
        # to ~1 per chunk. `chunk` bounds the block to ~chunk points
        # (memory ~3*8*chunk bytes for the offset positions).
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        if self.hmode in ('target', 'hybrid'):
            # equilibrium-h readout: derive the scale from the target's own
            # pointwise value; the passed h (the particle's current state)
            # is ignored ('target') or kept in the scaffold ('hybrid')
            v = np.maximum(self.rho0(pos), np.finfo(np.float64).tiny)
            support = ((3.0 * self.nsmooth * self.m / (4.0 * np.pi * v))
                       ** (1.0 / 3.0))[:, None]
            if self.hmode == 'hybrid':
                hp = np.asarray(h, dtype=np.float64)[:, None] \
                    * (2.0 * self.h_scale)
                support = np.where(v[:, None] >= self.rho_edge,
                                   support, hp)
        else:
            support = np.asarray(h, dtype=np.float64)[:, None] \
                * (2.0 * self.h_scale)
        base = self.base
        if hasattr(base, 'rho0_stencil'):
            # fully-fused fast path (envelope.PaddedTableDensity): the
            # weighted stencil sum runs inside one numba kernel
            return base.rho0_stencil(pos, support[:, 0], self.offsets,
                                     self.weights)
        read = base.rho0 if hasattr(base, 'rho0') else base
        n = len(pos)
        k = len(self.offsets)
        out = np.empty(n)
        rows = max(1, int(chunk) // k)
        for i0 in range(0, n, rows):
            sl = slice(i0, min(i0 + rows, n))
            block = (pos[sl, None, :]
                     + support[sl, :, None] * self.offsets[None, :, :])
            v = np.asarray(read(block.reshape(-1, 3)),
                           dtype=np.float64).reshape(-1, k)
            out[sl] = v @ self.weights
        return out


def main(argv=None):
    """One-shot converter: legacy densitytable_xdr -> self-describing .npz.

        python -m glassgen.density densitytable_xdr table.npz --direction r_sph

    The coordinate mapping is baked into the .npz header (this is the ONLY
    place --direction still exists); the tools then take --table table.npz
    with no direction flag (CLEANUP_PLAN C6)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        description='convert a legacy C-generator densitytable_xdr into the '
                    'self-describing .npz density-table format (the mapping '
                    'is stored in the table header)')
    ap.add_argument('xdr', help='input densitytable_xdr file')
    ap.add_argument('out', help='output .npz path')
    ap.add_argument('--direction', nargs='+', default=['r_sph'],
                    choices=MAPPINGS,
                    help='coordinate mapping the table is indexed by: one '
                         'value for a 1D table, two for a 2D table '
                         '(default r_sph); ignored for 3D tables, which '
                         'are indexed by (x, y, z)')
    args = ap.parse_args(argv)
    mapping = (args.direction[0] if len(args.direction) == 1
               else tuple(args.direction))
    try:
        t = TableDensity.from_xdr(args.xdr, mapping=mapping)
    except ValueError as e:
        sys.exit(f'{args.xdr}: {e}')
    # np.savez appends .npz itself when missing - name it up front instead
    out = args.out if args.out.endswith('.npz') else args.out + '.npz'
    t.save(out)
    desc = 'x'.join(str(len(a)) for a in t.axes)
    print(f'wrote {out}: {t.dim}D table {desc}'
          + (f', mapping {t.mapping}' if t.mapping else ' (x,y,z indexed)'))


if __name__ == '__main__':
    main()
