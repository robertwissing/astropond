import struct

import numpy as np
import pytest

from glassgen.density import CallableDensity, TableDensity, map_coord


def write_xdr_table(path, dim, n, R, axes, values):
    """Reference writer matching setupfiles/IC_denstable.py (xdrlib format:
    each array prefixed by a big-endian u32 length, big-endian f32 data)."""
    with open(path, 'wb') as f:
        f.write(f"{dim}\n{n}\n{R:.15g}\n".encode('ascii'))
        for ax in axes:
            f.write(struct.pack('>I', len(ax)))
            f.write(np.asarray(ax, dtype='>f4').tobytes())
        flat = np.asarray(values, dtype='>f4').ravel()
        f.write(struct.pack('>I', len(flat)))
        f.write(flat.tobytes())


def test_xdr_roundtrip_1d(tmp_path):
    n = 100
    x = np.linspace(0, 2.0, n)
    rho = 1.0 + np.exp(-x)
    path = tmp_path / 'densitytable_xdr'
    write_xdr_table(path, 1, n, 2.0, (x,), rho)
    tab = TableDensity.from_xdr(str(path), mapping='r_sph')
    assert tab.dim == 1
    assert np.allclose(tab.axes[0], x, atol=1e-6)
    assert np.allclose(tab.values, rho, atol=1e-6)


def test_xdr_roundtrip_2d(tmp_path):
    n = 16
    x = np.linspace(-1, 1, n)
    vals = np.add.outer(x ** 2, x)  # (n, n), C-order
    path = tmp_path / 'tab2d'
    write_xdr_table(path, 2, n, 1.0, (x, x), vals)
    tab = TableDensity.from_xdr(str(path), mapping=('x', 'y'))
    assert tab.dim == 2
    assert np.allclose(tab.values, vals, atol=1e-5)


def test_linear_table_interp_is_exact():
    x = np.linspace(0, 1, 11)
    tab = TableDensity((x,), 2.0 + 3.0 * x, mapping='x')
    pos = np.zeros((50, 3))
    pos[:, 0] = np.linspace(0.01, 0.99, 50)
    assert np.allclose(tab.rho0(pos), 2.0 + 3.0 * pos[:, 0])


def test_interp_clamps_at_edges():
    x = np.linspace(0, 1, 11)
    tab = TableDensity((x,), 2.0 + 3.0 * x, mapping='x')
    pos = np.array([[5.0, 0, 0], [-5.0, 0, 0]])
    assert np.allclose(tab.rho0(pos), [5.0, 5.0])  # |x| mapping + clamp


def test_mappings():
    pos = np.array([[3.0, -4.0, 12.0]])
    assert map_coord(pos, 'x')[0] == 3.0
    assert map_coord(pos, 'y')[0] == 4.0
    assert map_coord(pos, 'z')[0] == 12.0
    assert np.isclose(map_coord(pos, 'r_sph')[0], 13.0)
    assert np.isclose(map_coord(pos, 'r_cyl')[0], 5.0)
    pos2 = np.array([[-3.0, 0.0, 0.0]])
    assert map_coord(pos2, 'signed_x')[0] == -3.0
    with pytest.raises(ValueError):
        map_coord(pos, 'bogus')


def test_callable_vs_table_agree():
    fn = lambda r: 1.0 + np.exp(-r * r)
    r = np.linspace(0, 3, 3000)
    tab = TableDensity((r,), fn(r), mapping='r_sph')
    cal = CallableDensity(fn, mapping='r_sph')
    rng = np.random.default_rng(5)
    pos = rng.uniform(-1, 1, (200, 3))
    assert np.allclose(tab.rho0(pos), cal.rho0(pos), rtol=1e-5)


def test_3d_table():
    x = np.linspace(-1, 1, 21)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    vals = 1.0 + X + 2 * Y + 3 * Z  # trilinear -> interp exact
    tab = TableDensity((x, x, x), vals)
    rng = np.random.default_rng(6)
    pos = rng.uniform(-0.9, 0.9, (100, 3))
    expect = 1.0 + pos[:, 0] + 2 * pos[:, 1] + 3 * pos[:, 2]
    assert np.allclose(tab.rho0(pos), expect)


# --------------------------------------------------------------------------- #
#  M4: kernel-matched target readout (KernelMatchedDensity)
# --------------------------------------------------------------------------- #

def test_kernel_matched_constant_exact():
    from glassgen.density import CallableDensity, KernelMatchedDensity
    km = KernelMatchedDensity(CallableDensity(lambda p: np.full(len(p), 3.7)))
    pos = np.random.default_rng(0).random((100, 3))
    h = np.full(100, 0.3)
    assert np.allclose(km.rho0_h(pos, h), 3.7)
    assert np.allclose(km.rho0(pos), 3.7)        # point read passthrough


def test_kernel_matched_linear_exact():
    # antipodally symmetric stencil: a linear field averages to its centre
    from glassgen.density import CallableDensity, KernelMatchedDensity
    km = KernelMatchedDensity(CallableDensity(
        lambda p: 5.0 + 2.0 * p[:, 0] - 1.0 * p[:, 2]))
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    h = np.array([0.5, 1.0])
    expect = 5.0 + 2.0 * pos[:, 0] - 1.0 * pos[:, 2]
    assert np.allclose(km.rho0_h(pos, h), expect)


def test_kernel_matched_resolved_matches_point_read():
    # smooth gaussian, kernel much smaller than its scale: ~ the point read
    from glassgen.density import CallableDensity, KernelMatchedDensity
    base = CallableDensity(lambda p: np.exp(-(p ** 2).sum(1) / 2.0))
    km = KernelMatchedDensity(base)
    pos = np.random.default_rng(1).normal(size=(200, 3)) * 0.5
    h = np.full(200, 0.01)                        # support 0.02 << scale 1
    assert np.allclose(km.rho0_h(pos, h), base.rho0(pos), rtol=1e-3)


def test_kernel_matched_thin_slab_reads_average_not_point():
    # slab of thickness 0.1 << kernel support: the mid-plane readout is the
    # kernel-smeared slab (way below the peak), and it grows back toward the
    # peak as h shrinks - the hole mechanism and its fix in one assert
    from glassgen.density import CallableDensity, KernelMatchedDensity
    base = CallableDensity(
        lambda p: np.where(np.abs(p[:, 2]) < 0.05, 10.0, 1e-6))
    km = KernelMatchedDensity(base)
    mid = np.zeros((1, 3))
    wide = km.rho0_h(mid, np.array([1.0]))[0]     # support 2 >> slab 0.1
    narrow = km.rho0_h(mid, np.array([0.02]))[0]  # support 0.04 < slab
    assert wide < 1.0                             # smeared far below the peak
    assert np.isclose(narrow, 10.0)               # resolved: the point value
    # an off-plane point just outside the slab still sees slab mass through
    # its kernel -> target >> the vacuum point read (no fake overpressure)
    off = np.array([[0.0, 0.0, 0.1]])
    assert km.rho0_h(off, np.array([0.2]))[0] > 1e3 * base.rho0(off)[0]


def test_kernel_matched_target_hmode():
    # hmode='target' (2026-07-13): the averaging scale comes from the
    # target's own density - 2h_t = (3*nsmooth*m/(4pi*rho0))^(1/3) - and
    # the passed h is IGNORED (the applied field is state-independent)
    import pytest
    from glassgen.density import CallableDensity, KernelMatchedDensity
    base = CallableDensity(lambda p: np.exp(-(p ** 2).sum(1) / 2.0))
    m, ns = 1e-4, 64
    kmt = KernelMatchedDensity(base, hmode='target', m=m, nsmooth=ns)
    kmp = KernelMatchedDensity(base)
    pos = np.random.default_rng(2).normal(size=(50, 3)) * 0.5
    # equivalence at equilibrium: particle-h == the derived target-h
    h_eq = 0.5 * (3.0 * ns * m / (4.0 * np.pi * base.rho0(pos))) ** (1 / 3)
    ref = kmp.rho0_h(pos, h_eq)
    assert np.allclose(kmt.rho0_h(pos, h_eq), ref)
    # state-independence: garbage h gives the same answer
    assert np.allclose(kmt.rho0_h(pos, 100.0 * h_eq), ref)
    assert np.allclose(kmt.rho0_h(pos, None), ref)
    # the particle mode DOES depend on h (sanity of the contrast)
    assert not np.allclose(kmp.rho0_h(pos, 100.0 * h_eq), ref)
    # guards
    with pytest.raises(ValueError):
        KernelMatchedDensity(base, hmode='target')          # needs m
    with pytest.raises(ValueError):
        KernelMatchedDensity(base, hmode='equilibrium')
    with pytest.raises(ValueError):
        KernelMatchedDensity(base, hmode='hybrid', m=m)     # needs rho_edge
    # hybrid: target-h where the point read >= rho_edge, particle-h below
    re = float(base.rho0(np.array([[0.8, 0.0, 0.0]])))
    kmh = KernelMatchedDensity(base, hmode='hybrid', m=m, nsmooth=ns,
                               rho_edge=re)
    inner = pos[np.linalg.norm(pos, axis=1) < 0.5]     # read >= re
    outer = 2.0 * pos[np.linalg.norm(pos, axis=1) > 0.6]
    h_odd = np.full(len(inner), 7.7)
    assert np.allclose(kmh.rho0_h(inner, h_odd),
                       kmt.rho0_h(inner, h_odd))       # target side
    h_out = np.full(len(outer), 0.05)
    assert np.allclose(kmh.rho0_h(outer, h_out),
                       kmp.rho0_h(outer, h_out))       # particle side


def test_kernel_matched_stencil_weights():
    from glassgen.density import _km_stencil
    off, w = _km_stencil(2)
    assert off.shape == (52, 3) and w.shape == (52,)
    assert np.isclose(w.sum(), 1.0)
    assert (np.linalg.norm(off, axis=1) <= 1.0 + 1e-12).all()
    # antipodal symmetry (linear exactness): every offset's negation is
    # present (same shell, same weight by construction)
    setk = {tuple(x) for x in np.round(off, 12)}
    assert all(tuple(x) in setk for x in np.round(-off, 12))
    # no direction in a coordinate plane: an AXIS-ALIGNED thin slab (the
    # galaxy disk) must not be oversampled the way a cube-lattice stencil's
    # 8 in-plane directions oversample it
    dirz = np.abs(off[:, 2]) / np.linalg.norm(off, axis=1)
    assert dirz.min() > 0.01
    # and the slab readout is near the TRUE kernel average (0.746 for a
    # 0.1-thick rho=10 slab under a support-2 kernel, Monte-Carlo reference)
    from glassgen.density import CallableDensity, KernelMatchedDensity
    km = KernelMatchedDensity(CallableDensity(
        lambda p: np.where(np.abs(p[:, 2]) < 0.05, 10.0, 1e-6)))
    v = km.rho0_h(np.zeros((1, 3)), np.array([1.0]))[0]
    assert 0.35 < v < 1.5
