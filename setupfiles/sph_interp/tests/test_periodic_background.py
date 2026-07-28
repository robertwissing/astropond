"""Regression tests for sparse periodic Voronoi background generators."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from sph_interp import Particles, VoronoiGrid, kernel_support_bounds
from sph_interp import targets
from IC_interpolate_to_grid import (build_parser, build_target,
                                    support_domain_audit, target_domain)


def _compact_points(n=256):
    rng = np.random.default_rng(9182)
    return np.clip(0.04 * rng.normal(size=(n, 3)), -0.12, 0.12)


def test_periodic_coverage_detects_compact_but_not_uniform_generators():
    lo = np.full(3, -0.5)
    hi = np.full(3, 0.5)
    compact_radius, safe_radius = VoronoiGrid._periodic_coverage_radius(
        _compact_points(), lo, hi, 4)

    x = (np.arange(8) + 0.5) / 8.0 - 0.5
    uniform = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    uniform_radius, _ = VoronoiGrid._periodic_coverage_radius(
        uniform, lo, hi, 4)

    assert compact_radius > safe_radius
    assert uniform_radius < safe_radius


@pytest.mark.parametrize("backend", ["pyvoro", "multivoro"])
def test_auto_background_closes_compact_periodic_mesh(capsys, backend):
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"optional {backend} backend is not installed")
    points = _compact_points()
    original = points.copy()
    grid = VoronoiGrid.from_points(
        points, bounds=[[-0.5, 0.5]] * 3, periodic=True,
        backend=backend, merge_tol=0)

    assert np.array_equal(points, original)  # canonical wrapping owns its copy
    assert grid.volumes().sum() == pytest.approx(1.0, rel=1e-10, abs=1e-12)
    message = capsys.readouterr().out
    assert "adding 16^3 = 4096 background generators" in message
    assert "half-period cells" in message


def test_open_auto_background_routes_arepo_through_guarded_box(
        monkeypatch, capsys):
    points = _compact_points(8)
    sentinel = object()
    captured = {}

    monkeypatch.setattr(targets, "_resolve_backend", lambda backend: "arepo")
    monkeypatch.setattr(
        VoronoiGrid, "_merge_close",
        staticmethod(lambda pts, tol: (pts, 0)))

    def guarded(cls, pts, lo, hi, n_threads):
        captured["npoint"] = len(pts)
        captured["lo"] = lo
        captured["hi"] = hi
        return sentinel

    monkeypatch.setattr(
        VoronoiGrid, "_from_arepo_guarded_open", classmethod(guarded))
    result = VoronoiGrid.from_points(
        points, bounds=[[-0.5, 0.5]] * 3, periodic=False,
        backend="auto", merge_tol=0, periodic_background="auto")

    assert result is sentinel
    assert captured["npoint"] == len(points) + 16 ** 3
    assert "open-domain 16^3 = 4096 background guard generators" in (
        capsys.readouterr().out)


def test_kernel_support_bounds_is_union_of_individual_2h_boxes():
    pos = np.array([[0.0, 1.0, 2.0], [10.0, -2.0, 4.0]])
    h = np.array([1.0, 3.0])

    lo, hi = kernel_support_bounds(pos, h)

    np.testing.assert_allclose(lo, [-2.0, -8.0, -2.0])
    np.testing.assert_allclose(hi, [16.0, 4.0, 10.0])


def test_support_domain_audit_flags_large_low_density_reach():
    particles = SimpleNamespace(
        pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        h=np.array([0.1, 10.0]),
        rho=np.array([1.0, 1.0e-8]))

    audit = support_domain_audit(
        particles, np.full(3, -2.0), np.full(3, 2.0))

    assert audit["large"]
    assert audit["culprit"] == 1
    assert audit["volume_factor"] > 100.0
    np.testing.assert_allclose(audit["support_lo"], [-19.0, -20.0, -20.0])
    np.testing.assert_allclose(audit["support_hi"], [21.0, 20.0, 20.0])


def _open_domain_case(grid, capture_domain, interactive=False, cubic=None):
    particles = SimpleNamespace(
        pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        h=np.array([0.1, 10.0]),
        rho=np.array([1.0, 1.0e-8]),
        mass=np.ones(2),
        periodic=(False, False, False),
        box=np.full(3, 4.0))
    args = SimpleNamespace(
        box=np.full(3, 4.0), grid=grid, cubic=cubic,
        capture_domain=capture_domain, _interactive=interactive,
        open_padding="rho_edge_pad", open_pad_fraction=0.1, nsmooth=64)
    return args, particles


@pytest.mark.parametrize("grid", ["uniform", "amr", "voronoi"])
def test_full_support_capture_is_common_to_every_open_3d_target(grid):
    args, particles = _open_domain_case(grid, "full")

    lo, hi = target_domain(args, particles)

    np.testing.assert_allclose(lo, [-19.0, -20.0, -20.0])
    np.testing.assert_allclose(hi, [21.0, 20.0, 20.0])
    assert not args._capture_clipped


def test_default_common_capture_keeps_selected_domain_without_prompt(
        monkeypatch, capsys):
    args, particles = _open_domain_case(
        "uniform", capture_domain=None, interactive=True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("support warning must not prompt"))

    lo, hi = target_domain(args, particles)

    np.testing.assert_allclose(lo, -2.0)
    np.testing.assert_allclose(hi, 2.0)
    assert args._capture_clipped
    assert args._capture_domain == "domain"
    warning = capsys.readouterr().out
    assert "Rerun with --capture-domain full" in warning
    assert "full 2h:" in warning
    assert "lo=[-19." in warning


def test_full_support_capture_preserves_requested_cubic_domain():
    args, particles = _open_domain_case(
        "amr", capture_domain="full", cubic="domain")

    lo, hi = target_domain(args, particles)

    np.testing.assert_allclose(hi - lo, np.full(3, (hi - lo)[0]))
    assert np.all(lo <= np.array([-19.0, -20.0, -20.0]))
    assert np.all(hi >= np.array([21.0, 20.0, 20.0]))


def test_amr_reports_reached_level_separately_from_user_cap(capsys):
    particles = Particles(
        pos=np.zeros((1, 3)), mass=np.ones(1), rho=np.ones(1),
        h=np.full(1, 0.1), values={"rho": np.ones(1)}, periodic=False,
        box=np.full(3, 4.0))
    args = SimpleNamespace(
        grid="amr", box=np.full(3, 4.0), cubic=None,
        capture_domain="domain", _interactive=False,
        open_padding="rho_edge_pad", open_pad_fraction=0.1, nsmooth=64,
        base_npx=2, max_level=4, npx=32,
        refine="count", refine_value=None, grade=True)

    grid = build_target(args, particles)

    assert grid.cell_level.max() == 0
    output = capsys.readouterr().out
    assert "max_level reached=0 (cap=4)" in output
    assert "finest occupied leaf level" in output
    assert "applying --grade" in output
    assert "grade=2:1" in output
    assert "count refinement threshold=8 particles (default)" in output


def test_mass_refinement_defaults_to_eight_mean_particle_masses(capsys):
    particles = Particles(
        pos=np.zeros((1, 3)), mass=np.full(1, 2.0), rho=np.ones(1),
        h=np.full(1, 0.1), values={"rho": np.ones(1)}, periodic=False,
        box=np.full(3, 4.0))
    args = SimpleNamespace(
        grid="amr", box=np.full(3, 4.0), cubic=None,
        capture_domain="domain", _interactive=False,
        open_padding="rho_edge_pad", open_pad_fraction=0.1, nsmooth=64,
        base_npx=2, max_level=4, npx=32,
        refine="mass", refine_value=None, grade=False)

    build_target(args, particles)

    output = capsys.readouterr().out
    assert "mass refinement threshold=16 = 8 x mean particle mass (2)" in output


def test_grade_is_default_and_no_grade_is_explicit_opt_out():
    parser = build_parser()

    plain = parser.parse_args(["input", "output"])
    graded = parser.parse_args(["input", "output", "--grade"])
    ungraded = parser.parse_args(["input", "output", "--no-grade"])

    assert plain.grade is True
    assert graded.grade is True
    assert ungraded.grade is False
