import numpy as np

from sph_interp.interpolate import GridResult
from sph_interp.particles import Particles
from sph_interp.targets import AMRGrid, UniformGrid
from sph_interp.visualise import (_amr_containing_cells, _pdf_curve,
                                  _shared_norm, resolution_audit,
                                  sample_at_particles, visualise)


def test_log_pdf_is_probability_per_dex():
    values = np.array([2.0, 20.0])
    weights = np.array([1.0, 3.0])
    edges = np.array([1.0, 10.0, 100.0])
    pdf = _pdf_curve(values, weights, edges, log=True)

    np.testing.assert_allclose(pdf, [0.25, 0.75])
    np.testing.assert_allclose(
        np.sum(pdf * np.diff(np.log10(edges))), 1.0)


def test_log_pdf_does_not_renormalise_probability_below_plot_range():
    values = np.array([0.2, 2.0, 20.0])
    weights = np.array([4.0, 1.0, 3.0])
    edges = np.array([1.0, 10.0, 100.0])
    pdf = _pdf_curve(values, weights, edges, log=True)

    np.testing.assert_allclose(pdf, [0.125, 0.375])
    np.testing.assert_allclose(
        np.sum(pdf * np.diff(np.log10(edges))), 0.5)


def test_shared_norm_covers_robust_particle_and_grid_ranges_equally():
    particles = np.geomspace(1.0e-4, 10.0, 10000)
    grid = np.geomspace(1.0e-6, 0.3, 100)

    norm = _shared_norm((particles, grid), log=True, robust=True)

    assert norm.vmin <= np.quantile(grid, 0.01)
    assert norm.vmax >= np.quantile(particles, 0.995)


def test_resolution_audit_compares_finest_cell_with_small_h_tail():
    particles = Particles(
        pos=np.zeros((3, 3)), mass=np.ones(3),
        rho=np.ones(3), h=np.array([0.1, 0.2, 0.3]),
        values={"rho": np.ones(3)}, periodic=False)
    grid = UniformGrid(lo=np.zeros(3), hi=np.ones(3), npx=2)

    audit = resolution_audit(grid, particles)

    np.testing.assert_allclose(audit["dx_min"], 0.5)
    np.testing.assert_allclose(audit["h_min"], 0.1)
    np.testing.assert_allclose(audit["h_p05"], 0.11)
    np.testing.assert_allclose(audit["dx_over_h_min"], 5.0)


def _mixed_level_grid():
    # Left base cell stays coarse. The right base cell is split into 8 leaves.
    cell_lo = [[0.0, 0.0, 0.0]]
    cell_size = [[0.5, 1.0, 1.0]]
    level = [0]
    base = [0]
    for x in (0.5, 0.75):
        for y in (0.0, 0.5):
            for z in (0.0, 0.5):
                cell_lo.append([x, y, z])
                cell_size.append([0.25, 0.5, 0.5])
                level.append(1)
                base.append(1)
    return AMRGrid(
        lo=np.zeros(3), hi=np.ones(3), base_npx=(2, 1, 1),
        cell_lo=np.asarray(cell_lo), cell_size=np.asarray(cell_size),
        cell_level=np.asarray(level), cell_base=np.asarray(base),
        base_start=np.array([0, 1]), base_count=np.array([1, 8]),
        periodic=False)


def test_amr_sampling_uses_containing_leaf_not_nearest_centre():
    grid = _mixed_level_grid()
    pos = np.array([[0.49, 0.10, 0.10], [0.51, 0.10, 0.10]])
    particles = Particles(
        pos=pos, mass=np.ones(2), rho=np.ones(2), h=np.full(2, 0.01),
        values={"rho": np.ones(2)}, periodic=False)
    values = np.arange(grid.ncell, dtype=float) + 10.0
    result = GridResult(
        data={"rho": values}, norm=None, target=grid, method="petkova")

    cells = _amr_containing_cells(grid, pos)
    assert cells[0] == 0
    np.testing.assert_array_equal(
        sample_at_particles(result, particles, "rho"), values[cells])


def test_particle_panel_uses_mass_centre_and_kernel_intersections():
    core = np.zeros((20, 3))
    core[:, 0] = np.linspace(-0.5, 0.5, len(core))
    outlier = np.array([[0.0, 0.0, 10.0]])
    pos = np.vstack([core, outlier])
    mass = np.r_[np.ones(len(core)), 1.0e-6]
    particles = Particles(
        pos=pos, mass=mass, rho=np.ones(len(pos)), h=np.full(len(pos), 0.02),
        values={"rho": np.ones(len(pos))}, periodic=False)
    grid = UniformGrid(lo=-np.ones(3), hi=np.ones(3), npx=4)
    rho = np.ones((4, 4, 4))
    result = GridResult(
        data={"rho": rho, "_mass": np.ones_like(rho)}, norm=None,
        target=grid, method="petkova")

    fig = visualise(result, particles, field="rho", center="mass", log=True)
    title = fig.axes[0].get_title()
    assert "kernels intersect z=" in title
    assert "(20 pts)" in title
    pdf_ax = next(ax for ax in fig.axes
                  if ax.get_title().endswith("distribution"))
    np.testing.assert_allclose(pdf_ax.get_xlim()[0], 1.0 / 10.0**0.05)
    assert any("dx_min/h_min=" in text.get_text() for text in fig.texts)
