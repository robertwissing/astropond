import numpy as np
import pytest

from glassgen import reconstruct


try:
    si = reconstruct._import_sph_interp()
    HAVE_SPH_INTERP = True
except ImportError:
    si = None
    HAVE_SPH_INTERP = False


needs_sph_interp = pytest.mark.skipif(
    not HAVE_SPH_INTERP, reason="sph_interp not found in testsuite layout")


@needs_sph_interp
def test_periodic_face_table_rt0_divergence_and_normal_continuity(tmp_path):
    npx = (4, 5, 6)
    lo = -0.5 * np.ones(3)
    box = np.ones(3)
    ix, iy, iz = np.meshgrid(
        np.arange(npx[0]), np.arange(npx[1]), np.arange(npx[2]), indexing="ij")
    # Each normal component is independent of its own normal coordinate, so the
    # face field has exactly zero finite-volume (and RT0 pointwise) divergence.
    bx = 0.4 + np.sin(2.0 * np.pi * (iy + 0.5) / npx[1])
    by = -0.2 + 0.3 * np.cos(2.0 * np.pi * (iz + 0.5) / npx[2])
    bz = 0.1 + 0.2 * np.sin(2.0 * np.pi * (ix + 0.5) / npx[0])
    table = si.FaceMagneticTable(lo, box, npx, bx, by, bz)
    np.testing.assert_array_equal(table.cell_divergence(), 0.0)

    # Approach one x face from both cells.  Bx is the shared normal trace and
    # converges to the same face value; tangential components need not match.
    xface = lo[0] + 2.0 * table.pixwidth[0]
    eps = 1.0e-12
    q = np.array([[xface - eps, 0.07, -0.11],
                  [xface + eps, 0.07, -0.11],
                  [xface, 0.07, -0.11]])
    values = table.query(q)
    assert values[0, 0] == pytest.approx(values[1, 0], abs=1e-10)
    assert values[2, 0] == pytest.approx(values[0, 0], abs=1e-10)

    path = tmp_path / "face_B.npz"
    table.save(path, metadata={"fixture": "rt0"})
    loaded, metadata = si.FaceMagneticTable.load(path)
    assert metadata == {"fixture": "rt0"}
    np.testing.assert_array_equal(loaded.query(q), values)


@needs_sph_interp
def test_balsara2_open_affine_field_is_exact_and_divergence_free():
    """A trace-free affine field lies exactly in the Balsara2 space."""
    npx = (4, 5, 6)
    lo = -0.5 * np.ones(3)
    box = np.ones(3)
    spacing = box / np.asarray(npx)
    offset = np.asarray([0.4, -0.3, 0.2])
    matrix = np.asarray([[0.3, 0.7, -0.2],
                         [-0.5, -0.1, 0.4],
                         [0.6, -0.8, -0.2]])
    assert np.trace(matrix) == pytest.approx(0.0)

    def component_on_faces(component):
        shape = list(npx)
        shape[component] += 1
        coordinates = []
        for axis, count in enumerate(shape):
            if axis == component:
                coordinates.append(lo[axis] + np.arange(count) * spacing[axis])
            else:
                coordinates.append(
                    lo[axis] + (np.arange(count) + 0.5) * spacing[axis])
        mesh = np.meshgrid(*coordinates, indexing="ij")
        position = np.stack(mesh, axis=-1)
        return offset[component] + np.einsum(
            "j,...j->...", matrix[component], position)

    table = si.OpenFaceMagneticTable(
        lo, box, npx, *(component_on_faces(i) for i in range(3)))
    reconstruction = table.balsara2_reconstruction(limiter="mc")
    rng = np.random.default_rng(991)
    positions = rng.uniform(lo, lo + box, size=(400, 3))
    expected = offset + positions @ matrix.T

    np.testing.assert_allclose(reconstruction.query(positions), expected,
                               rtol=0.0, atol=8e-15)
    np.testing.assert_allclose(reconstruction.divergence(positions), 0.0,
                               rtol=0.0, atol=8e-15)
    np.testing.assert_allclose(reconstruction.divergence_coefficients(), 0.0,
                               rtol=0.0, atol=8e-15)
    expected_faces = np.stack([
        np.stack([table.Bx_face[:-1], table.Bx_face[1:]], axis=-1),
        np.stack([table.By_face[:, :-1], table.By_face[:, 1:]], axis=-1),
        np.stack([table.Bz_face[:, :, :-1], table.Bz_face[:, :, 1:]], axis=-1),
    ], axis=-2)
    np.testing.assert_allclose(reconstruction.face_averages(), expected_faces,
                               rtol=0.0, atol=8e-15)
    assert np.sqrt(np.mean((table.query(positions) - expected) ** 2)) > 0.01

    # The exact integral on the symmetric unit cube is known analytically.
    expected_energy = 0.5 * (
        np.dot(offset, offset) + np.sum(matrix * matrix) / 12.0)
    assert reconstruction.magnetic_energy() == pytest.approx(
        expected_energy, rel=2e-14, abs=2e-15)


@needs_sph_interp
def test_balsara2_periodic_preserves_shared_normal_trace_and_face_divergence():
    npx = (7, 8, 9)
    lo = -0.5 * np.ones(3)
    box = np.ones(3)
    dx = box / np.asarray(npx)
    x = lo[0] + (np.arange(npx[0]) + 0.5) * dx[0]
    y = lo[1] + (np.arange(npx[1]) + 0.5) * dx[1]
    z = lo[2] + (np.arange(npx[2]) + 0.5) * dx[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    # Exact transverse cell averages of a smooth, periodic solenoidal field.
    sy = np.sinc(1.0 / npx[1])
    sz = np.sinc(1.0 / npx[2])
    sx = np.sinc(1.0 / npx[0])
    table = si.FaceMagneticTable(
        lo, box, npx,
        0.3 + sy * np.sin(2.0 * np.pi * Y),
        -0.2 + 0.5 * sz * np.cos(2.0 * np.pi * Z),
        0.1 + 0.4 * sx * np.sin(2.0 * np.pi * X))
    reconstruction = table.balsara2_reconstruction(limiter="mc")
    np.testing.assert_array_equal(table.cell_divergence(), 0.0)
    np.testing.assert_allclose(reconstruction.divergence_coefficients(), 0.0,
                               rtol=0.0, atol=2e-14)

    rng = np.random.default_rng(992)
    positions = rng.uniform(lo, lo + box, size=(5000, 3))
    exact = np.column_stack([
        0.3 + np.sin(2.0 * np.pi * positions[:, 1]),
        -0.2 + 0.5 * np.cos(2.0 * np.pi * positions[:, 2]),
        0.1 + 0.4 * np.sin(2.0 * np.pi * positions[:, 0]),
    ])
    rt0_error = np.sqrt(np.mean((table.query(positions) - exact) ** 2))
    balsara_error = np.sqrt(np.mean(
        (reconstruction.query(positions) - exact) ** 2))
    assert balsara_error < 0.55 * rt0_error

    # Approach a shared x-face from both cells.  The shared face polynomial,
    # not merely its average, gives a single-valued normal component.
    face_x = lo[0] + 3.0 * dx[0]
    eps = 1.0e-12
    points = np.asarray([[face_x - eps, 0.073, -0.119],
                         [face_x + eps, 0.073, -0.119],
                         [face_x, 0.073, -0.119]])
    values = reconstruction.query(points)
    assert values[0, 0] == pytest.approx(values[1, 0], abs=2e-10)
    assert values[2, 0] == pytest.approx(values[0, 0], abs=2e-10)


@needs_sph_interp
def test_petkova_B_over_rho_capture_conserves_carrier_and_face_divergence():
    nside = 5
    g = (np.arange(nside) + 0.5) / nside - 0.5
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(8)
    pos += rng.uniform(-0.015, 0.015, size=pos.shape)
    n = len(pos)
    mass = np.full(n, 1.0 / n)
    rho = 1.0 + 0.15 * np.sin(2.0 * np.pi * pos[:, 0])
    h = np.full(n, 0.17)
    values = {
        "Bx": 0.3 + np.sin(2.0 * np.pi * pos[:, 1]),
        "By": -0.2 + 0.7 * np.sin(2.0 * np.pi * pos[:, 0]),
        "Bz": 0.1 + 0.4 * np.cos(2.0 * np.pi * pos[:, 1]),
    }
    particles = si.Particles(pos=pos, mass=mass, rho=rho, h=h,
                             values=values, box=np.ones(3), periodic=True)
    grid = si.UniformGrid.centered(np.ones(3), npx=6)
    table, audit, result = si.capture_periodic_face_field(particles, grid)

    assert result.method == "petkova-B-over-rho"
    carrier = audit["particle_to_raw_integral_B"]
    assert carrier["relative_norm"] < 2e-8
    assert abs(audit["volume_relative_error"]) < 2e-8
    assert audit["raw_to_projected_integral_B"]["relative_norm"] < 2e-13
    assert audit["face_divB_max"] < 2e-13
    assert np.max(np.abs(table.cell_divergence())) < 2e-13

    sampled = table.query(pos)
    assert sampled.shape == (n, 3)
    assert np.all(np.isfinite(sampled))
    particle_audit = si.sph_divergence_audit(
        pos, mass, rho, h, sampled, box=np.ones(3), periodic=True)
    assert particle_audit["operator"] == "difference-SPH-cubic-spline-h_i"
    assert np.all(np.isfinite(particle_audit["divB"]))
    assert particle_audit["normalized_max"] >= 0.0
    imprinted, imprint_audit = si.imprint_periodic_face_field(table, particles)
    np.testing.assert_array_equal(imprinted, sampled)
    np.testing.assert_array_equal(imprint_audit["divB"],
                                  particle_audit["divB"])
    assert audit["magnetic_energy"]["source_particle_volume_quadrature"] > 0.0
    assert audit["magnetic_energy"]["projected_rt0"] > 0.0
    balsara_B, balsara_audit = si.imprint_periodic_face_field(
        table, particles, reconstruction="balsara2")
    np.testing.assert_array_equal(
        balsara_B, table.balsara2_reconstruction().query(particles.pos))
    assert balsara_audit["reconstruction"] == "balsara2"
    assert balsara_audit["polynomial_divB_coefficient_max"] < 2e-13


@needs_sph_interp
def test_sph_divergence_annihilates_constant_field():
    rng = np.random.default_rng(12)
    pos = rng.uniform(-0.5, 0.5, size=(150, 3))
    mass = np.full(len(pos), 1.0 / len(pos))
    rho = np.ones(len(pos))
    h = np.full(len(pos), 0.2)
    B = np.tile([0.4, -0.7, 0.2], (len(pos), 1))
    div = si.sph_divergence(pos, mass, rho, h, B,
                            box=np.ones(3), periodic=True)
    np.testing.assert_array_equal(div, 0.0)


@needs_sph_interp
def test_open_uniform_hodge_has_zero_boundary_flux_and_divergence():
    grid = si.UniformGrid(-0.5 * np.ones(3), 0.5 * np.ones(3), (5, 6, 7))
    rng = np.random.default_rng(21)
    result = si.GridResult(
        {name: rng.normal(size=grid.npx) for name in ("Bx", "By", "Bz")},
        None, grid, "fixture")
    projected = si.project_solenoidal_open(result, add_to_result=False)
    assert np.max(np.abs(projected["divB"])) < 2e-13
    np.testing.assert_array_equal(projected["Bx_face"][[0, -1]], 0.0)
    np.testing.assert_array_equal(projected["By_face"][:, [0, -1]], 0.0)
    np.testing.assert_array_equal(projected["Bz_face"][:, :, [0, -1]], 0.0)


@needs_sph_interp
def test_open_uniform_hodge_preserves_mean_and_constant_background():
    grid = si.UniformGrid(-0.5 * np.ones(3), 0.5 * np.ones(3), (5, 6, 7))
    rng = np.random.default_rng(211)
    raw = {name: rng.normal(size=grid.npx) for name in ("Bx", "By", "Bz")}
    result = si.GridResult(raw, None, grid, "fixture")
    projected = si.project_solenoidal_open(
        result, add_to_result=False, background="mean")
    table = si.OpenFaceMagneticTable.from_projection(grid, projected)
    expected_mean = np.asarray([np.mean(raw[name])
                                for name in ("Bx", "By", "Bz")])

    assert np.max(np.abs(projected["divB"])) < 3e-13
    np.testing.assert_allclose(
        np.mean(table.cell_centred(), axis=(0, 1, 2)), expected_mean,
        rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(projected["Bx_face"][[0, -1]],
                               expected_mean[0], rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(projected["By_face"][:, [0, -1]],
                               expected_mean[1], rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(projected["Bz_face"][:, :, [0, -1]],
                               expected_mean[2], rtol=0.0, atol=2e-16)

    constant = np.asarray([0.4, -0.7, 0.2])
    constant_result = si.GridResult(
        {name: np.full(grid.npx, value)
         for name, value in zip(("Bx", "By", "Bz"), constant)},
        None, grid, "constant")
    constant_projection = si.project_solenoidal_open(
        constant_result, add_to_result=False, background="mean")
    constant_table = si.OpenFaceMagneticTable.from_projection(
        grid, constant_projection)
    np.testing.assert_array_equal(
        constant_table.query([[0.0, 0.0, 0.0], [0.41, -0.37, 0.29]]),
        np.tile(constant, (2, 1)))
    np.testing.assert_array_equal(constant_projection["divB"], 0.0)


@needs_sph_interp
def test_open_uniform_hodge_preserves_spatial_normal_trace():
    grid = si.UniformGrid(-0.5 * np.ones(3), 0.5 * np.ones(3), (5, 6, 7))
    nx, ny, nz = grid.npx
    x = (np.arange(nx) + 0.5) / nx - 0.5
    y = (np.arange(ny) + 0.5) / ny - 0.5
    X, Y, _ = np.meshgrid(x, y, (np.arange(nz) + 0.5) / nz - 0.5,
                          indexing="ij")
    raw = {"Bx": Y, "By": -X, "Bz": 0.2 + 0.3 * X}
    trace = {
        "Bx_lo": np.broadcast_to(y[:, None], (ny, nz)),
        "Bx_hi": np.broadcast_to(y[:, None], (ny, nz)),
        "By_lo": np.broadcast_to(-x[:, None], (nx, nz)),
        "By_hi": np.broadcast_to(-x[:, None], (nx, nz)),
        "Bz_lo": np.broadcast_to((0.2 + 0.3 * x)[:, None], (nx, ny)),
        "Bz_hi": np.broadcast_to((0.2 + 0.3 * x)[:, None], (nx, ny)),
    }
    result = si.GridResult(raw, None, grid, "spatial-trace")
    projected = si.project_solenoidal_open(
        result, add_to_result=False, normal_trace=trace)

    assert np.max(np.abs(projected["divB"])) < 3e-13
    np.testing.assert_array_equal(projected["Bx_face"][0], trace["Bx_lo"])
    np.testing.assert_array_equal(projected["Bx_face"][-1], trace["Bx_hi"])
    np.testing.assert_array_equal(projected["By_face"][:, 0], trace["By_lo"])
    np.testing.assert_array_equal(projected["By_face"][:, -1], trace["By_hi"])
    np.testing.assert_array_equal(projected["Bz_face"][:, :, 0], trace["Bz_lo"])
    np.testing.assert_array_equal(projected["Bz_face"][:, :, -1], trace["Bz_hi"])
    audit = projected["_boundary_audit"]
    assert audit["trace_mode"] == "prescribed"
    assert abs(audit["net_flux_after_balance"]) < 3e-15
    assert audit["uniform_outward_correction"] == 0.0


@needs_sph_interp
def test_open_uniform_hodge_rejects_or_balances_incompatible_trace():
    grid = si.UniformGrid(-0.5 * np.ones(3), 0.5 * np.ones(3), (4, 5, 6))
    rng = np.random.default_rng(212)
    result = si.GridResult(
        {name: rng.normal(size=grid.npx) for name in ("Bx", "By", "Bz")},
        None, grid, "incompatible-trace")
    trace = {
        "Bx_lo": 0.0, "Bx_hi": 0.1,
        "By_lo": 0.0, "By_hi": 0.0,
        "Bz_lo": 0.0, "Bz_hi": 0.0,
    }
    with pytest.raises(ValueError, match="nonzero net outward flux"):
        si.project_solenoidal_open(
            result, add_to_result=False, normal_trace=trace,
            boundary_flux="error")

    with pytest.warns(RuntimeWarning, match="balanced incompatible open"):
        projected = si.project_solenoidal_open(
            result, add_to_result=False, normal_trace=trace)
    audit = projected["_boundary_audit"]
    assert audit["net_flux_before"] == pytest.approx(0.1)
    assert audit["uniform_outward_correction"] < 0.0
    assert abs(audit["net_flux_after_balance"]) < 3e-15
    assert np.max(np.abs(projected["divB"])) < 4e-13


def _open_particles(nside=4):
    g = np.linspace(-0.18, 0.18, nside)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    n = len(pos)
    mass = np.full(n, 1.0 / n)
    rho = 1.0 + 0.1 * pos[:, 0]
    h = np.full(n, 0.12)
    values = {"Bx": 0.2 + pos[:, 1],
              "By": -0.1 - pos[:, 0],
              "Bz": 0.05 + 0.5 * pos[:, 1]}
    return si.Particles(pos, mass, rho, h, values, None, False)


@needs_sph_interp
def test_open_uniform_B_over_rho_capture_is_conservative():
    particles = _open_particles()
    grid = si.UniformGrid(-0.45 * np.ones(3), 0.45 * np.ones(3), 7)
    table, audit, result = si.capture_open_face_field(particles, grid)
    assert result.method == "petkova-B-over-rho-open"
    assert audit["particle_to_raw_integral_B"]["relative_norm"] < 2e-8
    assert abs(audit["volume_relative_error"]) < 2e-8
    assert audit["boundary_flux_max"] == 0.0
    assert audit["face_divB_max"] < 2e-13
    assert np.all(np.isfinite(table.query(particles.pos)))
    B, div_audit = si.imprint_open_face_field(table, particles)
    np.testing.assert_array_equal(B, table.query(particles.pos))
    assert np.all(np.isfinite(div_audit["divB"]))


@needs_sph_interp
def test_open_uniform_mean_background_preserves_raw_carrier():
    particles = _open_particles()
    grid = si.UniformGrid(-0.45 * np.ones(3), 0.45 * np.ones(3), 7)
    table, audit, _ = si.capture_open_face_field(
        particles, grid, background="mean")
    assert audit["boundary_condition"] == "fixed-raw-mean-normal"
    assert audit["particle_to_raw_integral_B"]["relative_norm"] < 2e-8
    assert audit["raw_to_projected_integral_B"]["relative_norm"] < 2e-13
    assert audit["face_divB_max"] < 3e-13
    assert np.max(np.abs(table.cell_divergence())) < 3e-13


@needs_sph_interp
def test_open_particle_mean_background_recovers_sparse_constant_field():
    geometry = _open_particles()
    constant = np.asarray([0.4, -0.7, 0.2])
    particles = si.Particles(
        geometry.pos, geometry.mass, geometry.rho, geometry.h,
        {name: np.full(geometry.n, value)
         for name, value in zip(("Bx", "By", "Bz"), constant)},
        None, False)
    grid = si.UniformGrid(-0.45 * np.ones(3), 0.45 * np.ones(3), 7)
    table, audit, _ = si.capture_open_face_field(
        particles, grid, background="particle-mean")

    assert audit["boundary_condition"] == "fixed-particle-mean-normal"
    np.testing.assert_allclose(audit["boundary_background"], constant,
                               rtol=0.0, atol=2e-15)
    assert audit["particle_to_raw_integral_B"]["relative_norm"] < 2e-15
    assert audit["raw_to_projected_integral_B"]["relative_norm"] < 2e-14
    assert audit["face_divB_max"] < 3e-13
    np.testing.assert_allclose(table.query(particles.pos),
                               np.tile(constant, (particles.n, 1)),
                               rtol=0.0, atol=3e-15)


@needs_sph_interp
def test_open_amr_face_projection_matches_coarse_fine_fluxes():
    particles = _open_particles()

    def refine_left(count, mass, hmin, size, level, lo):
        return (level == 0) & (lo[:, 0] < 0.0)

    refine_left.needs_lo = True
    grid = si.AMRGrid.build(
        particles, base_npx=2, max_level=1, criterion=refine_left,
        lo=-0.5 * np.ones(3), hi=0.5 * np.ones(3), periodic=False,
        refine_empty=True)
    topology = si.AMRFaceTopology.build(grid)
    assert np.any((topology.minus >= 0) & (topology.plus >= 0)
                  & (grid.cell_level[np.maximum(topology.minus, 0)]
                     != grid.cell_level[np.maximum(topology.plus, 0)]))
    raw = np.random.default_rng(31).normal(size=(grid.ncell, 3))
    initial = topology.initial_values(raw)
    projected, _ = topology.project(initial)
    table = si.AMRFaceMagneticTable(topology, projected)
    assert np.max(np.abs(table.cell_divergence())) < 5e-12
    assert table.restriction_error() < 2e-15
    np.testing.assert_array_equal(projected[topology.boundary], 0.0)

    captured, audit, result = si.capture_open_amr_face_field(particles, grid)
    assert result.method == "petkova-B-over-rho-open-amr"
    assert audit["coarse_fine_patches"] > 0
    assert audit["particle_to_raw_integral_B"]["relative_norm"] < 2e-8
    assert abs(audit["volume_relative_error"]) < 2e-8
    assert audit["face_divB_max"] < 5e-12
    assert audit["restriction_flux_error_max"] < 2e-15
    assert np.max(np.abs(captured.cell_divergence())) < 5e-12
    reconstruction = captured.hdiv_reconstruction()
    assert reconstruction.virtual_divergence_max < 2e-11
    sampled = reconstruction.query(particles.pos)
    assert sampled.shape == (particles.n, 3)
    assert np.all(np.isfinite(sampled))

    # The normal trace is shared across the x=0 coarse/fine interface.
    eps = 1e-11
    across = reconstruction.query(
        [[-eps, 0.13, -0.14], [eps, 0.13, -0.14]])
    assert across[0, 0] == pytest.approx(across[1, 0], abs=2e-9)
    imprinted, imprint_audit, same_reconstruction = (
        si.imprint_open_amr_face_field(captured, particles))
    np.testing.assert_array_equal(imprinted, sampled)
    assert same_reconstruction.virtual_divergence_max < 2e-11
    assert np.all(np.isfinite(imprint_audit["divB"]))
    balsara_B, balsara_audit, balsara_reconstruction = (
        si.imprint_open_amr_face_field(
            captured, particles, reconstruction="balsara2"))
    np.testing.assert_array_equal(
        balsara_B, balsara_reconstruction.query(particles.pos))
    assert "Balsara2" in balsara_audit["reconstruction"]
    assert balsara_audit["patch_flux_error_max"] < 2e-15


@needs_sph_interp
def test_open_amr_balsara2_is_affine_exact_across_coarse_fine_faces():
    particles = _open_particles()

    def refine_left(count, mass, hmin, size, level, lo):
        return (level == 0) & (lo[:, 0] < 0.0)

    refine_left.needs_lo = True
    grid = si.AMRGrid.build(
        particles, base_npx=2, max_level=1, criterion=refine_left,
        lo=-0.5 * np.ones(3), hi=0.5 * np.ones(3), periodic=False,
        refine_empty=True)
    topology = si.AMRFaceTopology.build(grid)
    offset = np.asarray([0.4, -0.3, 0.2])
    matrix = np.asarray([[0.3, 0.7, -0.2],
                         [-0.5, -0.1, 0.4],
                         [0.6, -0.8, -0.2]])
    vectors = offset + topology.centre @ matrix.T
    values = vectors[np.arange(topology.nface), topology.axis]
    table = si.AMRFaceMagneticTable(topology, values)
    assert np.max(np.abs(table.cell_divergence())) < 2e-14

    reconstruction = table.balsara2_reconstruction(limiter="mc")
    rng = np.random.default_rng(993)
    positions = rng.uniform(grid.lo, grid.hi, size=(1000, 3))
    expected = offset + positions @ matrix.T
    np.testing.assert_allclose(reconstruction.query(positions), expected,
                               rtol=0.0, atol=2e-13)
    assert reconstruction.virtual_divergence_max < 2e-13
    assert reconstruction.patch_flux_error_max < 2e-16

    eps = 1e-11
    across = reconstruction.query(
        [[-eps, 0.13, -0.14], [eps, 0.13, -0.14]])
    assert across[0, 0] == pytest.approx(across[1, 0], abs=2e-10)
    expected_energy = 0.5 * (
        np.dot(offset, offset) + np.sum(matrix * matrix) / 12.0)
    assert reconstruction.magnetic_energy() == pytest.approx(
        expected_energy, rel=3e-13, abs=3e-14)


@needs_sph_interp
def test_open_amr_balsara2_improves_smooth_field_over_rt0():
    particles = _open_particles()

    def refine_left(count, mass, hmin, size, level, lo):
        return (level == 0) & (lo[:, 0] < 0.0)

    refine_left.needs_lo = True
    grid = si.AMRGrid.build(
        particles, base_npx=3, max_level=1, criterion=refine_left,
        lo=-0.5 * np.ones(3), hi=0.5 * np.ones(3), periodic=False,
        refine_empty=True)
    topology = si.AMRFaceTopology.build(grid)
    centre = topology.centre
    width = topology.patch_size
    normal = topology.axis
    bx = 0.3 + np.sinc(width[:, 1]) * np.sin(2.0 * np.pi * centre[:, 1])
    by = (-0.2 + 0.5 * np.sinc(width[:, 2])
          * np.cos(2.0 * np.pi * centre[:, 2]))
    bz = (0.1 + 0.4 * np.sinc(width[:, 0])
          * np.sin(2.0 * np.pi * centre[:, 0]))
    vectors = np.column_stack([bx, by, bz])
    table = si.AMRFaceMagneticTable(
        topology, vectors[np.arange(topology.nface), normal])
    assert np.max(np.abs(table.cell_divergence())) < 2e-14

    rt0 = table.hdiv_reconstruction()
    balsara2 = table.balsara2_reconstruction(limiter="mc")
    rng = np.random.default_rng(994)
    positions = rng.uniform(grid.lo, grid.hi, size=(8000, 3))
    exact = np.column_stack([
        0.3 + np.sin(2.0 * np.pi * positions[:, 1]),
        -0.2 + 0.5 * np.cos(2.0 * np.pi * positions[:, 2]),
        0.1 + 0.4 * np.sin(2.0 * np.pi * positions[:, 0]),
    ])
    rt0_error = np.sqrt(np.mean((rt0.query(positions) - exact) ** 2))
    balsara_error = np.sqrt(np.mean(
        (balsara2.query(positions) - exact) ** 2))
    assert balsara_error < 0.8 * rt0_error
    assert balsara2.virtual_divergence_max < 2e-12
    assert balsara2.patch_flux_error_max < 2e-16


@needs_sph_interp
def test_open_amr_projection_preserves_spatial_boundary_trace():
    particles = _open_particles()

    def refine_left(count, mass, hmin, size, level, lo):
        return (level == 0) & (lo[:, 0] < 0.0)

    refine_left.needs_lo = True
    grid = si.AMRGrid.build(
        particles, base_npx=2, max_level=1, criterion=refine_left,
        lo=-0.5 * np.ones(3), hi=0.5 * np.ones(3), periodic=False,
        refine_empty=True)
    topology = si.AMRFaceTopology.build(grid)
    centres = grid.centres()
    raw = np.column_stack([
        centres[:, 1], -centres[:, 0], 0.2 + 0.3 * centres[:, 0]])
    initial = topology.initial_values(raw)
    boundary_ids = np.where(topology.boundary)[0]
    for face in boundary_ids:
        axis = int(topology.axis[face])
        point = topology.centre[face]
        field = np.asarray([point[1], -point[0], 0.2 + 0.3 * point[0]])
        initial[face] = field[axis]

    projected, _phi, audit = topology.project(
        initial, return_audit=True)
    np.testing.assert_array_equal(
        projected[topology.boundary], initial[topology.boundary])
    assert abs(audit["net_flux_after_balance"]) < 3e-15
    assert np.max(np.abs(topology.cell_divergence(projected))) < 8e-12

    incompatible = initial.copy()
    high_x = topology.boundary & (topology.plus < 0) & (topology.axis == 0)
    incompatible[high_x] += 0.1
    with pytest.raises(ValueError, match="nonzero net outward flux"):
        topology.project(incompatible, boundary_flux="error")
    with pytest.warns(RuntimeWarning, match="balanced incompatible AMR"):
        balanced, _phi, balance_audit = topology.project(
            incompatible, return_audit=True)
    assert balance_audit["uniform_outward_correction"] < 0.0
    assert abs(balance_audit["net_flux_after_balance"]) < 3e-15
    assert np.max(np.abs(topology.cell_divergence(balanced))) < 8e-12

    def boundary_field(points):
        return np.column_stack([
            0.2 + points[:, 1],
            -0.1 - points[:, 0],
            0.05 + 0.5 * points[:, 1],
        ])

    captured, capture_audit, _ = si.capture_open_amr_face_field(
        particles, grid, normal_trace=boundary_field)
    boundary_ids = np.where(captured.topology.boundary)[0]
    expected_vectors = boundary_field(captured.topology.centre[boundary_ids])
    expected_normal = expected_vectors[
        np.arange(len(boundary_ids)), captured.topology.axis[boundary_ids]]
    np.testing.assert_array_equal(captured.values[boundary_ids], expected_normal)
    assert capture_audit["boundary_condition"] == "fixed-spatial-normal"
    assert np.max(np.abs(captured.cell_divergence())) < 8e-12


@needs_sph_interp
def test_validation_compares_reference_on_identical_target_particles():
    nside = 5
    spacing = 1.0 / nside
    axis = (np.arange(nside) + 0.5) * spacing - 0.5
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(44)
    pos += rng.uniform(-0.12 * spacing, 0.12 * spacing, pos.shape)
    n = len(pos)
    particles = si.Particles(
        pos, np.full(n, 1.0 / n), np.ones(n), np.full(n, 0.18), {},
        np.ones(3), True)

    def reference(points):
        points = np.asarray(points)
        return np.column_stack([
            0.3 + np.sin(2.0 * np.pi * points[:, 1]),
            -0.2 + 0.5 * np.cos(2.0 * np.pi * points[:, 2]),
            0.1 + 0.4 * np.sin(2.0 * np.pi * points[:, 0]),
        ])

    # Build an exact shared-face representation of a constant field.  The
    # non-constant analytic arm then guarantees nonzero, finite comparison
    # metrics without making this unit test depend on a Petkova resolution.
    shape = (4, 4, 4)
    table = si.FaceMagneticTable(
        -0.5 * np.ones(3), np.ones(3), shape,
        np.full(shape, 0.3), np.full(shape, -0.2), np.full(shape, 0.1))
    report, reference_B, reconstructed_B = (
        si.validate_magnetic_reconstruction(
            table, particles, reference, flux_npx=8, spectrum_npx=8,
            return_fields=True))

    np.testing.assert_array_equal(reference_B, reference(pos))
    np.testing.assert_array_equal(reconstructed_B, table.query(pos))
    assert report["n_particles"] == n
    assert report["point_error"]["relative_l2"] > 0.0
    assert report["sph_divergence"]["reference"]["normalized_mean"] > 0.0
    assert report["sph_divergence"]["reconstructed"]["normalized_mean"] == 0.0
    assert report["magnetic_energy"]["face_table"] > 0.0
    assert report["volume_integral_B"]["relative_norm"] > 0.0
    assert len(report["resolved_fluxes"]["plane_fractions"]) == 4
    assert report["spectra"]["sample_npx"] == 8


@needs_sph_interp
def test_validation_accepts_particle_only_reference_array():
    particles = _open_particles(nside=3)
    grid = si.UniformGrid(-0.45 * np.ones(3), 0.45 * np.ones(3), 5)
    table, _, _ = si.capture_open_face_field(particles, grid)
    reference = np.column_stack([
        particles.values["Bx"], particles.values["By"],
        particles.values["Bz"]])
    report = si.validate_magnetic_reconstruction(
        table, particles, reference, flux_npx=0, spectrum_npx=0)
    assert "resolved_fluxes" not in report
    assert "spectra" not in report
    assert np.isfinite(report["point_error"]["relative_l2"])
    assert np.isfinite(report["volume_integral_B"]["relative_norm"])


@needs_sph_interp
def test_rt0_reference_error_decreases_with_grid_resolution():
    nside = 7
    spacing = 1.0 / nside
    axis = (np.arange(nside) + 0.5) * spacing - 0.5
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    pos += np.random.default_rng(91).uniform(
        -0.15 * spacing, 0.15 * spacing, pos.shape)
    n = len(pos)
    particles = si.Particles(
        pos, np.full(n, 1.0 / n), np.ones(n), np.full(n, 0.14), {},
        np.ones(3), True)

    def reference(points):
        points = np.asarray(points)
        return np.column_stack([
            0.3 + np.sin(2.0 * np.pi * points[:, 1]),
            -0.2 + 0.5 * np.cos(2.0 * np.pi * points[:, 2]),
            0.1 + 0.4 * np.sin(2.0 * np.pi * points[:, 0]),
        ])

    def exact_face_table(npx):
        centres = (np.arange(npx) + 0.5) / npx - 0.5
        ix, iy, iz = np.meshgrid(centres, centres, centres, indexing="ij")
        bx = 0.3 + np.sin(2.0 * np.pi * iy)
        by = -0.2 + 0.5 * np.cos(2.0 * np.pi * iz)
        bz = 0.1 + 0.4 * np.sin(2.0 * np.pi * ix)
        return si.FaceMagneticTable(
            -0.5 * np.ones(3), np.ones(3), (npx,) * 3, bx, by, bz)

    coarse = si.validate_magnetic_reconstruction(
        exact_face_table(4), particles, reference,
        flux_npx=0, spectrum_npx=0)
    fine = si.validate_magnetic_reconstruction(
        exact_face_table(8), particles, reference,
        flux_npx=0, spectrum_npx=0)
    assert fine["point_error"]["relative_l2"] \
        < 0.65 * coarse["point_error"]["relative_l2"]
    # SPH divergence is a separate particle/operator compatibility diagnostic,
    # not the convergence norm for the continuous field: on this fixed target
    # geometry it is nearly flat (and need not be monotone) while the B error
    # converges.  Preserve that distinction in the regression contract.
    assert np.isfinite(
        coarse["sph_divergence"]["reconstructed"]["normalized_mean"])
    assert np.isfinite(
        fine["sph_divergence"]["reconstructed"]["normalized_mean"])


@needs_sph_interp
def test_validation_on_independently_relaxed_final_glass():
    from glassgen import relax

    n = 5 ** 3
    mass = np.full(n, 1.0 / n)
    initial = np.random.default_rng(120).uniform(-0.5, 0.5, size=(n, 3))
    relaxed = relax(
        initial, mass, lambda points: np.ones(len(points)),
        box=np.ones(3), nsmooth=32, max_iter=1200, icr0_stop=3e-3)
    assert relaxed.converged
    particles = si.Particles(
        relaxed.pos, mass, relaxed.rho, relaxed.h, {}, np.ones(3), True)

    def reference(points):
        points = np.asarray(points)
        return np.column_stack([
            0.3 + np.sin(2.0 * np.pi * points[:, 1]),
            -0.2 + 0.5 * np.cos(2.0 * np.pi * points[:, 2]),
            0.1 + 0.4 * np.sin(2.0 * np.pi * points[:, 0]),
        ])

    shape = (6, 6, 6)
    centres = (np.arange(6) + 0.5) / 6.0 - 0.5
    ix, iy, iz = np.meshgrid(centres, centres, centres, indexing="ij")
    table = si.FaceMagneticTable(
        -0.5 * np.ones(3), np.ones(3), shape,
        0.3 + np.sin(2.0 * np.pi * iy),
        -0.2 + 0.5 * np.cos(2.0 * np.pi * iz),
        0.1 + 0.4 * np.sin(2.0 * np.pi * ix))
    report = si.validate_magnetic_reconstruction(
        table, particles, reference, flux_npx=0, spectrum_npx=0)
    assert report["n_particles"] == n
    assert np.isfinite(report["point_error"]["relative_l2"])
    assert np.isfinite(
        report["sph_divergence"]["reference"]["normalized_mean"])
    assert np.isfinite(report["volume_integral_B"]["relative_norm"])
