"""Make an interpolated cell-centred B divergence-free for constrained transport.

`project_solenoidal` performs the MAC (staggered Marker-and-Cell) Hodge
projection on a UniformGrid result: it averages the cell-centred B onto the cell
faces, removes the divergent part by solving a discrete Poisson equation, and
returns the six RAMSES-style FACE-centred components (Bx_l/Bx_r/By_l/By_r/
Bz_l/Bz_r). These satisfy the *discrete* constrained-transport divergence

    (Bx_r - Bx_l)/dx + (By_r - By_l)/dy + (Bz_r - Bz_l)/dz = 0

to machine precision per cell — guaranteed algebraically (not by interpolation
quality) because the same staggered grad/div stencils compose into the Poisson
operator used in the solve: div(B - grad phi) = div B - L phi = 0 identically.

Periodic only (FFT). The projection changes the field by exactly its divergent
(monopole) component — typically ~1% for a reasonably div-cleaned SPMHD snapshot,
i.e. it removes the unphysical part and leaves the physical field essentially
intact. See the constrained-transport note in export.py.
"""

import numpy as np

from .targets import UniformGrid


def project_solenoidal(result, fields=("Bx", "By", "Bz"), periodic=True,
                       add_to_result=True):
    """Project a cell-centred B (on a UniformGrid result) to a div-free face B.

    result : a `GridResult` whose target is a `UniformGrid`, carrying the three
             cell-centred magnetic components named by `fields`.
    fields : the (Bx, By, Bz) field names in `result.data`.
    periodic : FFT projection assumes a periodic box (only mode implemented).
    add_to_result : also insert the outputs into `result.data` (so a subsequent
             `to_ramses(..., hydro_fields=RAMSES_MHD_VARS)` finds them).

    Returns dict with the six face components `Bx_l,Bx_r,By_l,By_r,Bz_l,Bz_r`
    (each (nx,ny,nz)); the cleaned cell-centred field `Bx_c,By_c,Bz_c` (face
    averages, for plotting/analysis); and `divB` (residual cell divergence of the
    cleaned face field — ~1e-14).
    """
    grid = getattr(result, "target", None)
    if not isinstance(grid, UniformGrid):
        raise TypeError("project_solenoidal needs a UniformGrid GridResult "
                        f"(got {type(grid).__name__})")
    if not periodic:
        raise NotImplementedError(
            "only the periodic FFT projection is implemented; a non-periodic box "
            "needs Neumann BCs (DCT/DST) or a multigrid solve.")

    bx = np.ascontiguousarray(result.data[fields[0]], dtype=np.float64)
    by = np.ascontiguousarray(result.data[fields[1]], dtype=np.float64)
    bz = np.ascontiguousarray(result.data[fields[2]], dtype=np.float64)
    nx, ny, nz = grid.npx
    dx, dy, dz = grid.pixwidth

    # cell-centred -> face-centred (MAC): face i+1/2 = mean of cells i, i+1
    Bxf = 0.5 * (bx + np.roll(bx, -1, axis=0))
    Byf = 0.5 * (by + np.roll(by, -1, axis=1))
    Bzf = 0.5 * (bz + np.roll(bz, -1, axis=2))

    def celldiv(ax, ay, az):
        return ((ax - np.roll(ax, 1, axis=0)) / dx
                + (ay - np.roll(ay, 1, axis=1)) / dy
                + (az - np.roll(az, 1, axis=2)) / dz)

    D = celldiv(Bxf, Byf, Bzf)

    # discrete Poisson  L phi = D  via FFT, with the 7-point-stencil symbol
    # lambda = -sum (2/dp)^2 sin^2(pi m / n)  (NOT -k^2: must match the stencil)
    sx = (2.0 / dx) * np.sin(np.pi * (np.fft.fftfreq(nx) * nx) / nx)
    sy = (2.0 / dy) * np.sin(np.pi * (np.fft.fftfreq(ny) * ny) / ny)
    sz = (2.0 / dz) * np.sin(np.pi * (np.fft.fftfreq(nz) * nz) / nz)
    lam = -(sx[:, None, None]**2 + sy[None, :, None]**2 + sz[None, None, :]**2)
    lam[0, 0, 0] = 1.0                       # zero-mode of phi := 0
    phi = np.fft.ifftn(np.fft.fftn(D) / lam).real

    # correct the faces: subtract the discrete face-gradient of phi
    Bxf = Bxf - (np.roll(phi, -1, axis=0) - phi) / dx
    Byf = Byf - (np.roll(phi, -1, axis=1) - phi) / dy
    Bzf = Bzf - (np.roll(phi, -1, axis=2) - phi) / dz

    out = {
        # RAMSES face components: right face = Bxf[i]; left face = Bxf[i-1]
        "Bx_l": np.roll(Bxf, 1, axis=0), "Bx_r": Bxf,
        "By_l": np.roll(Byf, 1, axis=1), "By_r": Byf,
        "Bz_l": np.roll(Bzf, 1, axis=2), "Bz_r": Bzf,
        # cleaned cell-centred field (for analysis/plots)
        "Bx_c": 0.5 * (Bxf + np.roll(Bxf, 1, axis=0)),
        "By_c": 0.5 * (Byf + np.roll(Byf, 1, axis=1)),
        "Bz_c": 0.5 * (Bzf + np.roll(Bzf, 1, axis=2)),
        "divB": celldiv(Bxf, Byf, Bzf),
    }
    if add_to_result:
        result.data.update(out)
    return out


def project_solenoidal_open(result, fields=("Bx", "By", "Bz"),
                            add_to_result=True, background=None,
                            normal_trace=None, boundary_flux="balance"):
    """Project a UniformGrid field while preserving its outer normal trace.

    This is the open-domain counterpart of :func:`project_solenoidal`.  The
    projection subtracts a uniform background, fixes the residual normal field
    to zero on all outer faces, projects the residual interior faces, and adds
    the background back.  Interior faces are Hodge-projected by a cell-centred
    Neumann Poisson solve diagonalized with a DCT-II.

    ``background=None`` retains the compact-field, fixed-zero-normal boundary
    condition.  ``background="mean"`` uses the geometric cell-volume mean of
    the raw field and therefore preserves each component's grid volume
    integral.  An explicit three-vector prescribes a known uniform far-field;
    this is the appropriate mode for a box-threading constant field.

    ``normal_trace`` is the general boundary model.  It may be ``"nearest"``
    (nearest-cell extrapolation) or a mapping with ``Bx_lo``, ``Bx_hi``,
    ``By_lo``, ``By_hi``, ``Bz_lo``, and ``Bz_hi`` arrays.  These are Cartesian
    component values, not outward-signed fluxes.  Their shapes are the
    corresponding transverse face shapes.  A divergence-free interior exists
    only when the net outward boundary flux is zero.  ``boundary_flux="balance"``
    applies, reports, and warns about the minimum uniform outward-normal
    correction before projection; ``"error"`` checks compatibility strictly.

    Face arrays named ``Bx_face``/``By_face``/``Bz_face`` include both domain
    boundaries and consequently have shapes ``(nx+1,ny,nz)``,
    ``(nx,ny+1,nz)``, and ``(nx,ny,nz+1)``.  RAMSES-style per-cell left/right
    views are returned as well.
    """
    from scipy.fft import dctn, idctn

    grid = getattr(result, "target", None)
    if not isinstance(grid, UniformGrid):
        raise TypeError("project_solenoidal_open needs a UniformGrid GridResult "
                        f"(got {type(grid).__name__})")
    bx = np.ascontiguousarray(result.data[fields[0]], dtype=np.float64)
    by = np.ascontiguousarray(result.data[fields[1]], dtype=np.float64)
    bz = np.ascontiguousarray(result.data[fields[2]], dtype=np.float64)
    shape = tuple(grid.npx)
    for name, value in zip(fields, (bx, by, bz)):
        if value.shape != shape:
            raise ValueError(f"field {name!r} has shape {value.shape}, expected {shape}")
    nx, ny, nz = shape
    dx, dy, dz = grid.pixwidth

    if normal_trace is not None and background is not None:
        raise ValueError("specify either background or normal_trace, not both")
    if boundary_flux not in ("error", "balance"):
        raise ValueError("boundary_flux must be 'error' or 'balance'")

    if background is None:
        background_vector = np.zeros(3, dtype=np.float64)
    elif isinstance(background, str):
        if background != "mean":
            raise ValueError("open projection background string must be 'mean'")
        background_vector = np.asarray(
            [np.mean(bx), np.mean(by), np.mean(bz)], dtype=np.float64)
    else:
        background_vector = np.asarray(background, dtype=np.float64)
        if background_vector.shape != (3,):
            raise ValueError("open projection background must be a three-vector")
        if not np.all(np.isfinite(background_vector)):
            raise ValueError("open projection background must be finite")

    if normal_trace is None:
        bx = bx - background_vector[0]
        by = by - background_vector[1]
        bz = bz - background_vector[2]

    Bxf = np.zeros((nx + 1, ny, nz), dtype=np.float64)
    Byf = np.zeros((nx, ny + 1, nz), dtype=np.float64)
    Bzf = np.zeros((nx, ny, nz + 1), dtype=np.float64)
    Bxf[1:nx] = 0.5 * (bx[:-1] + bx[1:])
    Byf[:, 1:ny] = 0.5 * (by[:, :-1] + by[:, 1:])
    Bzf[:, :, 1:nz] = 0.5 * (bz[:, :, :-1] + bz[:, :, 1:])

    trace_mode = "uniform-background"
    if normal_trace is not None:
        trace_mode = "nearest-cell" if isinstance(normal_trace, str) else "prescribed"
        if isinstance(normal_trace, str):
            if normal_trace != "nearest":
                raise ValueError("normal_trace string must be 'nearest'")
            trace = {
                "Bx_lo": bx[0], "Bx_hi": bx[-1],
                "By_lo": by[:, 0], "By_hi": by[:, -1],
                "Bz_lo": bz[:, :, 0], "Bz_hi": bz[:, :, -1],
            }
        else:
            required = ("Bx_lo", "Bx_hi", "By_lo", "By_hi",
                        "Bz_lo", "Bz_hi")
            missing = [key for key in required if key not in normal_trace]
            if missing:
                raise ValueError(
                    "normal_trace is missing " + ", ".join(missing))
            trace = normal_trace

        trace_shapes = {
            "Bx_lo": (ny, nz), "Bx_hi": (ny, nz),
            "By_lo": (nx, nz), "By_hi": (nx, nz),
            "Bz_lo": (nx, ny), "Bz_hi": (nx, ny),
        }
        resolved_trace = {}
        for key, trace_shape in trace_shapes.items():
            value = np.asarray(trace[key], dtype=np.float64)
            try:
                value = np.array(np.broadcast_to(value, trace_shape), copy=True)
            except ValueError as exc:
                raise ValueError(
                    f"normal_trace {key} has shape {value.shape}, "
                    f"expected {trace_shape}") from exc
            if not np.all(np.isfinite(value)):
                raise ValueError(f"normal_trace {key} contains non-finite values")
            resolved_trace[key] = value
        Bxf[0], Bxf[-1] = resolved_trace["Bx_lo"], resolved_trace["Bx_hi"]
        Byf[:, 0], Byf[:, -1] = resolved_trace["By_lo"], resolved_trace["By_hi"]
        Bzf[:, :, 0], Bzf[:, :, -1] = (
            resolved_trace["Bz_lo"], resolved_trace["Bz_hi"])

    def boundary_net_flux(ax, ay, az):
        return float(
            (np.sum(ax[-1]) - np.sum(ax[0])) * dy * dz
            + (np.sum(ay[:, -1]) - np.sum(ay[:, 0])) * dx * dz
            + (np.sum(az[:, :, -1]) - np.sum(az[:, :, 0])) * dx * dy)

    def boundary_flux_scale(ax, ay, az):
        return float(
            (np.sum(np.abs(ax[-1])) + np.sum(np.abs(ax[0]))) * dy * dz
            + (np.sum(np.abs(ay[:, -1])) + np.sum(np.abs(ay[:, 0]))) * dx * dz
            + (np.sum(np.abs(az[:, :, -1]))
               + np.sum(np.abs(az[:, :, 0]))) * dx * dy)

    flux_before = boundary_net_flux(Bxf, Byf, Bzf)
    flux_tolerance = (2048.0 * np.finfo(np.float64).eps
                      * max(boundary_flux_scale(Bxf, Byf, Bzf),
                            np.finfo(np.float64).tiny))
    outward_correction = 0.0
    if abs(flux_before) > flux_tolerance:
        if boundary_flux == "error":
            raise ValueError(
                "normal boundary trace has nonzero net outward flux "
                f"{flux_before:.16e} (tolerance {flux_tolerance:.3e})")
        total_boundary_area = 2.0 * (
            ny * nz * dy * dz + nx * nz * dx * dz + nx * ny * dx * dy)
        outward_correction = -flux_before / total_boundary_area
        import warnings
        warnings.warn(
            "balanced incompatible open magnetic boundary trace: "
            f"net outward flux {flux_before:.16e}, uniform outward-normal "
            f"correction {outward_correction:.16e}",
            RuntimeWarning, stacklevel=2)
        Bxf[0] -= outward_correction
        Bxf[-1] += outward_correction
        Byf[:, 0] -= outward_correction
        Byf[:, -1] += outward_correction
        Bzf[:, :, 0] -= outward_correction
        Bzf[:, :, -1] += outward_correction
    flux_after_balance = boundary_net_flux(Bxf, Byf, Bzf)

    def celldiv(ax, ay, az):
        return ((ax[1:] - ax[:-1]) / dx
                + (ay[:, 1:] - ay[:, :-1]) / dy
                + (az[:, :, 1:] - az[:, :, :-1]) / dz)

    D = celldiv(Bxf, Byf, Bzf)
    kx = np.arange(nx, dtype=np.float64)
    ky = np.arange(ny, dtype=np.float64)
    kz = np.arange(nz, dtype=np.float64)
    lam = (-4.0 * np.sin(0.5 * np.pi * kx / nx)[:, None, None] ** 2 / dx ** 2
           -4.0 * np.sin(0.5 * np.pi * ky / ny)[None, :, None] ** 2 / dy ** 2
           -4.0 * np.sin(0.5 * np.pi * kz / nz)[None, None, :] ** 2 / dz ** 2)
    Dhat = dctn(D, type=2, norm="ortho")
    lam[0, 0, 0] = 1.0
    Dhat[0, 0, 0] = 0.0
    phi = idctn(Dhat / lam, type=2, norm="ortho")

    Bxf[1:nx] -= (phi[1:] - phi[:-1]) / dx
    Byf[:, 1:ny] -= (phi[:, 1:] - phi[:, :-1]) / dy
    Bzf[:, :, 1:nz] -= (phi[:, :, 1:] - phi[:, :, :-1]) / dz
    if normal_trace is None:
        Bxf += background_vector[0]
        Byf += background_vector[1]
        Bzf += background_vector[2]
    div = celldiv(Bxf, Byf, Bzf)
    out = {
        "Bx_face": Bxf, "By_face": Byf, "Bz_face": Bzf,
        "Bx_l": Bxf[:-1], "Bx_r": Bxf[1:],
        "By_l": Byf[:, :-1], "By_r": Byf[:, 1:],
        "Bz_l": Bzf[:, :, :-1], "Bz_r": Bzf[:, :, 1:],
        "Bx_c": 0.5 * (Bxf[:-1] + Bxf[1:]),
        "By_c": 0.5 * (Byf[:, :-1] + Byf[:, 1:]),
        "Bz_c": 0.5 * (Bzf[:, :, :-1] + Bzf[:, :, 1:]),
        "divB": div,
        "_boundary_audit": {
            "trace_mode": trace_mode,
            "flux_policy": boundary_flux,
            "net_flux_before": flux_before,
            "net_flux_after_balance": flux_after_balance,
            "uniform_outward_correction": outward_correction,
            "compatibility_tolerance": flux_tolerance,
        },
    }
    if add_to_result:
        result.data.update({key: value for key, value in out.items()
                            if not key.startswith("_")})
    return out
