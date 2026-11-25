#!/usr/bin/env python3
"""Reproducible Petkova ensembles for the published-forcing MHD campaign.

Caches contain measured spectra, fixed-bin PDFs, and 2D maps, together with
snapshot/analysis provenance. Existing artifacts are never overwritten.
This is the user's fixed-alpha Gasoline comparator, not an identical PHANTOM
numerical method or a claim of resolution convergence.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np

import IC_analysis_turbbox as analysis
from IC_analysis_general import read_log_series


FORCING_MD5 = "1453a92ea11a43ea7a291b6a1cd886f7"
SCHEMA = 1
RHO_EDGES = np.linspace(-16.0, 16.0, 321)  # Delta ln(rho/rho0) = 0.1
B2_EDGES = np.linspace(-20.0, 6.0, 261)   # absolute log10(B^2), mu0=1
EXPECTED = {
    "bPeriodic": 1, "dxPeriod": 1, "dyPeriod": 1, "dzPeriod": 1,
    "bDoGravity": 0, "bGasIsothermal": 1, "bGasCooling": 0,
    "nSmooth": 64, "bVariableAlpha": 0, "dConstAlpha": 0.5,
    "dConstBeta": 4, "bDivBClean": 1, "dMHDfch": 1,
    "dMHDPsiDecFac": 1, "dMHDAlphaB": 0, "dMagDiffLin": 0,
    "dMagDiffTurb": 0, "dMagDiffShock": 0, "dMagDiffTricco": 1,
    "dMagDiffRecon": 0, "dMagDiffAlphaBMax": 1,
    "dMagDiffTriccoEps": 0, "bMagDiffSigFast": 1,
    "dThermalDiffusionCoeff": 0, "dDelta": 0.0005,
    "StDecay": 0.05, "StEnergy": 8, "StStartTime": 0,
    "StDtFreq": 0.005, "StKmin": 6.28, "StKmax": 18.90,
    "StSolWeight": 1, "StAmplFac": 1, "StSpectForm": 1,
    "StCorrectBulkMotion": 1,
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(analysis._json_safe(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def numeric_contract(header, expected=EXPECTED):
    """Check every realised occurrence, including contradictory restart headers."""
    measured, errors = {}, []
    for key, target in expected.items():
        tokens = re.findall(rf"\b{re.escape(key)}:\s*([^\s,;]+)", header)
        values = []
        for token in tokens:
            try:
                values.append(float(token))
            except ValueError:
                values.append(float("nan"))
        measured[key] = values
        if not values or not all(np.isfinite(v) and np.isclose(
                v, target, rtol=2e-6, atol=1e-12) for v in values):
            errors.append(f"{key}: expected {target}, found {values}")
    return measured, errors


def audit_contract(run, crossings=100):
    """Fail before analysis if the realised fixed-alpha campaign has drifted."""
    run = Path(run).resolve()
    logs = list(run.glob("*.log"))
    if len(logs) != 1:
        raise ValueError(f"expected one unambiguous run log: {run}")
    header = "\n".join(line for line in logs[0].read_text().splitlines()
                       if line.startswith("#"))
    measured, errors = numeric_contract(header)
    manifest = json.loads((run / "gasoline-build-manifest.json").read_text())
    launch = json.loads((run / "launch.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    defines = set(manifest["effective_defines"])
    for flag in ("-DOUTURBDRIVER", "-DMHD", "-DRECONSTRUCTION", "-DRECONB",
                 "-DKERNELOPTIMIZED", "-DRECON_DIVGATE_K=1.0",
                 "-DRECON_DIVGATE_RAMP"):
        if flag not in defines:
            errors.append(f"missing build flag {flag}")
    if any(flag.startswith("-DVARALPHA") for flag in defines):
        errors.append("VARALPHA is incompatible with the accepted contract")
    forcing = Path(launch["forcing_file"])
    forcing_hash = digest(forcing, "md5")
    if forcing_hash != FORCING_MD5 or launch["forcing_md5"] != FORCING_MD5:
        errors.append("published forcing checksum differs")
    sources = re.findall(r"# OUturb source: (.*?) ;", header)
    if not sources or any(Path(s).resolve() != forcing.resolve() for s in sources):
        errors.append("realised forcing source differs from launch record")
    binary_hash = digest(launch["binary"])
    if binary_hash != manifest["binary"]["sha256"]:
        errors.append("binary checksum differs from recorded build")
    log = read_log_series(str(run), ("dTime", "Ekin", "Emag"))
    steps = round(crossings * 100)
    if (completion.get("exit_code") != 0 or log is None
            or len(log["dTime"]) != steps + 1
            or not np.isclose(log["dTime"][-1], crossings * .05)
            or launch.get("steps") != steps):
        errors.append(f"incomplete {crossings:g}-crossing campaign")
    if log is not None and any(np.any(~np.isfinite(v)) for v in log.values()):
        errors.append("non-finite log energy/time")
    if errors:
        raise ValueError("realised campaign contract failed:\n" + "\n".join(errors))
    return {
        "status": "PASS", "run": str(run), "parameters": measured,
        "build_key": manifest["build_key"], "binary_sha256": binary_hash,
        "forcing_md5": forcing_hash, "log_sha256": digest(logs[0]),
        "elapsed_seconds": completion["elapsed_seconds"],
        "crossings": crossings,
        "effective_defines": sorted(defines),
        "simulation_kernel": "KERNELOPTIMIZED(O_64), not cubic spline",
        "analysis_kernel": "existing Petkova cubic-spline kernel-volume integral",
        "paper_equivalence": False,
    }


def select_samples(rows, start, end, count):
    """Nearest stored snapshots to uniform times, rejecting missing coverage."""
    if count < 2 or not (np.isfinite(start) and np.isfinite(end) and end > start):
        raise ValueError("ensemble requires count>=2 and finite start<end")
    tau = np.array([r["time"] / 0.05 for r in rows])
    if len(tau) < 2 or np.any(np.diff(tau) <= 0):
        raise ValueError("snapshots must have unique increasing times")
    tolerance = 1e-8
    if start < tau[0] - tolerance or end > tau[-1] + tolerance:
        raise ValueError("requested window is outside stored snapshot coverage")
    targets = np.linspace(start, end, count)
    indices = np.array([np.argmin(abs(tau - target)) for target in targets])
    if len(np.unique(indices)) != count:
        raise ValueError("not enough distinct snapshots for requested sample count")
    return [rows[i] for i in indices]


def fixed_pdf(values, edges, weights=None):
    """PDF per fixed coordinate bin; tails retain their missing probability."""
    x = np.asarray(values).ravel()
    w = np.ones(x.size) if weights is None else np.asarray(weights).ravel()
    if w.shape != x.shape or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("PDF weights must be matching, finite and non-negative")
    total = float(w.sum())
    if total <= 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("PDF needs positive weight and increasing bins")
    histogram = np.histogram(x, bins=edges, weights=w)[0]
    return histogram / total / np.diff(edges), float(1 - histogram.sum() / total)


def aggregate(records, key):
    values = np.stack([r[key] for r in records])
    if len(values) < 2 or np.any(~np.isfinite(values)):
        raise ValueError("ensemble needs at least two finite measurements")
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def grid_diagnostics(data, row):
    """Derive all products from the same Petkova grid and particle bulk frame."""
    rho = np.asarray(data["rho"], dtype=float)
    v = np.stack([data[k] for k in ("vx", "vy", "vz")])
    v -= np.array([row[k] for k in ("bulk_vx", "bulk_vy", "bulk_vz")])[:, None, None, None]
    b = np.stack([data[k] for k in ("Bx", "By", "Bz")])
    if np.any(~np.isfinite(rho)) or np.any(rho < 0) or np.any(~np.isfinite(v)) or np.any(~np.isfinite(b)):
        raise ValueError("non-finite grid or negative density")
    v2, b2 = np.sum(v*v, axis=0), np.sum(b*b, axis=0)
    products = {"time_crossings": row["time"] / 0.05,
                "grid_resolution": rho.shape[0], "npart": row["npart"]}
    for name, field in (("v", v), ("rho13v", rho[None]**(1/3)*v),
                        ("kinetic", rho[None]**0.5*v), ("magnetic", b),
                        ("density", np.stack([rho, np.zeros_like(rho), np.zeros_like(rho)]))):
        spectrum = analysis.isotropic_vector_spectrum(field)
        products["k"] = spectrum["k"]
        for component in ("total", "solenoidal", "compressive"):
            products[name + "_" + component] = spectrum[component]
        products[name + "_full_grid_power"] = 0.5 * np.mean(np.sum(field*field, axis=0))
        products[name + "_zero_mode_power"] = 0.5 * np.sum(field.mean(axis=(1,2,3))**2)
        products[name + "_outside_sphere_power"] = (
            products[name + "_full_grid_power"] - products[name + "_zero_mode_power"]
            - np.sum(spectrum["total"]))
    s = np.full(rho.shape, -np.inf)
    np.log(rho / row["mass"], out=s, where=rho > 0)
    logb2 = np.full(b2.shape, -np.inf)
    np.log10(b2, out=logb2, where=b2 > 0)
    products["rho_pdf"], products["rho_pdf_outside"] = fixed_pdf(s, RHO_EDGES)
    products["rho_mass_pdf"], products["rho_mass_pdf_outside"] = fixed_pdf(s, RHO_EDGES, rho)
    products["b2_pdf"], products["b2_pdf_outside"] = fixed_pdf(logb2, B2_EDGES)
    products["rho_edges"], products["b2_edges"] = RHO_EDGES, B2_EDGES
    products["grid_mass_relative_error"] = float(np.sum(data["_mass"]) / row["mass"] - 1)
    products["grid_kinetic_over_particle"] = (0.5 * np.mean(rho*v2) / row["ekin"])
    products["grid_magnetic_over_particle"] = (0.5 * np.mean(b2) / row["emag"])
    products["volume_compressive_fraction"] = (
        np.sum(products["v_compressive"]) / np.sum(products["v_total"]))
    mid = rho.shape[2] // 2
    # The central z=0 plane is the mean of the two straddling cell averages.
    products["rho_slice"] = 0.5*(rho[:,:,mid-1] + rho[:,:,mid])
    bmag = np.sqrt(b2)
    products["b_slice"] = 0.5*(bmag[:,:,mid-1] + bmag[:,:,mid])
    products["column_density"] = np.mean(rho, axis=2)
    products["column_bmag"] = np.mean(bmag, axis=2)
    return products


def numerical_signature(source, names):
    """Canonical AST for the numerical definitions, excluding CLI/report code.

    Used only with a source archive whose whole-file hashes match the cache.
    This permits a reviewed presentation change without discarding expensive
    gridding measurements; numerical changes still reject the old cache.
    """
    nodes = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node
    selected = {name: ast.dump(nodes[name], include_attributes=False) for name in names}
    referenced = {node.id for name in names for node in ast.walk(nodes[name])
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    # Import rebinding can change the numerical algorithm without changing
    # its function body. Include imports for every referenced global name.
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                binding = alias.asname or (alias.name.split('.')[0]
                          if isinstance(node, ast.Import) else alias.name)
                if binding in referenced:
                    selected['import:'+binding] = ast.dump(node, include_attributes=False)
    return selected


def compatible_cache_identity(found, expected, source_archive=None):
    if found == expected:
        return True
    if source_archive is None:
        return False
    archived = Path(source_archive)
    checked = dict(found)
    definitions = (
        ("module_sha256", Path(__file__), ("grid_diagnostics", "fixed_pdf", "RHO_EDGES", "B2_EDGES")),
        ("spectrum_module_sha256", Path(analysis.__file__),
         ("petkova_grid", "infer_grid_resolution", "isotropic_vector_spectrum", "_rfft_geometry")),
    )
    for key, current, names in definitions:
        old = archived / current.name
        if not old.is_file() or digest(old) != found.get(key):
            return False
        if numerical_signature(old.read_text(), names) != numerical_signature(current.read_text(), names):
            return False
        checked[key] = expected[key]
    return checked == expected


def cached_grid(row, ngrid, cache, reuse_grid=None, source_archive=None):
    """Exclusive-write cache with source fingerprints and analysis identity."""
    filename = Path(row["filename"]).resolve()
    native = round(row["npart"]**(1/3))
    key = f"n{native}_step{filename.suffix[1:]}_grid{ngrid}"
    path = cache / (key + ".npz")
    sources = [filename] + [Path(str(filename) + suffix) for suffix in
                           (".BFieldx", ".BFieldy", ".BFieldz")]
    smooth = Path(str(filename) + ".smoothlength")
    if smooth.exists():
        sources.append(smooth)
    identity = {
        "schema": SCHEMA, "sources": {str(p): digest(p) for p in sources},
        "grid": ngrid, "method": "petkova", "nsmooth": 64,
        "module_sha256": digest(__file__),
        "spectrum_module_sha256": digest(analysis.__file__),
        "kernel_module_sha256": digest(Path(__file__).parent / "sph_interp/petkova_method.py"),
    }
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            found = json.loads(str(saved["identity"]))
            if not compatible_cache_identity(found, identity, source_archive):
                raise ValueError(f"stale cache: choose a new cache directory: {path}")
            return {k: saved[k] for k in saved.files if k != "identity"}
    tic = time.monotonic()
    if reuse_grid is not None:
        with np.load(reuse_grid) as saved:
            data = {k: saved[k] for k in saved.files}
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            grid, _particles, _error = analysis.petkova_grid(str(filename), True, ngrid)
        data = grid.data
    result = grid_diagnostics(data, row)
    result["elapsed_seconds"] = time.monotonic() - tic
    with path.open("xb") as stream:
        np.savez_compressed(stream, identity=json.dumps(identity, sort_keys=True), **result)
    print(f"cached {key}: {result['elapsed_seconds']:.1f}s; mass error "
          f"{result['grid_mass_relative_error']:+.2e}", flush=True)
    return result


def collect(run, rows, output, count=11, start=80, end=100,
            evolution=(1,4,8,12,20,40,60), reuse_final64=None, source_archive=None):
    native = round(rows[-1]["npart"]**(1/3))
    selected = select_samples(rows, start, end, count)
    tau = np.array([r["time"] / 0.05 for r in rows])
    extra = [rows[np.argmin(abs(tau - t))] for t in evolution]
    unique = {r["filename"]: r for r in selected + extra}
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    records = {}
    # Late diagnostics become available first, followed by the growth sequence.
    for row in unique.values():
        reuse = (reuse_final64 if native == 64 and
                 np.isclose(row["time"], 5) else None)
        records[row["filename"]] = cached_grid(row, 2*native, cache, reuse, source_archive)
    return {
        "run": str(run), "native": native, "rows": rows,
        "late": [records[r["filename"]] for r in selected],
        "evolution": sorted(records.values(), key=lambda r: float(r["time_crossings"])),
        "late_window": [start, end], "samples": count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-cache", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--window", type=float, nargs=2, default=(80,100))
    parser.add_argument("--reuse-final64", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    for run in args.runs:
        contract = audit_contract(run)
        launch = json.loads((Path(run)/"launch.json").read_text())
        native = launch["resolution"]
        rows = json.loads((args.history_cache/f"history_{native}.json").read_text())
        if len(rows) != 400 or any(Path(r["filename"]).parent.resolve() != Path(run).resolve() for r in rows):
            raise ValueError("history cache does not cover this completed run")
        destination = args.output/f"contract_{native}.json"
        if not destination.exists():
            write_json(destination, contract)
        collect(run, rows, args.output, args.samples, *args.window,
                reuse_final64=args.reuse_final64)
    print("Petkova ensemble collection complete", flush=True)


if __name__ == "__main__":
    main()
