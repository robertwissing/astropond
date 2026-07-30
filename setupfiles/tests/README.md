# IC-tool tests

Automated tests for the three IC-editing tools in `setupfiles/`:

| module | covers |
|---|---|
| `test_setbfield.py` | `IC_setBField.py` — direction/profile/strength algebra, the turbulent draw, the CLI |
| `test_setquantity.py` | `IC_setQuantity.py` — the operation vocabulary, chaining, the aux twins |
| `test_setup_impact.py` | `IC_setup_impact.py` — orbit algebra, the overlap guard, the CLI |

Run from anywhere:

```sh
conda run -n glass python -m pytest setupfiles/tests -q -p no:dash
```

`-p no:dash` is needed on this machine (a dash plugin breaks collection);
the `glass` env is the one with numba and the glassgen dependencies.

## What these are weighted toward

Not coverage for its own sake. Each module targets the failure modes that
have actually occurred in this code and that produce a **plausible-looking
output file** rather than a crash:

* **silent no-ops** — the old `setQuantity.py` computed `B/rho` and then wrote
  the original array back (twice), and advertised a `smooth` operation that
  had no implementation at all. So every operation test asserts the output
  *changed in the way asked for*, never merely that the run exited 0.
* **half-updated vector aux** — an edit that touches the packed `.BField` but
  not the `.BFieldx/y/z` twins leaves two disagreeing copies of the field.
  `check_twins()` is applied after every vector operation, and it checks
  length as well as values (a gas-only twin beside a full-length packed file
  is a bug that shipped once already).
* **a physically wrong field that looks fine** — the turbulent profile is
  checked against things a mis-drawn field cannot fake: solenoidality to
  machine precision in Fourier space, the requested filter dividing out
  *exactly* at fixed seed regardless of spectral index, and the realised
  spectral slope over several seeds.
* **orbits that are wrong for a week before anyone notices** — the keplerian
  branch is checked against closed forms (parabolic ⇒ escape speed, `L = r v_t`
  constant, purely tangential at pericentre).

## Fixtures

`conftest.py` builds synthetic tipsy snapshots directly with numpy rather than
through `readtipsy`'s writers, so a bug in the shared I/O module cannot mask a
bug in the tool under test. `disc_snapshot` (gas + dark + a `BField` aux) and
`periodic_snapshot` (uniform box) cover almost everything; `make_disc()` is
there for tests that need their own geometry.
