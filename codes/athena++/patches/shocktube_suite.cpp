//========================================================================================
//! \file shocktube_suite.cpp
//! \brief 1-D Riemann problem AS THE TESTSUITE SETS IT UP (IC_setup_shocktube).
//!
//! Athena++'s own shock_tube.cpp initialises a clean two-state jump with outflow
//! boundaries -- the textbook Riemann problem. Our SPH runs are NOT that. They use
//!   * a PERIODIC box, twice as long as the shock region (setup_shocktube.ti = 2),
//!     so a SECOND (image) Riemann fan develops at the wrapped edge, and
//!   * an exponential velocity taper near the box edges, applied by
//!     IC_setup_shocktube.create as
//!         lim = 0.4 * dxbound            (dxbound = x1max - x1min, the FULL length)
//!         v  *= exp(-rate * (|x| - lim))    for |x| > lim,  rate = 8
//!     which softens that image discontinuity but is itself extra structure.
//! Comparing such a run against a central-fan-only table means the box edges (and,
//! for the fast-inflow presets, most of the box) are scored as run error.
//!
//! This generator reproduces that initial state exactly, so a high-resolution run of
//! it is a reference valid over the WHOLE box, at the run's own time -- no mirroring
//! assumption, no clipping. Driven by setupfiles/shocktube_reference.py.
//!
//! Left/right states use the same parameter names and the same 1/sqrt(4pi) magnetic
//! normalisation as Athena++'s shock_tube.cpp (dl,pl,ul,vl,wl,bxl,byl,bzl and the r
//! equivalents), which is also IC_setup_shocktube's convention, so the states map
//! across verbatim. Shock is always along x1; set taper_lim >= half the box (and
//! outflow boundaries in the athinput) to recover the plain Riemann problem.
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // std::abs(), std::exp()
#include <sstream>    // stringstream

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

namespace {
// The IC_setup_shocktube taper: velocities only, symmetric about x=0, unity inside.
Real TaperFac(Real x, Real lim, Real rate) {
  Real ax = std::abs(x);
  return (ax > lim) ? std::exp(-rate*(ax - lim)) : 1.0;
}
}  // namespace

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

  Real xshock = pin->GetOrAddReal("problem", "xshock", 0.0);
  if (xshock < pmy_mesh->mesh_size.x1min || xshock > pmy_mesh->mesh_size.x1max) {
    msg << "### FATAL ERROR in shocktube_suite" << std::endl
        << "xshock=" << xshock << " lies outside the x1 domain" << std::endl;
    ATHENA_ERROR(msg);
  }
  // taper_lim defaults to the suite's own 0.4*dxbound; taper_rate to its 8.
  Real dxbound = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Real lim  = pin->GetOrAddReal("problem", "taper_lim", 0.4*dxbound);
  Real rate = pin->GetOrAddReal("problem", "taper_rate", 8.0);

  Real wl[NHYDRO+NFIELD], wr[NHYDRO+NFIELD];
  wl[IDN] = pin->GetReal("problem", "dl");
  wl[IVX] = pin->GetReal("problem", "ul");
  wl[IVY] = pin->GetReal("problem", "vl");
  wl[IVZ] = pin->GetReal("problem", "wl");
  wr[IDN] = pin->GetReal("problem", "dr");
  wr[IVX] = pin->GetReal("problem", "ur");
  wr[IVY] = pin->GetReal("problem", "vr");
  wr[IVZ] = pin->GetReal("problem", "wr");
  if (NON_BAROTROPIC_EOS) {
    wl[IPR] = pin->GetReal("problem", "pl");
    wr[IPR] = pin->GetReal("problem", "pr");
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    wl[NHYDRO  ] = pin->GetReal("problem", "bxl");
    wl[NHYDRO+1] = pin->GetReal("problem", "byl");
    wl[NHYDRO+2] = pin->GetReal("problem", "bzl");
    wr[NHYDRO  ] = pin->GetReal("problem", "bxr");
    wr[NHYDRO+1] = pin->GetReal("problem", "byr");
    wr[NHYDRO+2] = pin->GetReal("problem", "bzr");
  }

  // --- conserved hydro variables (tapered velocities) ---
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x = pcoord->x1v(i);
        const Real *w = (x < xshock) ? wl : wr;
        Real f = TaperFac(x, lim, rate);
        Real vx = f*w[IVX], vy = f*w[IVY], vz = f*w[IVZ];
        phydro->u(IDN,k,j,i) = w[IDN];
        phydro->u(IM1,k,j,i) = vx*w[IDN];
        phydro->u(IM2,k,j,i) = vy*w[IDN];
        phydro->u(IM3,k,j,i) = vz*w[IDN];
        if (NON_BAROTROPIC_EOS) {
          if (GENERAL_EOS) {
            phydro->u(IEN,k,j,i) = peos->EgasFromRhoP(w[IDN], w[IPR]);
          } else {
            phydro->u(IEN,k,j,i) = w[IPR]/(peos->GetGamma() - 1.0);
          }
          phydro->u(IEN,k,j,i) += 0.5*w[IDN]*(vx*vx + vy*vy + vz*vz);
        }
      }
    }
  }

  // --- face-centred fields ---
  // B is NOT tapered (only velocities are). B_x must be the same in both states for
  // div.B = 0 in 1-D; a preset that jumps B_x is not a well-posed Riemann problem.
  if (MAGNETIC_FIELDS_ENABLED) {
    if (std::abs(wl[NHYDRO] - wr[NHYDRO]) > 1e-12*(1.0 + std::abs(wl[NHYDRO]))) {
      msg << "### FATAL ERROR in shocktube_suite" << std::endl
          << "bxl=" << wl[NHYDRO] << " != bxr=" << wr[NHYDRO]
          << ": a jump in B_x violates div.B = 0 in 1-D" << std::endl;
      ATHENA_ERROR(msg);
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {   // x1f is a face: use the face coordinate
          pfield->b.x1f(k,j,i) = (pcoord->x1f(i) < xshock) ? wl[NHYDRO] : wr[NHYDRO];
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          const Real *w = (pcoord->x1v(i) < xshock) ? wl : wr;
          pfield->b.x2f(k,j,i) = w[NHYDRO+1];
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          const Real *w = (pcoord->x1v(i) < xshock) ? wl : wr;
          pfield->b.x3f(k,j,i) = w[NHYDRO+2];
        }
      }
    }
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            const Real *w = (pcoord->x1v(i) < xshock) ? wl : wr;
            phydro->u(IEN,k,j,i) += 0.5*(SQR(w[NHYDRO]) + SQR(w[NHYDRO+1])
                                         + SQR(w[NHYDRO+2]));
          }
        }
      }
    }
  }
}
