//========================================================================================
// from_gridic.cpp -- testsuite problem generator (copied into src/pgen at build)
//
// Initialises Athena++ from the testsuite's canonical GridIC stream binary
// (setupfiles/sph_interp/gridic.py): a fine UNIFORM field of
// (rho,vx,vy,vz,P[,Bx_l,Bx_r,By_l,By_r,Bz_l,Bz_r]) deposited from the SPH IC
// (petkova, mass-conserving) with the 6 div-free FACE-centred B components
// from project_solenoidal. Same file the RAMSES suite_mhd condinit reads.
//
// First cut: the mesh must MATCH the GridIC exactly (nx1,nx2,nx3) == (nx,ny,nz)
// -- the tipsy->athena converter generates the athinput that way -- so every
// cell/face is a direct lookup, no averaging. Face convention: our Bx_l of
// cell ix is the face at ix-1/2 == Athena's b.x1f(k,j,ix); the outermost
// b.x1f(ie+1) face uses Bx_r of the last cell (shared faces are stored
// redundantly and consistently by project_solenoidal). Cells are located by
// POSITION (mesh_size.x1min + dx), so any MeshBlock decomposition works.
//
// athinput:  <problem> input_filename = ic_gridic.bin  (default)
// A hydro GridIC (mhd flag 0) on an MHD build runs with B=0.
//========================================================================================

// C++ headers
#include <cmath>      // floor
#include <cstdint>    // int32_t
#include <fstream>    // ifstream
#include <iostream>   // cout
#include <mutex>      // call_once
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>
#include <vector>

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

constexpr std::int32_t GIC_MAGIC = 0x47524943;  // 'GRIC'

struct GridIC {
  std::int32_t nx = 0, ny = 0, nz = 0, mhd = 0;
  double boxlen = 1.0, gamma = 5.0 / 3.0, time = 0.0, box[3] = {1.0, 1.0, 1.0};
  std::vector<double> rho, vx, vy, vz, P;
  std::vector<double> bxl, bxr, byl, byr, bzl, bzr;

  // C-order flat index, ix slowest -- matches gridic.py
  std::size_t idx(int ix, int iy, int iz) const {
    return (static_cast<std::size_t>(ix) * ny + iy) * nz + iz;
  }
};

GridIC gic;
std::once_flag gic_once;

void ReadArray(std::ifstream &f, std::vector<double> &a, std::size_t n) {
  a.resize(n);
  f.read(reinterpret_cast<char *>(a.data()), n * sizeof(double));
}

void LoadGridIC(const std::string &filename) {
  std::ifstream f(filename, std::ios::binary);
  if (!f) {
    throw std::runtime_error("[from_gridic] cannot open " + filename);
  }
  std::int32_t hdr[6];  // magic, version, nx, ny, nz, nvar
  f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
  if (hdr[0] != GIC_MAGIC) {
    std::stringstream msg;
    msg << "[from_gridic] bad magic in " << filename << " (" << hdr[0] << ")";
    throw std::runtime_error(msg.str());
  }
  gic.nx = hdr[2]; gic.ny = hdr[3]; gic.nz = hdr[4];
  f.read(reinterpret_cast<char *>(&gic.mhd), sizeof(std::int32_t));
  double dhdr[6];  // boxlen, gamma, time, box(3)
  f.read(reinterpret_cast<char *>(dhdr), sizeof(dhdr));
  gic.boxlen = dhdr[0]; gic.gamma = dhdr[1]; gic.time = dhdr[2];
  gic.box[0] = dhdr[3]; gic.box[1] = dhdr[4]; gic.box[2] = dhdr[5];
  std::size_t n = static_cast<std::size_t>(gic.nx) * gic.ny * gic.nz;
  ReadArray(f, gic.rho, n);
  ReadArray(f, gic.vx, n);
  ReadArray(f, gic.vy, n);
  ReadArray(f, gic.vz, n);
  ReadArray(f, gic.P, n);
  if (gic.mhd != 0) {
    ReadArray(f, gic.bxl, n); ReadArray(f, gic.bxr, n);
    ReadArray(f, gic.byl, n); ReadArray(f, gic.byr, n);
    ReadArray(f, gic.bzl, n); ReadArray(f, gic.bzr, n);
  } else {
    gic.bxl.assign(n, 0.0); gic.bxr.assign(n, 0.0);
    gic.byl.assign(n, 0.0); gic.byr.assign(n, 0.0);
    gic.bzl.assign(n, 0.0); gic.bzr.assign(n, 0.0);
  }
  if (!f) {
    throw std::runtime_error("[from_gridic] short read from " + filename);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::string filename =
      pin->GetOrAddString("problem", "input_filename", "ic_gridic.bin");
  std::call_once(gic_once, LoadGridIC, filename);

  // first cut: the mesh must be the GridIC grid, cell for cell
  if (pmy_mesh->mesh_size.nx1 != gic.nx || pmy_mesh->mesh_size.nx2 != gic.ny ||
      pmy_mesh->mesh_size.nx3 != gic.nz) {
    std::stringstream msg;
    msg << "[from_gridic] mesh " << pmy_mesh->mesh_size.nx1 << "x"
        << pmy_mesh->mesh_size.nx2 << "x" << pmy_mesh->mesh_size.nx3
        << " != GridIC " << gic.nx << "x" << gic.ny << "x" << gic.nz
        << " (the athinput must match the GridIC resolution)";
    throw std::runtime_error(msg.str());
  }

  const Real x1min = pmy_mesh->mesh_size.x1min;
  const Real x2min = pmy_mesh->mesh_size.x2min;
  const Real x3min = pmy_mesh->mesh_size.x3min;
  const Real dx1 = (pmy_mesh->mesh_size.x1max - x1min) / gic.nx;
  const Real dx2 = (pmy_mesh->mesh_size.x2max - x2min) / gic.ny;
  const Real dx3 = (pmy_mesh->mesh_size.x3max - x3min) / gic.nz;
  const Real gm1 = peos->GetGamma() - 1.0;

  auto cell = [&](Real xv, Real xmin, Real dx, int n) {
    int ix = static_cast<int>(std::floor((xv - xmin) / dx));
    return ix < 0 ? 0 : (ix >= n ? n - 1 : ix);
  };

  // face-centred B first (the total energy needs the cell-averaged B)
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; k++) {
      int giz = cell(pcoord->x3v(k), x3min, dx3, gic.nz);
      for (int j = js; j <= je; j++) {
        int giy = cell(pcoord->x2v(j), x2min, dx2, gic.ny);
        for (int i = is; i <= ie + 1; i++) {
          // face i-1/2 lies at x1f(i): the owning cell's Bx_l, or the last
          // cell's Bx_r on the outermost face (redundant shared storage)
          int gif = static_cast<int>(
              std::floor((pcoord->x1f(i) - x1min) / dx1 + 0.5));
          pfield->b.x1f(k, j, i) = (gif < gic.nx)
              ? gic.bxl[gic.idx(gif, giy, giz)]
              : gic.bxr[gic.idx(gic.nx - 1, giy, giz)];
        }
      }
    }
    for (int k = ks; k <= ke; k++) {
      int giz = cell(pcoord->x3v(k), x3min, dx3, gic.nz);
      for (int j = js; j <= je + 1; j++) {
        int gjf = static_cast<int>(
            std::floor((pcoord->x2f(j) - x2min) / dx2 + 0.5));
        for (int i = is; i <= ie; i++) {
          int gix = cell(pcoord->x1v(i), x1min, dx1, gic.nx);
          pfield->b.x2f(k, j, i) = (gjf < gic.ny)
              ? gic.byl[gic.idx(gix, gjf, giz)]
              : gic.byr[gic.idx(gix, gic.ny - 1, giz)];
        }
      }
    }
    for (int k = ks; k <= ke + 1; k++) {
      int gkf = static_cast<int>(
          std::floor((pcoord->x3f(k) - x3min) / dx3 + 0.5));
      for (int j = js; j <= je; j++) {
        int giy = cell(pcoord->x2v(j), x2min, dx2, gic.ny);
        for (int i = is; i <= ie; i++) {
          int gix = cell(pcoord->x1v(i), x1min, dx1, gic.nx);
          pfield->b.x3f(k, j, i) = (gkf < gic.nz)
              ? gic.bzl[gic.idx(gix, giy, gkf)]
              : gic.bzr[gic.idx(gix, giy, gic.nz - 1)];
        }
      }
    }
  }

  // conserved hydro variables (direct lookup; mesh == GridIC grid)
  for (int k = ks; k <= ke; k++) {
    int giz = cell(pcoord->x3v(k), x3min, dx3, gic.nz);
    for (int j = js; j <= je; j++) {
      int giy = cell(pcoord->x2v(j), x2min, dx2, gic.ny);
      for (int i = is; i <= ie; i++) {
        int gix = cell(pcoord->x1v(i), x1min, dx1, gic.nx);
        std::size_t n = gic.idx(gix, giy, giz);
        Real rho = gic.rho[n];
        phydro->u(IDN, k, j, i) = rho;
        phydro->u(IM1, k, j, i) = rho * gic.vx[n];
        phydro->u(IM2, k, j, i) = rho * gic.vy[n];
        phydro->u(IM3, k, j, i) = rho * gic.vz[n];
        if (NON_BAROTROPIC_EOS) {
          Real e = gic.P[n] / gm1 +
                   0.5 * rho * (SQR(gic.vx[n]) + SQR(gic.vy[n]) + SQR(gic.vz[n]));
          if (MAGNETIC_FIELDS_ENABLED) {
            e += 0.5 * (SQR(0.5 * (pfield->b.x1f(k, j, i) + pfield->b.x1f(k, j, i+1))) +
                        SQR(0.5 * (pfield->b.x2f(k, j, i) + pfield->b.x2f(k, j+1, i))) +
                        SQR(0.5 * (pfield->b.x3f(k, j, i) + pfield->b.x3f(k+1, j, i))));
          }
          phydro->u(IEN, k, j, i) = e;
        }
      }
    }
  }

  if (gid == 0) {
    std::cout << "[from_gridic] loaded " << filename << " " << gic.nx << "x"
              << gic.ny << "x" << gic.nz << " boxlen=" << gic.boxlen
              << " mhd=" << gic.mhd << std::endl;
  }
  return;
}
