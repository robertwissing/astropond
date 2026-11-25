#!/usr/bin/env bash
# Build Athena++ for the testsuite (code=7 in runtest.sh).
#
# DEPENDENCIES: python3 (configure.py), g++, mpicxx/mpicc (Intel oneAPI here,
#   wrapping gcc), curl (one-time HDF5 source download). Run get_athena.sh first.
#
# PARALLELISM: MPI by default. Athena++'s .athdf writer needs a PARALLEL HDF5
#   under -mpi and the system only ships the serial lib, so this script first
#   builds a local parallel HDF5 (CC=mpicc --enable-parallel) into
#   $here/hdf5-parallel (one-time, idempotent) and configures Athena++ with
#   --hdf5_path + an rpath so no LD_LIBRARY_PATH is needed at run time.
#   MPI ranks own MeshBlocks, so the athinput generator must split the mesh
#   into >= nranks meshblocks (ICconvert_tipsytoathena.py --np).
#
# The suite feeds ICs via our problem generator patches/from_gridic.cpp
# (reads the canonical GridIC binary — the SAME ic_gridic.bin RAMSES uses),
# copied into src/pgen/ at build time. Two artifacts (both are RENAMED away
# from bin/athena because `make clean` — run before each configure — deletes
# exactly that path and would eat a sibling build's binary):
#   - athena_hydro : --prob=from_gridic          (hydro)
#   - athena_mhd   : --prob=from_gridic -b       (MHD, constrained transport)
Two more binaries build 1-D shock-tube reference solutions. Both take the
# left/right states straight from the athinput (dl/pl/ul/.../bzl -- the same
# 1/sqrt(4pi) B normalisation IC_setup_shocktube uses), so neither needs an IC
# file or a GridIC conversion:
#   - athena_shocktube       : --prob=shock_tube      -b   (MHD/CT)
#       Athena++'s own pgen: the clean two-state Riemann problem. Used to CHECK
#       the transcribed SPLASH tables in IC_analysis_shocktube.MHD_REF.
#   - athena_shocktube_suite : --prob=shocktube_suite -b   (MHD/CT)
#       our patches/shocktube_suite.cpp: the SAME initial state the SPH runs get
#       (periodic box + IC_setup_shocktube's exponential velocity taper), so its
#       output is a reference over the whole box rather than the central fan.
#   - athena_shocktube_iso   : the same, --eos isothermal, for the gamma=1 presets
#       (steady shock #8, isothermal B98 #13). Pressure comes from
#       <hydro> iso_sound_speed instead of pl/pr. NOT usable for the C-shock (#7),
#       which is held together by ambipolar diffusion, not ideal MHD.
# All are driven by setupfiles/shocktube_reference.py.
# Idempotent (skips an existing binary).
#
# UNSUPPORTED TESTS: SPH/gravity-centric tests (accdisk, coldkeplerian, planet,
#   planetcollision, evrard, polytrope self-gravity) and cosmology (zeldovich,
#   cosmowave); non-cubic (slab) boxes need a non-cubic GridIC strategy first
#   (same limitation as RAMSES — kh, 2D orzag slabs).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SRC="${ATHENA_SRC:-$here/athena}"
JOBS="${JOBS:-$(nproc)}"
H5DIR="${ATHENA_HDF5_DIR:-$here/hdf5-parallel}"
H5VER="${ATHENA_HDF5_VER:-1.14.5}"

[ -d "$SRC" ] || { echo "FATAL: $SRC missing — run get_athena.sh first"; exit 1; }
[ -f "$here/patches/from_gridic.cpp" ] || { echo "FATAL: patches/from_gridic.cpp missing"; exit 1; }

# one-time local PARALLEL HDF5 (the system lib is serial-only)
if [ -f "$H5DIR/lib/libhdf5.so" ]; then
    echo "parallel HDF5 already built: $H5DIR"
else
    echo "Building parallel HDF5 $H5VER -> $H5DIR"
    tarball="hdf5-$H5VER.tar.gz"
    ( mkdir -p "$here/hdf5-build" && cd "$here/hdf5-build"
      [ -f "$tarball" ] || curl -fsSL -o "$tarball" \
        "https://github.com/HDFGroup/hdf5/releases/download/hdf5_$H5VER/$tarball"
      tar xzf "$tarball"
      cd "hdf5-$H5VER"
      CC=mpicc ./configure --enable-parallel --disable-fortran --disable-cpp \
                           --disable-tools --disable-tests --prefix="$H5DIR" > configure.log
      make -j"$JOBS" > make.log 2>&1
      make install > install.log 2>&1 )
    [ -f "$H5DIR/lib/libhdf5.so" ] || { echo "FATAL: parallel HDF5 build failed"; exit 1; }
    echo "Done: $H5DIR"
fi

cp "$here/patches/from_gridic.cpp"      "$SRC/src/pgen/from_gridic.cpp"
cp "$here/patches/shocktube_suite.cpp" "$SRC/src/pgen/shocktube_suite.cpp"

build_one () {  # <exec> <pgen> [extra configure args...]
    local exec="$1" prob="$2"; shift 2
    if [ -x "$SRC/bin/$exec" ]; then echo "Athena++ already built: $SRC/bin/$exec"; return 0; fi
    echo "Building Athena++ -> $exec (--prob=$prob $* -mpi -hdf5)"
    ( cd "$SRC" && python3 configure.py --prob="$prob" "$@" -mpi -hdf5 \
        --hdf5_path="$H5DIR" --cflag="-Wl,-rpath,$H5DIR/lib"
      make clean >/dev/null 2>&1 || true
      make -j"$JOBS"
      mv bin/athena "bin/$exec" )
    [ -x "$SRC/bin/$exec" ] || { echo "FATAL: build produced no $exec"; exit 1; }
    echo "Done: $SRC/bin/$exec"
}

build_one athena_hydro     from_gridic
build_one athena_mhd       from_gridic -b
build_one athena_shocktube       shock_tube      -b
build_one athena_shocktube_suite shocktube_suite -b
build_one athena_shocktube_iso   shocktube_suite -b --eos isothermal

echo ""
echo "Athena++ build complete:"
echo "  $SRC/bin/athena_hydro (hydro, from_gridic pgen, MPI)"
echo "  $SRC/bin/athena_mhd   (MHD/CT, from_gridic pgen, MPI)"
echo "  $SRC/bin/athena_shocktube       (MHD/CT, clean Riemann problem)"
echo "  $SRC/bin/athena_shocktube_suite (MHD/CT, suite IC: periodic + velocity taper)"
echo "  $SRC/bin/athena_shocktube_iso   (as above, isothermal EOS)"
echo "    both driven by setupfiles/shocktube_reference.py"
echo "Run with mpirun -np N (runtest: ATHENANP, default nproc); the athinput"
echo "  must have >= N meshblocks (the converter's --np handles it)."
