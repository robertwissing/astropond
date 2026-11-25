#!/usr/bin/env bash
# Shared, non-code-specific helpers for content-addressed code builds.
# This file is sourced by codes/<code>/build_<code>.sh adapters.

build_common_die() {
    printf 'FATAL: %s\n' "$*" >&2
    return 1
}

build_sha256_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 | awk '{print $1}'
    else
        build_common_die "neither sha256sum nor shasum is available"
    fi
}

build_sha256_text() {
    printf '%s' "$1" | build_sha256_stream
}

build_sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    else
        build_common_die "neither sha256sum nor shasum is available"
    fi
}

build_source_fingerprint() {
    local src="$1"
    [ -d "$src" ] || build_common_die "source directory does not exist: $src"

    if git -C "$src" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        (
            cd "$src"
            # Include every tracked file, then source/build-like untracked
            # files.  Ignored objects/executables must not make a build change
            # its own source fingerprint.
            {
                git ls-files --cached -z
                git ls-files --others --exclude-standard -z |
                    while IFS= read -r -d '' path; do
                        case "$path" in
                            Makefile|Makefile.*|*/Makefile|*/Makefile.*|*.mk|*.c|*.h|*.cc|*.cpp|*.cxx|*.f|*.f90|*.inc|*.def)
                                printf '%s\0' "$path"
                                ;;
                        esac
                    done
            } |
                LC_ALL=C sort -z |
                while IFS= read -r -d '' path; do
                    [ -f "$path" ] || continue
                    printf '%s\0' "$path"
                    build_sha256_file "$path"
                done
        ) | build_sha256_stream
        return
    fi

    # Non-git fallback: hash build/source inputs, excluding generated objects and
    # executables. Paths are sorted so directory traversal order is irrelevant.
    (
        cd "$src"
        find . -type f \( -name 'Makefile*' -o -name '*.mk' -o \
            -name '*.c' -o -name '*.h' -o -name '*.cc' -o -name '*.cpp' -o \
            -name '*.f' -o -name '*.f90' -o -name '*.inc' \) -print0 |
            LC_ALL=C sort -z |
            while IFS= read -r -d '' path; do
                printf '%s\0' "$path"
                build_sha256_file "$path"
            done
    ) | build_sha256_stream
}

build_toolchain_fingerprint() {
    local compiler="$1"
    {
        printf 'compiler=%s\n' "$compiler"
        if command -v "$compiler" >/dev/null 2>&1; then
            command -v "$compiler"
            "$compiler" --version 2>&1 || true
            "$compiler" -show 2>&1 || true
        else
            printf 'compiler-unavailable\n'
        fi
        if command -v make >/dev/null 2>&1; then
            make --version 2>&1 | sed -n '1p'
        else
            printf 'make-unavailable\n'
        fi
    } | build_sha256_stream
}

# What -march=$2 means for the real compiler $1 (gcc/g++, not an MPI or Charm++
# wrapper) on THIS host: prints "<cpu> <sha256 of every target option it
# enables>". "native" names a different CPU on every node, so the fingerprint,
# not the word, is what a build key needs. Only the "-m<option> <value>" lines
# are hashed: gcc wraps the rest of --help to $COLUMNS.
build_cpu_target() {
    local options cpu
    options="$(LC_ALL=C COLUMNS=200 "$1" -march="$2" -Q --help=target 2>/dev/null |
        awk '$1 ~ /^-m/ {print $1, $2}')" || options=""
    cpu="$(awk '$1 == "-march=" {print $2; exit}' <<<"$options")"
    [ -n "$cpu" ] || { build_common_die "$1 cannot resolve -march=$2"; return 1; }
    printf '%s %s\n' "$cpu" "$(build_sha256_text "$options")"
}

# --- the RUNPHYSICS set -------------------------------------------------------
# A build requirement is a SET of capability tokens, written exactly as a test's
# setup_<test>() declares it in runtest.sh: bare physics tokens (hydro|mhd|grav),
# key=value axes (eos=barotropic, cooling=metal) and modules (starform=divvoff).
# An adapter takes the set WHOLE as its --physics -- the set defines the build --
# and these helpers are the only way it is ever taken apart.
build_set_value() {  # $1 = key, $2 = set -> its value ("" if bare); 1 if absent
    local key="$1" token
    for token in $2; do
        [ "$token" = "$key" ] && { printf '\n'; return 0; }
        case "$token" in "$key="*) printf '%s\n' "${token#*=}"; return 0 ;; esac
    done
    return 1
}
build_set_has() {  # $1 = key, $2 = set -> 0 if the set carries it in any form
    build_set_value "$1" "$2" >/dev/null
}
build_set_eos() {  # $1 = set -> the eos axis; the default lives here alone
    local value
    value="$(build_set_value eos "$1")" || value=""
    printf '%s\n' "${value:-adiabatic}"
}
build_set_modules() {  # $1 = set -> the tokens that are neither physics nor axis
    local token out=""
    for token in $1; do
        case "$token" in
            hydro|mhd|grav|fdm|eos=*|cooling=*) ;;
            *) out="${out:+$out }$token" ;;
        esac
    done
    printf '%s\n' "$out"
}
build_set_physics() {  # $1 = set -> its physics/eos tokens (the ADDITIVE ones)
    local token out=""
    for token in $1; do
        case "$token" in
            hydro|mhd|grav|fdm|eos=*) out="${out:+$out }$token" ;;
        esac
    done
    printf '%s\n' "$out"
}
# The build's NAME, composed from the same tokens rather than looked up in a
# table of combinations: <hydro|mhd>[_<eos>][_fdm]. Composing it means a new eos
# value names its builds automatically, which is the point of an additive map.
build_set_profile_name() {  # $1 = set -> the build name
    local name=hydro eos
    if build_set_has mhd "$1"; then name=mhd; fi
    eos="$(build_set_eos "$1")"
    case "$eos" in adiabatic) ;; *) name="${name}_$eos" ;; esac
    if build_set_has fdm "$1"; then name="${name}_fdm"; fi
    printf '%s\n' "$name"
}
build_set_put() {  # $1 = key, $2 = value, $3 = set -> the set with key=value forced
    local key="$1" value="$2" token out=""
    for token in $3; do
        case "$token" in "$key"|"$key="*) continue ;; esac
        out="${out:+$out }$token"
    done
    printf '%s\n' "${out:+$out }$key=$value"
}
