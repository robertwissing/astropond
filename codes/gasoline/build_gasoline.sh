#!/usr/bin/env bash
# Resolve reproducible Gasoline executables for the testsuite.
#
# Test setups pass semantic requirements (physics and EOS).  This adapter owns
# their translation to Gasoline -D flags, content-addressed builds, and build
# provenance.  It never edits or cleans the Gasoline source tree.
set -euo pipefail
umask 022

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
codes_dir="$(cd "$here/.." && pwd -P)"
common_helpers="$codes_dir/build_common.sh"
config_file="${GASOLINE_BUILD_CONFIG:-$here/build_config.sh}"
extra_flags_file="${CODE_EXTRA_BUILD_FLAGS_FILE:-$codes_dir/extra_build_flags.sh}"
if [[ "$extra_flags_file" != /* ]]; then extra_flags_file="$(pwd -P)/$extra_flags_file"; fi

[ -r "$common_helpers" ] || { printf 'FATAL: missing build helpers: %s\n' "$common_helpers" >&2; exit 1; }
[ -r "$config_file" ] || { printf 'FATAL: missing Gasoline build config: %s\n' "$config_file" >&2; exit 1; }
[ -r "$extra_flags_file" ] || { printf 'FATAL: missing extra build-flags file: %s\n' "$extra_flags_file" >&2; exit 1; }
# shellcheck source=../build_common.sh
source "$common_helpers"
# shellcheck source=build_config.sh
source "$config_file"
# shellcheck source=../extra_build_flags.sh
source "$extra_flags_file"

gasoline_die() {
    printf 'FATAL: %s\n' "$*" >&2
    exit 1
}

gasoline_usage() {
    cat <<'EOF'
Usage:
  build_gasoline.sh capabilities
  build_gasoline.sh supports --physics SET
  build_gasoline.sh explain  --physics SET [options]
  build_gasoline.sh resolve  --physics SET [options]
  build_gasoline.sh all [options]
  build_gasoline.sh list [--build-root PATH]

Requirements:
  --physics "tok tok=variant ..."                the RUNPHYSICS set: it DEFINES the build
                                                 (hydro|mhd|grav, eos=, cooling=, modules)
  --eos     adiabatic|barotropic|isor            override, folded into the set
  --cooling none|metal                           override, folded into the set
  --modules "tok tok=variant"                    override, appended to the set

Build options:
  --kernel NAME          Kernel macro, with or without -D
  --define NAME[=VALUE]  Extra macro; repeat as needed
  -DNAME[=VALUE]         Short form of --define
  --source PATH          Gasoline source tree
  --build-root PATH      Immutable artifact root
  --target NAME          Gasoline Makefile target
  --jobs N               Parallel make jobs
  --mode auto|reuse-only Build cache misses or require a prebuilt artifact

Environment equivalents:
  GASOLINE_SRC, GASOLINE_BUILD_CONFIG, GASOLINE_KERNEL, GASOLINE_COOLING,
  CODE_EXTRA_BUILD_FLAGS_FILE, GASOLINE_DEFINES (legacy/temporary override),
  GASOLINE_BUILD_ROOT, GASOLINE_CPU_TARGET (-march/-mtune value, "" = generic),
  GASOLINE_TARGET, GASOLINE_BUILD_MODE, JOBS

Examples:
  build_gasoline.sh explain --physics "mhd eos=barotropic"
  build_gasoline.sh all --define CORRPARTITION
  Edit GASOLINE_EXTRA_BUILD_FLAGS in codes/extra_build_flags.sh, then run:
  build_gasoline.sh all
EOF
}

for required_symbol in \
    GASOLINE_REFERENCE_SRC GASOLINE_DEFAULT_TARGET GASOLINE_DEFAULT_COMPILER \
    GASOLINE_DEFAULT_KERNEL GASOLINE_SOURCE_INPUTS GASOLINE_COMMON_DEFINES \
    GASOLINE_BUILD_SETS; do
    declare -p "$required_symbol" >/dev/null 2>&1 ||
        gasoline_die "$config_file does not define $required_symbol"
done
declare -F gasoline_profile_for >/dev/null || gasoline_die "$config_file does not define gasoline_profile_for"
declare -F gasoline_physics_defines >/dev/null || gasoline_die "$config_file does not define gasoline_physics_defines"
declare -F gasoline_module_defines >/dev/null || gasoline_die "$config_file does not define gasoline_module_defines"
declare -F gasoline_print_capabilities >/dev/null || gasoline_die "$config_file does not define gasoline_print_capabilities"
extra_flags_declaration="$(declare -p GASOLINE_EXTRA_BUILD_FLAGS 2>/dev/null)" ||
    gasoline_die "$extra_flags_file does not define GASOLINE_EXTRA_BUILD_FLAGS"
[[ "$extra_flags_declaration" == declare\ -a* ]] ||
    gasoline_die "GASOLINE_EXTRA_BUILD_FLAGS in $extra_flags_file must be an indexed array"

command_name="${1:-}"
case "$command_name" in
    capabilities|supports|explain|resolve|all|list)
        shift
        ;;
    -h|--help|help|"")
        gasoline_usage
        exit 0
        ;;
    *)
        gasoline_usage >&2
        gasoline_die "unknown command: $command_name"
        ;;
esac

physics=""
eos_cli=""
modules_cli=""
cooling_cli=""
cooling_default="${GASOLINE_COOLING:-${GASOLINE_DEFAULT_COOLING:-}}"
kernel="${GASOLINE_KERNEL:-$GASOLINE_DEFAULT_KERNEL}"
source_dir="${GASOLINE_SRC:-$GASOLINE_REFERENCE_SRC}"
build_root="${GASOLINE_BUILD_ROOT:-$here/builds}"
make_target="${GASOLINE_TARGET:-$GASOLINE_DEFAULT_TARGET}"
compiler="$GASOLINE_DEFAULT_COMPILER"
cpu_target="${GASOLINE_CPU_TARGET:-}"
build_mode="${GASOLINE_BUILD_MODE:-auto}"
jobs="${JOBS:-$(nproc 2>/dev/null || printf '1')}"
declare -a user_defines=()

if [ "${#GASOLINE_EXTRA_BUILD_FLAGS[@]}" -gt 0 ]; then
    user_defines+=("${GASOLINE_EXTRA_BUILD_FLAGS[@]}")
fi

if [ -n "${GASOLINE_DEFINES:-}" ]; then
    # This environment form is deliberately whitespace separated; definitions
    # containing shell syntax or whitespace are not accepted below.
    read -r -a env_defines <<<"$GASOLINE_DEFINES"
    user_defines+=("${env_defines[@]}")
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --physics)
            [ "$#" -ge 2 ] || gasoline_die "--physics requires a value"
            physics="$2"; shift 2
            ;;
        --physics=*)
            physics="${1#*=}"; shift
            ;;
        --eos)
            [ "$#" -ge 2 ] || gasoline_die "--eos requires a value"
            eos_cli="$2"; shift 2
            ;;
        --eos=*)
            eos_cli="${1#*=}"; shift
            ;;
        --cooling)
            [ "$#" -ge 2 ] || gasoline_die "--cooling requires a value"
            cooling_cli="$2"; shift 2
            ;;
        --cooling=*)
            cooling_cli="${1#*=}"; shift
            ;;
        --modules)
            [ "$#" -ge 2 ] || gasoline_die "--modules requires a value"
            modules_cli="$2"; shift 2
            ;;
        --modules=*)
            modules_cli="${1#*=}"; shift
            ;;
        --kernel)
            [ "$#" -ge 2 ] || gasoline_die "--kernel requires a value"
            kernel="$2"; shift 2
            ;;
        --kernel=*)
            kernel="${1#*=}"; shift
            ;;
        --define|-D)
            [ "$#" -ge 2 ] || gasoline_die "$1 requires a value"
            user_defines+=("$2"); shift 2
            ;;
        --define=*)
            user_defines+=("${1#*=}"); shift
            ;;
        -D?*)
            user_defines+=("$1"); shift
            ;;
        --source)
            [ "$#" -ge 2 ] || gasoline_die "--source requires a value"
            source_dir="$2"; shift 2
            ;;
        --source=*)
            source_dir="${1#*=}"; shift
            ;;
        --build-root)
            [ "$#" -ge 2 ] || gasoline_die "--build-root requires a value"
            build_root="$2"; shift 2
            ;;
        --build-root=*)
            build_root="${1#*=}"; shift
            ;;
        --target)
            [ "$#" -ge 2 ] || gasoline_die "--target requires a value"
            make_target="$2"; shift 2
            ;;
        --target=*)
            make_target="${1#*=}"; shift
            ;;
        --jobs)
            [ "$#" -ge 2 ] || gasoline_die "--jobs requires a value"
            jobs="$2"; shift 2
            ;;
        --jobs=*)
            jobs="${1#*=}"; shift
            ;;
        --mode)
            [ "$#" -ge 2 ] || gasoline_die "--mode requires a value"
            build_mode="$2"; shift 2
            ;;
        --mode=*)
            build_mode="${1#*=}"; shift
            ;;
        -h|--help)
            gasoline_usage
            exit 0
            ;;
        *)
            gasoline_die "unknown option: $1"
            ;;
    esac
done

# --physics IS the set and defines the build. The standalone --eos/--cooling/
# --modules flags stay for hand-typed invocations and are folded INTO it, so
# there is still exactly one object to resolve and to report.
if [ -n "$eos_cli" ];     then physics="$(build_set_put eos "$eos_cli" "$physics")"; fi
if [ -n "$cooling_cli" ]; then physics="$(build_set_put cooling "$cooling_cli" "$physics")"; fi
if [ -n "$modules_cli" ]; then physics="$physics $modules_cli"; fi
eos="$(build_set_eos "$physics")"
cooling="$(build_set_value cooling "$physics")" || cooling=""
[ -n "$cooling" ] || cooling="$cooling_default"
modules="$(build_set_modules "$physics")"

case "$build_mode" in auto|reuse-only) ;; *) gasoline_die "invalid build mode: $build_mode" ;; esac
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || gasoline_die "--jobs must be a positive integer"
[[ "$make_target" =~ ^[A-Za-z0-9_.+-]+$ ]] || gasoline_die "unsafe Makefile target: $make_target"
[[ "$compiler" =~ ^[A-Za-z0-9_./+-]+$ ]] || gasoline_die "unsafe compiler name: $compiler"
[[ "$cpu_target" =~ ^[A-Za-z0-9_.-]*$ ]] || gasoline_die "unsafe GASOLINE_CPU_TARGET: $cpu_target"

absolute_path() {
    local path="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath -m "$path"
    elif [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$PWD" "$path"
    fi
}

source_dir="$(absolute_path "$source_dir")"
build_root="$(absolute_path "$build_root")"

normalize_define() {
    local value="${1#-D}"
    [[ "$value" =~ ^[A-Za-z_][A-Za-z0-9_]*(=[A-Za-z0-9_./:+-]+)?$ ]] ||
        gasoline_die "invalid or unsafe preprocessor definition: $1"
    printf '%s\n' "$value"
}

kernel="$(normalize_define "$kernel")"
declare -a normalized_user_defines=()
for define in "${user_defines[@]}"; do
    normalized_user_defines+=("$(normalize_define "$define")")
done
if [ "${#normalized_user_defines[@]}" -gt 0 ]; then
    mapfile -t normalized_user_defines < <(printf '%s\n' "${normalized_user_defines[@]}" | LC_ALL=C sort -u)
fi

for define in "${normalized_user_defines[@]}"; do
    define_name="${define%%=*}"
    case "$define_name" in
        MHD|BAROTROPIC|ISOR)
            gasoline_die "$define_name is profile-required; do not pass it as a user extra"
            ;;
    esac
done

# Cooling module. Selected through make variables rather than CODE_DEF, so it
# needs its own validation, its own key material, and its own cache record.
declare -F gasoline_cooling_make_vars >/dev/null 2>&1 ||
    gasoline_die "$config_file defines no gasoline_cooling_make_vars (cooling axis)"
cooling_options_declaration="$(declare -p GASOLINE_COOLING_OPTIONS 2>/dev/null)" ||
    gasoline_die "$config_file does not define GASOLINE_COOLING_OPTIONS"
[[ "$cooling_options_declaration" == "declare -a"* ]] ||
    gasoline_die "GASOLINE_COOLING_OPTIONS in $config_file must be an indexed array"
[ -n "$cooling" ] ||
    gasoline_die "no cooling module selected and $config_file sets no GASOLINE_DEFAULT_COOLING"
cooling_known=0
for candidate in "${GASOLINE_COOLING_OPTIONS[@]}"; do
    [ "$candidate" = "$cooling" ] && cooling_known=1
done
[ "$cooling_known" = 1 ] ||
    gasoline_die "unknown cooling module: $cooling (available: ${GASOLINE_COOLING_OPTIONS[*]})"

# The cooling selection is part of the build's identity, so it travels in the
# profile LABEL: one string then carries the whole selection through the cache
# directory, the manifest and `supports`, and every existing profile check
# covers cooling with no separate comparison.
profile_label_for() {
    local label="$1-$cooling" token
    for token in $modules; do
        label="$label-${token//[!A-Za-z0-9]/-}"
    done
    printf '%s\n' "$label"
}

requested_profile() {
    [ -n "$physics" ] || gasoline_die "$command_name requires --physics"
    local profile
    if ! profile="$(gasoline_profile_for "$physics")"; then
        printf 'Unsupported Gasoline build requirement: %s\n' "$physics" >&2
        return 1
    fi
    printf '%s\n' "$profile"
}

if [ "$command_name" = capabilities ]; then
    gasoline_print_capabilities
    exit 0
fi

if [ "$command_name" = supports ]; then
    supports_profile="$(requested_profile)" || exit $?
    profile_label_for "$supports_profile"
    exit 0
fi

declare -a effective_defines=()
declare -a effective_flags=()
declare -a effective_make_vars=()
declare -a source_input_records=()
declare -A define_values=()
declare -A define_origins=()

add_effective_define() {
    local define="$1" origin="$2" name="${1%%=*}"
    if [ -n "${define_values[$name]+set}" ]; then
        if [ "${define_values[$name]}" != "$define" ]; then
            gasoline_die "conflicting definitions for $name: ${define_values[$name]} (${define_origins[$name]}) versus $define ($origin)"
        fi
        return
    fi
    define_values[$name]="$define"
    define_origins[$name]="$origin"
    effective_defines+=("$define")
    effective_flags+=("-D$define")
}

prepare_defines() {
    local profile="$1" define profile_define_text token module_define_text
    effective_defines=()
    effective_flags=()
    define_values=()
    define_origins=()

    for define in "${GASOLINE_COMMON_DEFINES[@]}"; do
        add_effective_define "$(normalize_define "$define")" common
    done
    add_effective_define "$kernel" kernel
    for token in $(build_set_physics "$physics"); do
        if ! profile_define_text="$(gasoline_physics_defines "$token")"; then
            gasoline_die "$config_file has no Gasoline build for physics token: $token"
        fi
        while IFS= read -r define; do
            [ -n "$define" ] && add_effective_define "$(normalize_define "$define")" "physics:$token"
        done <<<"$profile_define_text"
    done
    for token in $modules; do
        if ! module_define_text="$(gasoline_module_defines "$token")"; then
            gasoline_die "unknown Gasoline physics module in $config_file: $token"
        fi
        while IFS= read -r define; do
            [ -n "$define" ] && add_effective_define "$(normalize_define "$define")" "module:$token"
        done <<<"$module_define_text"
    done
    for define in "${normalized_user_defines[@]}"; do
        add_effective_define "$define" user
    done
}

prepare_cooling() {
    local make_var make_var_text
    effective_make_vars=()
    if ! make_var_text="$(gasoline_cooling_make_vars "$cooling")"; then
        gasoline_die "build config rejected cooling module: $cooling"
    fi
    while IFS= read -r make_var; do
        [ -n "$make_var" ] || continue
        # One NAME=VALUE make assignment; VALUE may be empty (COOLING_OBJ=) and
        # may carry -D flags or several space-separated link tokens.
        [[ "$make_var" =~ ^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./:+=-]*([[:space:]]+[A-Za-z0-9_./:+=-]+)*$ ]] ||
            gasoline_die "invalid or unsafe cooling make variable: $make_var"
        effective_make_vars+=("$make_var")
    done <<<"$make_var_text"
    [ "${#effective_make_vars[@]}" -gt 0 ] ||
        gasoline_die "cooling module $cooling produced no make variables"
}

profile=""
base_profile=""
source_fingerprint=""
toolchain_fingerprint=""
config_fingerprint=""
helpers_fingerprint=""
adapter_fingerprint=""
build_key=""
short_key=""
cache_dir=""
executable_path=""
key_material=""

capture_source_inputs() {
    local spec input_path input_fingerprint
    source_input_records=()
    for spec in "${GASOLINE_SOURCE_INPUTS[@]}"; do
        if [[ "$spec" = /* ]]; then
            input_path="$(absolute_path "$spec")"
        else
            input_path="$(absolute_path "$source_dir/$spec")"
        fi
        [ -d "$input_path" ] || gasoline_die "declared Gasoline source input does not exist: $input_path"
        input_fingerprint="$(build_source_fingerprint "$input_path")"
        source_input_records+=("$spec"$'\t'"$input_path"$'\t'"$input_fingerprint")
    done
    source_fingerprint="$(printf '%s\n' "${source_input_records[@]}" | build_sha256_stream)"
}

prepare_context() {
    local selected_profile="$1" define make_var
    [ -d "$source_dir" ] || gasoline_die "Gasoline source directory does not exist: $source_dir"
    [ -f "$source_dir/Makefile" ] || gasoline_die "Gasoline Makefile is missing: $source_dir/Makefile"
    base_profile="$selected_profile"
    profile="$(profile_label_for "$selected_profile")"
    prepare_defines "$base_profile"
    prepare_cooling
    config_fingerprint="$(build_sha256_file "$config_file")"
    helpers_fingerprint="$(build_sha256_file "$common_helpers")"
    adapter_fingerprint="$(build_sha256_file "${BASH_SOURCE[0]}")"
    capture_source_inputs
    toolchain_fingerprint="$(build_toolchain_fingerprint "$compiler")"
    cpu_resolved=""
    cpu_flags=""
    if [ -n "$cpu_target" ]; then
        # An MPI wrapper names its real compiler first in -show; a bare compiler has no -show.
        real_compiler="$("$compiler" -show 2>/dev/null | awk 'NR == 1 {print $1}')" || real_compiler=""
        cpu_resolved="$(build_cpu_target "${real_compiler:-$compiler}" "$cpu_target")" || exit 1
        cpu_flags="-march=$cpu_target -mtune=$cpu_target"
    fi
    # Every Makefile target folds CODE_DEF into its CFLAGS, so it carries the CPU flags too.
    code_def="${effective_flags[*]}${cpu_flags:+ $cpu_flags}"
    key_material="$({
        printf 'schema=3\nadapter=gasoline\nprofile=%s\ncooling=%s\n' "$profile" "$cooling"
        printf 'cpu_target=%s\ncpu=%s\n' "$cpu_target" "$cpu_resolved"
        printf 'target=%s\ncompiler=%s\n' "$make_target" "$compiler"
        printf 'config=%s\nhelpers=%s\nadapter_script=%s\n' \
            "$config_fingerprint" "$helpers_fingerprint" "$adapter_fingerprint"
        printf 'source_path=%s\nsource=%s\ntoolchain=%s\n' \
            "$source_dir" "$source_fingerprint" "$toolchain_fingerprint"
        for define in "${effective_defines[@]}"; do
            printf 'define=%s\n' "$define"
        done
        for make_var in "${effective_make_vars[@]}"; do
            printf 'make_var=%s\n' "$make_var"
        done
    })"
    build_key="$(build_sha256_text "$key_material")"
    short_key="${build_key:0:12}"
    cache_dir="$build_root/$profile-$short_key"
    executable_path="$cache_dir/gasoline"
}

CACHE_REASON=""
cache_valid() {
    local actual_defines actual_hash actual_source_inputs expected_defines
    local expected_source_inputs recorded_hash recorded_key
    local actual_make_vars expected_make_vars
    CACHE_REASON=""
    if [ ! -d "$cache_dir" ]; then
        CACHE_REASON="missing"
        return 1
    fi
    for required_file in gasoline manifest.json effective-defines.txt make-vars.txt source-inputs.txt build-command.txt build.log build-key binary.sha256; do
        if [ ! -f "$cache_dir/$required_file" ]; then
            CACHE_REASON="missing $required_file"
            return 1
        fi
    done
    if [ ! -x "$executable_path" ]; then
        CACHE_REASON="gasoline is not executable"
        return 1
    fi
    expected_defines="$(printf '%s\n' "${effective_flags[@]}")"
    actual_defines="$(<"$cache_dir/effective-defines.txt")"
    if [ "$actual_defines" != "$expected_defines" ]; then
        CACHE_REASON="effective-defines mismatch"
        return 1
    fi
    expected_make_vars="$(printf '%s\n' "${effective_make_vars[@]}")"
    actual_make_vars="$(<"$cache_dir/make-vars.txt")"
    if [ "$actual_make_vars" != "$expected_make_vars" ]; then
        CACHE_REASON="make-vars mismatch"
        return 1
    fi
    expected_source_inputs="$(printf '%s\n' "${source_input_records[@]}")"
    actual_source_inputs="$(<"$cache_dir/source-inputs.txt")"
    if [ "$actual_source_inputs" != "$expected_source_inputs" ]; then
        CACHE_REASON="source-inputs mismatch"
        return 1
    fi
    recorded_key="$(<"$cache_dir/build-key")"
    if [ "$recorded_key" != "$build_key" ]; then
        CACHE_REASON="short-key collision or mismatched build-key"
        return 1
    fi
    recorded_hash="$(<"$cache_dir/binary.sha256")"
    actual_hash="$(build_sha256_file "$executable_path")"
    if [ "$recorded_hash" != "$actual_hash" ]; then
        CACHE_REASON="binary checksum mismatch"
        return 1
    fi
    CACHE_REASON="valid"
    return 0
}

print_context() {
    local cache_state="miss" user_flags="" define source_input
    if [ -d "$cache_dir" ]; then
        if cache_valid; then cache_state="hit"; else cache_state="invalid:$CACHE_REASON"; fi
    fi
    if [ "${#normalized_user_defines[@]}" -gt 0 ]; then
        printf -v user_flags '%s ' "${normalized_user_defines[@]/#/-D}"
        user_flags="${user_flags% }"
    fi
    printf 'adapter=gasoline\n'
    printf 'physics=%s\neos=%s\nprofile=%s\n' "${physics:-profile-only}" "$eos" "$profile"
    printf 'cooling=%s\n' "$cooling"
    printf 'modules=%s\n' "${modules:-none}"
    printf 'extra_build_flags_file=%s\n' "$extra_flags_file"
    printf 'kernel=-D%s\nuser_defines=%s\n' "$kernel" "$user_flags"
    printf 'effective_defines='; printf '%s ' "${effective_flags[@]}"; printf '\n'
    printf 'make_vars='; printf '%q ' "${effective_make_vars[@]}"; printf '\n'
    printf 'source=%s\nsource_fingerprint=%s\n' "$source_dir" "$source_fingerprint"
    for source_input in "${source_input_records[@]}"; do
        printf 'source_input=%s\n' "$source_input"
    done
    printf 'target=%s\ncompiler=%s\ntoolchain_fingerprint=%s\n' "$make_target" "$compiler" "$toolchain_fingerprint"
    printf 'cpu_target=%s\ncpu=%s\ncpu_flags=%s\n' "$cpu_target" "$cpu_resolved" "$cpu_flags"
    printf 'build_key=%s\ncache_dir=%s\nexecutable=%s\n' "$build_key" "$cache_dir" "$executable_path"
    printf 'cache_state=%s\nbuild_mode=%s\n' "$cache_state" "$build_mode"
}

write_manifest() {
    local stage="$1" binary_hash="$2" source_after="$3"
    python3 - "$stage/manifest.json" "$build_key" "$short_key" "$profile" \
        "$physics" "$eos" "$kernel" "$source_dir" "$source_fingerprint" \
        "$source_after" "$make_target" "$compiler" "$toolchain_fingerprint" \
        "$config_file" "$config_fingerprint" "$extra_flags_file" "$helpers_fingerprint" \
        "$adapter_fingerprint" "$binary_hash" "$stage/effective-defines.txt" \
        "$stage/source-inputs.txt" "$cooling" "$stage/make-vars.txt" \
        "$cpu_target" "$cpu_resolved" "$cpu_flags" <<'PY'
import datetime
import json
import sys

(
    output, build_key, short_key, profile, physics, eos, kernel, source,
    source_before, source_after, target, compiler, toolchain, config_file,
    config_hash, extra_flags_file, helpers_hash, adapter_hash, binary_hash, defines_file,
    source_inputs_file, cooling, make_vars_file, cpu_target, cpu_resolved, cpu_flags,
) = sys.argv[1:]
cpu, _, cpu_options = cpu_resolved.partition(" ")
with open(defines_file, encoding="utf-8") as stream:
    defines = [line.rstrip("\n") for line in stream if line.strip()]
with open(make_vars_file, encoding="utf-8") as stream:
    make_vars = [line.rstrip("\n") for line in stream if line.strip()]
source_inputs = []
with open(source_inputs_file, encoding="utf-8") as stream:
    for line in stream:
        label, path, fingerprint = line.rstrip("\n").split("\t", 2)
        source_inputs.append({"label": label, "path": path, "fingerprint": fingerprint})
manifest = {
    "schema": 3,
    "adapter": "gasoline",
    "build_key": build_key,
    "short_key": short_key,
    "profile": profile,
    "cooling": cooling,
    "kernel": "-D" + kernel,
    "effective_defines": defines,
    "make_vars": make_vars,
    "source": {
        "path": source,
        "fingerprint_before": source_before,
        "fingerprint_after": source_after,
        "inputs": source_inputs,
    },
    "build": {
        "target": target,
        "compiler": compiler,
        "toolchain_fingerprint": toolchain,
        "config_file": config_file,
        "config_fingerprint": config_hash,
        "extra_build_flags_file": extra_flags_file,
        "helpers_fingerprint": helpers_hash,
        "adapter_fingerprint": adapter_hash,
        "cpu_target": cpu_target,
        "cpu": cpu,
        "cpu_options_fingerprint": cpu_options,
        "cpu_flags": cpu_flags,
    },
    "binary": {"path": "gasoline", "sha256": binary_hash},
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(output, "x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

write_build_inputs() {
    local stage="$1" flag
    (
        set -o noclobber
        printf '%s\n' "${effective_flags[@]}" >"$stage/effective-defines.txt"
        printf '%s\n' "${effective_make_vars[@]}" >"$stage/make-vars.txt"
        printf '%s\n' "${source_input_records[@]}" >"$stage/source-inputs.txt"
        {
            printf 'make -B -j %q %q ' "$jobs" "$make_target"
            printf '%q ' "CODE_DEF=$code_def" "${effective_make_vars[@]}" \
                "EXE=$stage/gasoline"
            printf '\n'
        } >"$stage/build-command.txt"
    )
}

resolve_profile() {
    local selected_profile="$1" stage source_after source_fingerprint_before binary_hash lock_fd
    local -a make_command

    prepare_context "$selected_profile"
    if cache_valid; then
        printf 'Gasoline cache hit: %s\n' "$cache_dir" >&2
        printf '%s\n' "$executable_path"
        return 0
    fi
    if [ -d "$cache_dir" ]; then
        gasoline_die "refusing invalid immutable cache $cache_dir ($CACHE_REASON)"
    fi
    if [ "$build_mode" = reuse-only ]; then
        printf 'Gasoline cache miss for profile=%s key=%s\n' "$profile" "$build_key" >&2
        printf 'Prebuild with: %q all' "${BASH_SOURCE[0]}" >&2
        for define in "${normalized_user_defines[@]}"; do printf ' --define %q' "$define" >&2; done
        printf ' --physics %q --kernel %q --cooling %q\n' "$physics" "$kernel" "$cooling" >&2
        return 2
    fi
    mkdir -p "$build_root"
    command -v flock >/dev/null 2>&1 || gasoline_die "flock is required to serialize legacy in-source builds"
    exec {lock_fd}>>"$build_root/.build.lock"
    flock "$lock_fd"

    # Configuration/source may have changed while another build held the lock.
    prepare_context "$selected_profile"
    if cache_valid; then
        printf 'Gasoline cache hit: %s\n' "$cache_dir" >&2
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        printf '%s\n' "$executable_path"
        return 0
    fi
    if [ -d "$cache_dir" ]; then
        gasoline_die "refusing invalid immutable cache $cache_dir ($CACHE_REASON)"
    fi
    stage="$build_root/.staging-$profile-$short_key-${BASHPID}-${RANDOM}"
    mkdir "$stage"
    write_build_inputs "$stage"
    make_command=(make -B -j "$jobs" "$make_target" "CODE_DEF=$code_def" \
        "${effective_make_vars[@]}" "EXE=$stage/gasoline")
    printf 'Building Gasoline profile=%s key=%s\n' "$profile" "$short_key" >&2
    printf 'Source: %s\n' "$source_dir" >&2
    printf 'Defines: %s\n' "${effective_flags[*]}" >&2
    printf 'Cooling: %s (%s)\n' "$cooling" "${effective_make_vars[*]}" >&2
    printf 'CPU target: %s\n' "${cpu_target:-generic}${cpu_resolved:+ (${cpu_resolved%% *})}" >&2
    if ! (cd "$source_dir" && "${make_command[@]}") 2>&1 | tee -a "$stage/build.log" >&2; then
        printf 'FATAL: Gasoline build failed; preserved diagnostics in %s\n' "$stage" >&2
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        return 1
    fi
    [ -x "$stage/gasoline" ] || gasoline_die "build succeeded but did not create executable $stage/gasoline"

    source_fingerprint_before="$source_fingerprint"
    capture_source_inputs
    source_after="$source_fingerprint"
    if [ "$source_after" != "$source_fingerprint_before" ]; then
        printf 'FATAL: Gasoline source inputs changed during the build; preserved %s\n' "$stage" >&2
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        return 1
    fi
    binary_hash="$(build_sha256_file "$stage/gasoline")"
    (
        set -o noclobber
        printf '%s\n' "$build_key" >"$stage/build-key"
        printf '%s\n' "$binary_hash" >"$stage/binary.sha256"
    )
    write_manifest "$stage" "$binary_hash" "$source_after"

    [ ! -e "$cache_dir" ] || gasoline_die "cache appeared while holding build lock: $cache_dir"
    mv "$stage" "$cache_dir"
    cache_valid || gasoline_die "new cache failed validation: $cache_dir ($CACHE_REASON)"
    printf 'Built Gasoline: %s\n' "$cache_dir" >&2
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    printf '%s\n' "$executable_path"
}

case "$command_name" in
    explain)
        selected_profile="$(requested_profile)"
        prepare_context "$selected_profile"
        print_context
        ;;
    resolve)
        selected_profile="$(requested_profile)"
        resolve_profile "$selected_profile"
        ;;
    all)
        # Each entry is a full set, so re-derive the physics view per build.
        for build_set in "${GASOLINE_BUILD_SETS[@]}"; do
            physics="$build_set"
            eos="$(build_set_eos "$physics")"
            resolve_profile "$(gasoline_profile_for "$physics")"
        done
        ;;
    list)
        if [ ! -d "$build_root" ]; then
            printf 'No Gasoline build cache: %s\n' "$build_root"
            exit 0
        fi
        shopt -s nullglob
        found=0
        for cached_build in "$build_root"/*; do
            [ -d "$cached_build" ] || continue
            [ -f "$cached_build/build-key" ] || continue
            printf '%s\n' "$cached_build"
            found=1
        done
        [ "$found" -eq 1 ] || printf 'No completed Gasoline builds in %s\n' "$build_root"
        ;;
esac
