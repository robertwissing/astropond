#!/usr/bin/env bash
# Resolve reproducible ChaNGa executables for the testsuite.
#
# runtest.sh declares semantic physics/EOS requirements.  This adapter maps
# those requirements to configure flags, fingerprints ChaNGa plus its selected
# utility and Charm++ inputs, and publishes immutable executable artifacts.
# Legacy in-source configuration/object work is serialized and never cleaned.
set -euo pipefail
umask 022

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
codes_dir="$(cd "$here/.." && pwd -P)"
common_helpers="$codes_dir/build_common.sh"
fingerprint_helper="$here/charm_fingerprint.py"
config_file="${CHANGA_BUILD_CONFIG:-$here/build_config.sh}"
extra_flags_file="${CODE_EXTRA_BUILD_FLAGS_FILE:-$codes_dir/extra_build_flags.sh}"
if [[ "$extra_flags_file" != /* ]]; then extra_flags_file="$(pwd -P)/$extra_flags_file"; fi

[ -r "$common_helpers" ] || { printf 'FATAL: missing build helpers: %s\n' "$common_helpers" >&2; exit 1; }
[ -r "$fingerprint_helper" ] || { printf 'FATAL: missing Charm fingerprint helper: %s\n' "$fingerprint_helper" >&2; exit 1; }
[ -r "$config_file" ] || { printf 'FATAL: missing ChaNGa build config: %s\n' "$config_file" >&2; exit 1; }
[ -r "$extra_flags_file" ] || { printf 'FATAL: missing extra build-flags file: %s\n' "$extra_flags_file" >&2; exit 1; }
# shellcheck source=../build_common.sh
source "$common_helpers"
# shellcheck source=build_config.sh
source "$config_file"
# shellcheck source=../extra_build_flags.sh
source "$extra_flags_file"

changa_die() {
    printf 'FATAL: %s\n' "$*" >&2
    exit 1
}

changa_usage() {
    cat <<'EOF'
Usage:
  build_changa.sh capabilities
  build_changa.sh supports --physics SET
  build_changa.sh explain  --physics SET [options]
  build_changa.sh resolve  --physics SET [options]
  build_changa.sh all [options]
  build_changa.sh list [--build-root PATH]

Requirements:
  --physics "tok tok=variant ..."                the RUNPHYSICS set: it DEFINES the build
  --eos     adiabatic|barotropic|isor            override, folded into the set

Build options:
  --configure-flag FLAG  Extra --enable/--disable flag; repeat as needed
  --source PATH          ChaNGa source tree
  --charm-source PATH    Selected, already-built Charm++ tree
  --utility-source PATH  N-BodyShop utility tree
  --build-root PATH      Immutable artifact root
  --jobs N               Parallel make jobs
  --mode auto|reuse-only Build cache misses or require a prebuilt artifact

Environment equivalents:
  CHANGA_SRC, CHANGA_BUILD_CONFIG, CHANGA_CHARM_SRC, CHANGA_UTILITY_SRC,
  CODE_EXTRA_BUILD_FLAGS_FILE, CHANGA_CONFIGURE_FLAGS (legacy/temporary override),
  CHANGA_BUILD_ROOT, CHANGA_CPU_TARGET (-march/-mtune value, "" = generic),
  CHANGA_LOCK_ROOT, CHANGA_BUILD_MODE, JOBS
  CHANGA_FINGERPRINT_CACHE (metadata cache directory), CHANGA_CHARM_REHASH=1
  (force a full Charm content check)

Private MHD-TP example:
  Select build_config_private.sh, edit CHANGA_EXTRA_BUILD_FLAGS in
  codes/extra_build_flags.sh, then run build_changa.sh explain --physics mhd
EOF
}

for required_symbol in \
    CHANGA_REFERENCE_SRC CHANGA_DEFAULT_CHARM_SOURCE \
    CHANGA_DEFAULT_UTILITY_SOURCE CHANGA_CHARM_COMPONENTS \
    CHANGA_COMMON_CONFIGURE_FLAGS CHANGA_BUILD_SETS; do
    declare -p "$required_symbol" >/dev/null 2>&1 ||
        changa_die "$config_file does not define $required_symbol"
done
declare -F changa_profile_for >/dev/null || changa_die "$config_file does not define changa_profile_for"
declare -F changa_physics_configure_flags >/dev/null ||
    changa_die "$config_file does not define changa_physics_configure_flags"
declare -F changa_print_capabilities >/dev/null ||
    changa_die "$config_file does not define changa_print_capabilities"
extra_flags_declaration="$(declare -p CHANGA_EXTRA_BUILD_FLAGS 2>/dev/null)" ||
    changa_die "$extra_flags_file does not define CHANGA_EXTRA_BUILD_FLAGS"
[[ "$extra_flags_declaration" == declare\ -a* ]] ||
    changa_die "CHANGA_EXTRA_BUILD_FLAGS in $extra_flags_file must be an indexed array"

command_name="${1:-}"
case "$command_name" in
    capabilities|supports|explain|resolve|all|list)
        shift
        ;;
    -h|--help|help|"")
        changa_usage
        exit 0
        ;;
    *)
        changa_usage >&2
        changa_die "unknown command: $command_name"
        ;;
esac

physics=""
eos_cli=""
source_dir="${CHANGA_SRC:-$CHANGA_REFERENCE_SRC}"
charm_source="${CHANGA_CHARM_SRC:-}"
utility_source="${CHANGA_UTILITY_SRC:-}"
build_root="${CHANGA_BUILD_ROOT:-$here/builds}"
lock_root="${CHANGA_LOCK_ROOT:-$here/builds/.locks}"
build_mode="${CHANGA_BUILD_MODE:-auto}"
jobs="${JOBS:-$(nproc 2>/dev/null || printf '1')}"
cpu_target="${CHANGA_CPU_TARGET:-}"
declare -a user_configure_flags=()

if [ "${#CHANGA_EXTRA_BUILD_FLAGS[@]}" -gt 0 ]; then
    user_configure_flags+=("${CHANGA_EXTRA_BUILD_FLAGS[@]}")
fi

if [ -n "${CHANGA_CONFIGURE_FLAGS:-}" ]; then
    read -r -a environment_flags <<<"$CHANGA_CONFIGURE_FLAGS"
    user_configure_flags+=("${environment_flags[@]}")
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --physics)
            [ "$#" -ge 2 ] || changa_die "--physics requires a value"
            physics="$2"; shift 2
            ;;
        --physics=*) physics="${1#*=}"; shift ;;
        --eos)
            [ "$#" -ge 2 ] || changa_die "--eos requires a value"
            eos_cli="$2"; shift 2
            ;;
        --eos=*) eos_cli="${1#*=}"; shift ;;
        --configure-flag)
            [ "$#" -ge 2 ] || changa_die "--configure-flag requires a value"
            user_configure_flags+=("$2"); shift 2
            ;;
        --configure-flag=*) user_configure_flags+=("${1#*=}"); shift ;;
        --source)
            [ "$#" -ge 2 ] || changa_die "--source requires a value"
            source_dir="$2"; shift 2
            ;;
        --source=*) source_dir="${1#*=}"; shift ;;
        --charm-source)
            [ "$#" -ge 2 ] || changa_die "--charm-source requires a value"
            charm_source="$2"; shift 2
            ;;
        --charm-source=*) charm_source="${1#*=}"; shift ;;
        --utility-source)
            [ "$#" -ge 2 ] || changa_die "--utility-source requires a value"
            utility_source="$2"; shift 2
            ;;
        --utility-source=*) utility_source="${1#*=}"; shift ;;
        --build-root)
            [ "$#" -ge 2 ] || changa_die "--build-root requires a value"
            build_root="$2"; shift 2
            ;;
        --build-root=*) build_root="${1#*=}"; shift ;;
        --jobs)
            [ "$#" -ge 2 ] || changa_die "--jobs requires a value"
            jobs="$2"; shift 2
            ;;
        --jobs=*) jobs="${1#*=}"; shift ;;
        --mode)
            [ "$#" -ge 2 ] || changa_die "--mode requires a value"
            build_mode="$2"; shift 2
            ;;
        --mode=*) build_mode="${1#*=}"; shift ;;
        -h|--help)
            changa_usage
            exit 0
            ;;
        *) changa_die "unknown option: $1" ;;
    esac
done

# --physics IS the set and defines the build. --eos stays for hand-typed
# invocations and is folded INTO it, so there is one object to resolve.
if [ -n "$eos_cli" ]; then physics="$(build_set_put eos "$eos_cli" "$physics")"; fi
eos="$(build_set_eos "$physics")"

case "$build_mode" in auto|reuse-only) ;; *) changa_die "invalid build mode: $build_mode" ;; esac
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || changa_die "--jobs must be a positive integer"
[[ "$cpu_target" =~ ^[A-Za-z0-9_.-]*$ ]] || changa_die "unsafe CHANGA_CPU_TARGET: $cpu_target"

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
if [ -z "$charm_source" ]; then charm_source="$source_dir/$CHANGA_DEFAULT_CHARM_SOURCE"; fi
if [ -z "$utility_source" ]; then utility_source="$source_dir/$CHANGA_DEFAULT_UTILITY_SOURCE"; fi
charm_source="$(absolute_path "$charm_source")"
utility_source="$(absolute_path "$utility_source")"
build_root="$(absolute_path "$build_root")"
lock_root="$(absolute_path "$lock_root")"
fingerprint_root="$(absolute_path "${CHANGA_FINGERPRINT_CACHE:-$lock_root/charm-fingerprints}")"

normalize_configure_flag() {
    local flag="$1"
    [[ "$flag" =~ ^--(enable|disable)-[A-Za-z0-9][A-Za-z0-9_-]*(=[A-Za-z0-9_./:+-]+)?$ ]] ||
        changa_die "invalid or unsafe configure flag: $flag"
    printf '%s\n' "$flag"
}

configure_feature() {
    local flag="$1" body
    body="${flag#--enable-}"
    if [ "$body" = "$flag" ]; then body="${flag#--disable-}"; fi
    printf '%s\n' "${body%%=*}"
}

declare -a normalized_user_flags=()
for configure_flag in "${user_configure_flags[@]}"; do
    normalized_user_flags+=("$(normalize_configure_flag "$configure_flag")")
done
if [ "${#normalized_user_flags[@]}" -gt 0 ]; then
    mapfile -t normalized_user_flags < <(printf '%s\n' "${normalized_user_flags[@]}" | LC_ALL=C sort -u)
fi
for configure_flag in "${normalized_user_flags[@]}"; do
    case "$(configure_feature "$configure_flag")" in
        mhd|barotropic|isor)
            changa_die "$(configure_feature "$configure_flag") is profile-owned; do not pass it as a user flag"
            ;;
    esac
done

requested_profile() {
    [ -n "$physics" ] || changa_die "$command_name requires --physics"
    local selected
    if ! selected="$(changa_profile_for "$physics")"; then
        printf 'Unsupported ChaNGa build requirement: %s\n' "$physics" >&2
        return 1
    fi
    printf '%s\n' "$selected"
}

if [ "$command_name" = capabilities ]; then
    changa_print_capabilities
    exit 0
fi
if [ "$command_name" = supports ]; then
    requested_profile
    exit $?
fi

declare -a effective_flags=()
declare -a source_input_records=()
declare -A flag_values=()
declare -A flag_origins=()

add_effective_flag() {
    local flag feature origin
    flag="$(normalize_configure_flag "$1")"
    origin="$2"
    feature="$(configure_feature "$flag")"
    if [ -n "${flag_values[$feature]+set}" ]; then
        if [ "${flag_values[$feature]}" != "$flag" ]; then
            changa_die "conflicting configure flags for $feature: ${flag_values[$feature]} (${flag_origins[$feature]}) versus $flag ($origin)"
        fi
        return
    fi
    flag_values[$feature]="$flag"
    flag_origins[$feature]="$origin"
    effective_flags+=("$flag")
}

prepare_flags() {
    local selected_profile="$1" flag profile_flag_text token
    effective_flags=()
    flag_values=()
    flag_origins=()
    for flag in "${CHANGA_COMMON_CONFIGURE_FLAGS[@]}"; do
        add_effective_flag "$flag" common
    done
    for token in $(build_set_physics "$physics"); do
        if ! profile_flag_text="$(changa_physics_configure_flags "$token")"; then
            changa_die "$config_file has no ChaNGa build for physics token: $token"
        fi
        while IFS= read -r flag; do
            [ -n "$flag" ] && add_effective_flag "$flag" "physics:$token"
        done <<<"$profile_flag_text"
    done
    for flag in "${normalized_user_flags[@]}"; do
        add_effective_flag "$flag" user
    done
}

# Include untracked source-like files, except retained diagnostics scratch builds.
# Every tracked file remains an input, including any tracked diagnostics files.
# Utility's configure-generated structures/Makefile and structures/config.h are
# the only exceptions: they describe a previous build, not source provenance.
changa_filtered_source_fingerprint() {
    local root="$1"
    shift
    local -a excluded=("$@")
    local path skip excluded_path
    [ -d "$root" ] || build_common_die "source directory does not exist: $root"
    if git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        (
            cd "$root"
            {
                git ls-files --cached -z
                git ls-files --others --exclude-standard --exclude=diagnostics/ -z |
                    while IFS= read -r -d '' path; do
                        case "$path" in
                            Makefile|Makefile.*|*/Makefile|*/Makefile.*|*.mk|*.c|*.h|*.cc|*.cpp|*.cxx|*.f|*.f90|*.inc|*.def)
                                printf '%s\0' "$path"
                                ;;
                        esac
                    done
            } | LC_ALL=C sort -zu |
                while IFS= read -r -d '' path; do
                    skip=0
                    for excluded_path in "${excluded[@]}"; do
                        [ "$path" = "$excluded_path" ] && skip=1
                    done
                    [ "$skip" -eq 0 ] || continue
                    [ -f "$path" ] || continue
                    printf '%s\0' "$path"
                    build_sha256_file "$path"
                done
        ) | build_sha256_stream
        return
    fi
    (
        cd "$root"
        find . -type d -name diagnostics -prune -o -type f \( -name 'Makefile*' -o -name '*.mk' -o \
            -name '*.c' -o -name '*.h' -o -name '*.cc' -o -name '*.cpp' -o \
            -name '*.cxx' -o -name '*.f' -o -name '*.f90' -o -name '*.inc' \) -print0 |
            LC_ALL=C sort -z |
            while IFS= read -r -d '' path; do
                path="${path#./}"
                skip=0
                for excluded_path in "${excluded[@]}"; do
                    [ "$path" = "$excluded_path" ] && skip=1
                done
                [ "$skip" -eq 0 ] || continue
                printf '%s\0' "$path"
                build_sha256_file "$path"
            done
    ) | build_sha256_stream
}

changa_complete_tree_fingerprint() {
    local root="$1"
    [ -d "$root" ] || build_common_die "Charm++ component does not exist: $root"
    local -a options=(--cache-root "$fingerprint_root")
    [ "${CHANGA_CHARM_REHASH:-0}" != 1 ] || options+=(--refresh)
    python3 "$fingerprint_helper" "$root" "${options[@]}"
}

profile=""
source_fingerprint=""
toolchain_fingerprint=""
config_fingerprint=""
helpers_fingerprint=""
adapter_fingerprint=""
build_key=""
short_key=""
cache_dir=""
executable_path=""
launcher_path=""
charmc=""
charmrun_source=""
key_material=""

capture_source_inputs() {
    local component component_path component_resolved component_fingerprint
    local changa_fingerprint utility_fingerprint
    local -A component_fingerprints=()
    source_input_records=()
    changa_fingerprint="$(changa_filtered_source_fingerprint "$source_dir")"
    source_input_records+=("changa"$'\t'"$source_dir"$'\t'"$changa_fingerprint")
    utility_fingerprint="$(changa_filtered_source_fingerprint "$utility_source" \
        structures/Makefile structures/config.h)"
    source_input_records+=("utility"$'\t'"$utility_source"$'\t'"$utility_fingerprint")
    for component in "${CHANGA_CHARM_COMPONENTS[@]}"; do
        component_path="$charm_source/$component"
        [ -e "$component_path" ] || changa_die "selected Charm++ component is missing: $component_path"
        if command -v realpath >/dev/null 2>&1; then
            component_resolved="$(realpath -e "$component_path")"
        else
            component_resolved="$component_path"
        fi
        # Charm's tmp can resolve to include: scan a physical directory only once.
        if [ -z "${component_fingerprints[$component_resolved]+set}" ]; then
            component_fingerprints[$component_resolved]="$(changa_complete_tree_fingerprint "$component_resolved")" || return
        fi
        component_fingerprint="${component_fingerprints[$component_resolved]}"
        source_input_records+=("charm:$component"$'\t'"$component_resolved"$'\t'"$component_fingerprint")
    done
    source_fingerprint="$(printf '%s\n' "${source_input_records[@]}" | build_sha256_stream)"
}

prepare_context() {
    local selected_profile="$1" flag
    [ -d "$source_dir" ] || changa_die "ChaNGa source directory does not exist: $source_dir"
    [ -x "$source_dir/configure" ] || changa_die "ChaNGa configure script is missing or not executable: $source_dir/configure"
    [ -d "$utility_source/structures" ] || changa_die "utility structures directory is missing: $utility_source/structures"
    charmc="$charm_source/bin/charmc"
    charmrun_source="$charm_source/bin/charmrun"
    [ -x "$charmc" ] || changa_die "selected Charm++ compiler is missing: $charmc"
    [ -x "$charmrun_source" ] || changa_die "selected Charm++ launcher is missing: $charmrun_source"
    profile="$selected_profile"
    prepare_flags "$profile"
    config_fingerprint="$(build_sha256_file "$config_file")"
    helpers_fingerprint="$(build_sha256_file "$common_helpers")"
    adapter_fingerprint="$({
        build_sha256_file "${BASH_SOURCE[0]}"
        build_sha256_file "$fingerprint_helper"
    } | build_sha256_stream)"
    capture_source_inputs
    toolchain_fingerprint="$(build_toolchain_fingerprint "$charmc")"
    cpu_resolved=""
    cpu_flags=""
    if [ -n "$cpu_target" ]; then
        charm_cxx="$("$charmc" -print-building-blocks | sed -n "s/^CHARM_CXX='\(.*\)'\$/\1/p")"
        [ -n "$charm_cxx" ] || changa_die "$charmc does not report CHARM_CXX"
        cpu_resolved="$(build_cpu_target "${charm_cxx%% *}" "$cpu_target")" || exit 1
        cpu_flags="-march=$cpu_target -mtune=$cpu_target"
    fi
    key_material="$({
        printf 'schema=1\nadapter=changa\nprofile=%s\n' "$profile"
        printf 'cpu_target=%s\ncpu=%s\n' "$cpu_target" "$cpu_resolved"
        printf 'config=%s\nhelpers=%s\nadapter_script=%s\n' \
            "$config_fingerprint" "$helpers_fingerprint" "$adapter_fingerprint"
        printf 'source_path=%s\ncharm_path=%s\nutility_path=%s\n' \
            "$source_dir" "$charm_source" "$utility_source"
        printf 'source=%s\ntoolchain=%s\n' "$source_fingerprint" "$toolchain_fingerprint"
        for flag in "${effective_flags[@]}"; do printf 'configure_flag=%s\n' "$flag"; done
    })"
    build_key="$(build_sha256_text "$key_material")"
    short_key="${build_key:0:12}"
    cache_dir="$build_root/$profile-$short_key"
    executable_path="$cache_dir/ChaNGa.smp"
    launcher_path="$cache_dir/charmrun"
}

CACHE_REASON=""
cache_valid() {
    local actual_flags expected_flags actual_inputs expected_inputs
    local recorded_key recorded_hash actual_hash recorded_launcher_hash actual_launcher_hash
    CACHE_REASON=""
    if [ ! -d "$cache_dir" ]; then CACHE_REASON="missing"; return 1; fi
    for required_file in ChaNGa.smp charmrun manifest.json effective-configure-flags.txt \
        source-inputs.txt configure-command.txt build-command.txt configure.log \
        build.log build-key binary.sha256 launcher.sha256; do
        if [ ! -f "$cache_dir/$required_file" ]; then
            CACHE_REASON="missing $required_file"; return 1
        fi
    done
    [ -x "$executable_path" ] || { CACHE_REASON="ChaNGa.smp is not executable"; return 1; }
    [ -x "$launcher_path" ] || { CACHE_REASON="charmrun is not executable"; return 1; }
    expected_flags="$(printf '%s\n' "${effective_flags[@]}")"
    actual_flags="$(<"$cache_dir/effective-configure-flags.txt")"
    [ "$actual_flags" = "$expected_flags" ] || { CACHE_REASON="effective flags mismatch"; return 1; }
    expected_inputs="$(printf '%s\n' "${source_input_records[@]}")"
    actual_inputs="$(<"$cache_dir/source-inputs.txt")"
    [ "$actual_inputs" = "$expected_inputs" ] || { CACHE_REASON="source inputs mismatch"; return 1; }
    recorded_key="$(<"$cache_dir/build-key")"
    [ "$recorded_key" = "$build_key" ] || { CACHE_REASON="short-key collision or mismatched build-key"; return 1; }
    recorded_hash="$(<"$cache_dir/binary.sha256")"
    actual_hash="$(build_sha256_file "$executable_path")"
    [ "$recorded_hash" = "$actual_hash" ] || { CACHE_REASON="binary checksum mismatch"; return 1; }
    recorded_launcher_hash="$(<"$cache_dir/launcher.sha256")"
    actual_launcher_hash="$(build_sha256_file "$launcher_path")"
    [ "$recorded_launcher_hash" = "$actual_launcher_hash" ] || { CACHE_REASON="launcher checksum mismatch"; return 1; }
    CACHE_REASON="valid"
    return 0
}

print_context() {
    local cache_state="miss" source_input
    if [ -d "$cache_dir" ]; then
        if cache_valid; then cache_state="hit"; else cache_state="invalid:$CACHE_REASON"; fi
    fi
    printf 'adapter=changa\n'
    printf 'physics=%s\neos=%s\nprofile=%s\n' "${physics:-profile-only}" "$eos" "$profile"
    printf 'extra_build_flags_file=%s\n' "$extra_flags_file"
    printf 'effective_configure_flags='; printf '%s ' "${effective_flags[@]}"; printf '\n'
    printf 'source=%s\ncharm_source=%s\nutility_source=%s\n' "$source_dir" "$charm_source" "$utility_source"
    printf 'source_excluded_untracked=diagnostics/\n'
    printf 'source_fingerprint=%s\n' "$source_fingerprint"
    for source_input in "${source_input_records[@]}"; do printf 'source_input=%s\n' "$source_input"; done
    printf 'compiler=%s\ntoolchain_fingerprint=%s\n' "$charmc" "$toolchain_fingerprint"
    printf 'cpu_target=%s\ncpu=%s\ncpu_flags=%s\n' "$cpu_target" "$cpu_resolved" "$cpu_flags"
    printf 'build_key=%s\ncache_dir=%s\nexecutable=%s\nlauncher=%s\n' \
        "$build_key" "$cache_dir" "$executable_path" "$launcher_path"
    printf 'cache_state=%s\nbuild_mode=%s\n' "$cache_state" "$build_mode"
}

write_build_inputs() {
    local stage="$1" stage_target="$2"
    (
        set -o noclobber
        printf '%s\n' "${effective_flags[@]}" >"$stage/effective-configure-flags.txt"
        printf '%s\n' "${source_input_records[@]}" >"$stage/source-inputs.txt"
        {
            printf 'cd %q && CHARM_DIR=%q CHARMC=%q STRUCT_DIR=%q CFLAGS=%q CXXFLAGS=%q ' \
                "$source_dir" "$charm_source" "$charmc" "$utility_source/structures" \
                "$cpu_flags" "$cpu_flags"
            printf 'CHANGA_EFFECTIVE_CONFIGURE_FLAGS=%q ./configure ' "${effective_flags[*]}"
            printf '%q ' "${effective_flags[@]}"
            printf '\n'
        } >"$stage/configure-command.txt"
        {
            printf 'cd %q && env -u CFLAGS -u CXXFLAGS CHANGA_EFFECTIVE_CONFIGURE_FLAGS=%q make -B -j %q target=%q charmrun=charmrun\n' \
                "$source_dir" "${effective_flags[*]}" "$jobs" "$stage_target"
        } >"$stage/build-command.txt"
    )
}

write_manifest() {
    local stage="$1" binary_hash="$2" launcher_hash="$3" source_after="$4"
    python3 - "$stage/manifest.json" "$build_key" "$short_key" "$profile" \
        "$physics" "$eos" "$source_dir" "$charm_source" "$utility_source" \
        "$source_fingerprint" "$source_after" "$charmc" "$toolchain_fingerprint" \
        "$config_file" "$config_fingerprint" "$extra_flags_file" "$helpers_fingerprint" \
        "$adapter_fingerprint" "$binary_hash" "$launcher_hash" \
        "$stage/effective-configure-flags.txt" "$stage/source-inputs.txt" \
        "$cpu_target" "$cpu_resolved" "$cpu_flags" <<'PY'
import datetime
import json
import sys

(
    output, build_key, short_key, profile, physics, eos, source, charm_source,
    utility_source, source_before, source_after, compiler, toolchain,
    config_file, config_hash, extra_flags_file, helpers_hash, adapter_hash, binary_hash,
    launcher_hash, flags_file, source_inputs_file, cpu_target, cpu_resolved, cpu_flags,
) = sys.argv[1:]
cpu, _, cpu_options = cpu_resolved.partition(" ")
with open(flags_file, encoding="utf-8") as stream:
    flags = [line.rstrip("\n") for line in stream if line.strip()]
source_inputs = []
with open(source_inputs_file, encoding="utf-8") as stream:
    for line in stream:
        label, path, fingerprint = line.rstrip("\n").split("\t", 2)
        source_inputs.append({"label": label, "path": path, "fingerprint": fingerprint})
manifest = {
    "schema": 1,
    "adapter": "changa",
    "build_key": build_key,
    "short_key": short_key,
    "profile": profile,
    "effective_configure_flags": flags,
    "source": {
        "path": source,
        "charm_path": charm_source,
        "utility_path": utility_source,
        "fingerprint_before": source_before,
        "fingerprint_after": source_after,
        "inputs": source_inputs,
    },
    "build": {
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
    "binary": {"path": "ChaNGa.smp", "sha256": binary_hash},
    "launcher": {"path": "charmrun", "sha256": launcher_hash},
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(output, "x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

relative_path_from_source() {
    local destination="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath --relative-to="$source_dir" "$destination"
    else
        python3 - "$source_dir" "$destination" <<'PY'
import os
import sys
print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
    fi
}

resolve_profile() {
    local selected_profile="$1" stage stage_target source_before source_after
    local binary_hash launcher_hash lock_fd source_lock_key
    local -a configure_command make_command

    prepare_context "$selected_profile"
    if cache_valid; then
        printf 'ChaNGa cache hit: %s\n' "$cache_dir" >&2
        printf '%s\n' "$executable_path"
        return 0
    fi
    if [ -d "$cache_dir" ]; then
        changa_die "refusing invalid immutable cache $cache_dir ($CACHE_REASON)"
    fi
    if [ "$build_mode" = reuse-only ]; then
        printf 'ChaNGa cache miss for profile=%s key=%s\n' "$profile" "$build_key" >&2
        printf 'Prebuild with: CHANGA_BUILD_CONFIG=%q %q all\n' "$config_file" "${BASH_SOURCE[0]}" >&2
        return 2
    fi

    mkdir -p "$build_root" "$lock_root"
    command -v flock >/dev/null 2>&1 || changa_die "flock is required to serialize legacy in-source builds"
    source_lock_key="$(build_sha256_text "$source_dir|$charm_source|$utility_source")"
    exec {lock_fd}>>"$lock_root/source-${source_lock_key:0:20}.lock"
    flock "$lock_fd"

    # The source or profile config may have changed while this process waited.
    prepare_context "$selected_profile"
    if cache_valid; then
        printf 'ChaNGa cache hit: %s\n' "$cache_dir" >&2
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        printf '%s\n' "$executable_path"
        return 0
    fi
    if [ -d "$cache_dir" ]; then
        changa_die "refusing invalid immutable cache $cache_dir ($CACHE_REASON)"
    fi

    stage="$build_root/.staging-$profile-$short_key-${BASHPID}-${RANDOM}"
    mkdir "$stage"
    stage_target="$(relative_path_from_source "$stage/ChaNGa")"
    write_build_inputs "$stage" "$stage_target"
    configure_command=("$source_dir/configure" "${effective_flags[@]}")
    make_command=(make -B -j "$jobs" "target=$stage_target" charmrun=charmrun)
    printf 'Building ChaNGa profile=%s key=%s\n' "$profile" "$short_key" >&2
    printf 'Source: %s\n' "$source_dir" >&2
    printf 'Configure flags: %s\n' "${effective_flags[*]}" >&2
    printf 'CPU target: %s\n' "${cpu_target:-generic}${cpu_resolved:+ (${cpu_resolved%% *})}" >&2
    # configure bakes CFLAGS/CXXFLAGS into the Makefile; set them explicitly so
    # the shell's own values never reach an unkeyed build, and hide them from make.
    if ! (
        cd "$source_dir"
        CHARM_DIR="$charm_source" CHARMC="$charmc" STRUCT_DIR="$utility_source/structures" \
            CFLAGS="$cpu_flags" CXXFLAGS="$cpu_flags" \
            CHANGA_EFFECTIVE_CONFIGURE_FLAGS="${effective_flags[*]}" \
            "${configure_command[@]}"
    ) 2>&1 | tee -a "$stage/configure.log" >&2; then
        printf 'FATAL: ChaNGa configure failed; preserved diagnostics in %s\n' "$stage" >&2
        flock -u "$lock_fd"; exec {lock_fd}>&-
        return 1
    fi
    if ! (
        cd "$source_dir"
        unset CFLAGS CXXFLAGS
        CHANGA_EFFECTIVE_CONFIGURE_FLAGS="${effective_flags[*]}" "${make_command[@]}"
    ) 2>&1 | tee -a "$stage/build.log" >&2; then
        printf 'FATAL: ChaNGa build failed; preserved diagnostics in %s\n' "$stage" >&2
        flock -u "$lock_fd"; exec {lock_fd}>&-
        return 1
    fi
    [ -x "$stage/ChaNGa" ] || changa_die "build succeeded but did not create executable $stage/ChaNGa"
    mv "$stage/ChaNGa" "$stage/ChaNGa.smp"
    cp "$charmrun_source" "$stage/charmrun"
    chmod 0755 "$stage/ChaNGa.smp" "$stage/charmrun"

    source_before="$source_fingerprint"
    capture_source_inputs
    source_after="$source_fingerprint"
    if [ "$source_after" != "$source_before" ]; then
        printf 'FATAL: ChaNGa source inputs changed during the build; preserved %s\n' "$stage" >&2
        flock -u "$lock_fd"; exec {lock_fd}>&-
        return 1
    fi
    binary_hash="$(build_sha256_file "$stage/ChaNGa.smp")"
    launcher_hash="$(build_sha256_file "$stage/charmrun")"
    (
        set -o noclobber
        printf '%s\n' "$build_key" >"$stage/build-key"
        printf '%s\n' "$binary_hash" >"$stage/binary.sha256"
        printf '%s\n' "$launcher_hash" >"$stage/launcher.sha256"
    )
    write_manifest "$stage" "$binary_hash" "$launcher_hash" "$source_after"
    [ ! -e "$cache_dir" ] || changa_die "cache appeared while holding build lock: $cache_dir"
    mv "$stage" "$cache_dir"
    cache_valid || changa_die "new cache failed validation: $cache_dir ($CACHE_REASON)"
    printf 'Built ChaNGa: %s\n' "$cache_dir" >&2
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
        for build_set in "${CHANGA_BUILD_SETS[@]}"; do
            physics="$build_set"
            eos="$(build_set_eos "$physics")"
            resolve_profile "$(changa_profile_for "$physics")"
        done
        ;;
    list)
        if [ ! -d "$build_root" ]; then
            printf 'No ChaNGa build cache: %s\n' "$build_root"
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
        [ "$found" -eq 1 ] || printf 'No completed ChaNGa builds in %s\n' "$build_root"
        ;;
esac
