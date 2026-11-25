#!/usr/bin/env bash
# Non-destructive integration checks for the Gasoline build resolver.
# The unique /tmp tree is intentionally retained for inspection.
set -euo pipefail
set -o noclobber

test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
gasoline_dir="$(cd "$test_dir/.." && pwd -P)"
repo_root="$(cd "$gasoline_dir/../.." && pwd -P)"
builder="$gasoline_dir/build_gasoline.sh"
aliaser="$repo_root/codes/build_alias.sh"
fixture="$test_dir/fixture"
artifact_root="$(mktemp -d /tmp/gasoline-resolver-test.XXXXXXXX)"

export GASOLINE_BUILD_CONFIG="$fixture/build_config.sh"
export GASOLINE_SRC="$fixture/source"
export GASOLINE_DIR="$GASOLINE_SRC"   # what runtest.sh hands the builder
export GASOLINE_BUILD_ROOT="$artifact_root/builds"
export GASOLINE_BUILD_MODE=auto
export CODE_EXTRA_BUILD_FLAGS_FILE="$fixture/empty_build_flags.sh"

fail() {
    printf 'FAIL: %s\nArtifacts retained at %s\n' "$*" "$artifact_root" >&2
    exit 1
}

field() {
    local name="$1"
    awk -F= -v wanted="$name" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}'
}

hydro_explain="$("$builder" explain --physics hydro)"
grav_explain="$("$builder" explain --physics grav)"
mhd_explain="$("$builder" explain --physics mhd)"
grav_mhd_explain="$("$builder" explain --physics "grav mhd")"
collapse_explain="$("$builder" explain --physics "grav mhd eos=barotropic")"

hydro_key="$(field build_key <<<"$hydro_explain")"
grav_key="$(field build_key <<<"$grav_explain")"
mhd_key="$(field build_key <<<"$mhd_explain")"
grav_mhd_key="$(field build_key <<<"$grav_mhd_explain")"
collapse_key="$(field build_key <<<"$collapse_explain")"
[ "$hydro_key" = "$grav_key" ] || fail "hydro and gravity did not share a concrete build"
# grav contributes no define, so "mhd" and "grav mhd" are the same build.
[ "$mhd_key" = "$grav_mhd_key" ] || fail "mhd and 'grav mhd' did not share a concrete build"
[ "$hydro_key" != "$mhd_key" ] || fail "hydro and mhd keys unexpectedly match"
[ "$mhd_key" != "$collapse_key" ] || fail "barotropic collapse did not get a distinct key"

extras_ab="$("$builder" explain --physics mhd --define RECONSTRUCTION --define RECONB | field build_key)"
extras_ba="$("$builder" explain --physics mhd --define RECONB --define RECONSTRUCTION | field build_key)"
[ "$extras_ab" = "$extras_ba" ] || fail "user define order changed the build key"
extras_single="$("$builder" explain --physics mhd --define RECONB | field build_key)"
extras_duplicate="$("$builder" explain --physics mhd --define RECONB --define RECONB | field build_key)"
[ "$extras_single" = "$extras_duplicate" ] || fail "an identical duplicate define changed the build key"

file_extra_explain="$(CODE_EXTRA_BUILD_FLAGS_FILE="$fixture/extra_build_flags.sh" \
    "$builder" explain --physics mhd)"
file_extra_key="$(field build_key <<<"$file_extra_explain")"
cli_extra_key="$("$builder" explain --physics mhd --define EXTRA_VALUE=7 | field build_key)"
[ "$file_extra_key" = "$cli_extra_key" ] ||
    fail "file and CLI forms of the same Gasoline extra produced different keys"
[[ "$file_extra_explain" == *"user_defines=-DEXTRA_VALUE=7"* ]] ||
    fail "GASOLINE_EXTRA_BUILD_FLAGS was not included in explain output"

kernel_key="$("$builder" explain --physics hydro --kernel WENDLAND | field build_key)"
[ "$kernel_key" != "$hydro_key" ] || fail "kernel selection did not change the build key"
cc_path="$(command -v cc)" || fail "fixture compiler cc is unavailable"
compiler_key="$(GASOLINE_TEST_COMPILER="$cc_path" "$builder" explain --physics hydro | field build_key)"
[ "$compiler_key" != "$hydro_key" ] || fail "compiler identity did not change the build key"

# Physics is additive now, so hydro+barotropic is a build in its own right:
# BAROTROPIC is an equation of state and carries no dependence on MHD.
[ "$("$builder" supports --physics hydro --eos barotropic)" = "hydro_barotropic-metal" ] ||
    fail "hydro+barotropic did not resolve to the hydro_barotropic build"
if "$builder" explain --physics mhd --define MHD; then
    fail "profile-owned MHD was accepted as a user define"
fi
if "$builder" explain --physics hydro --kernel FLAG=1 --define FLAG=2; then
    fail "conflicting macro values were accepted"
fi
if "$builder" explain --physics hydro --define FLAG=1 --define FLAG=2; then
    fail "conflicting repeated user defines were accepted"
fi

if reuse_only_output="$(GASOLINE_BUILD_MODE=reuse-only "$builder" resolve \
        --physics hydro --define REUSE_ONLY_MISS \
        --build-root "$artifact_root/reuse-only-miss" 2>&1)"; then
    fail "reuse-only mode built a cache miss"
else
    reuse_only_status=$?
fi
[ "$reuse_only_status" -eq 2 ] || fail "reuse-only cache miss returned status $reuse_only_status instead of 2"
[[ "$reuse_only_output" == *"Gasoline cache miss"* ]] || fail "reuse-only miss did not explain the failure"
[ ! -e "$artifact_root/reuse-only-miss" ] || fail "reuse-only miss created a build root"

hydro_binary="$("$builder" resolve --physics hydro)"
[ -x "$hydro_binary" ] || fail "hydro resolve did not produce an executable"
hydro_again="$("$builder" resolve --physics grav)"
[ "$hydro_binary" = "$hydro_again" ] || fail "exact cache hit returned another path"
[[ "$("$hydro_binary")" == *"mhd=0 barotropic=0"* ]] || fail "hydro fixture flags are wrong"

mhd_binary="$("$builder" resolve --physics mhd --define EXTRA_VALUE=7)"
[[ "$("$mhd_binary")" == *"mhd=1 barotropic=0 extra=7"* ]] || fail "mhd/user fixture flags are wrong"
file_mhd_binary="$(CODE_EXTRA_BUILD_FLAGS_FILE="$fixture/extra_build_flags.sh" \
    "$builder" resolve --physics mhd)"
[ "$file_mhd_binary" = "$mhd_binary" ] ||
    fail "file-based Gasoline extra did not reuse the equivalent CLI artifact"
plain_mhd_binary="$("$builder" resolve --physics mhd)"
[[ "$("$plain_mhd_binary")" == *"mhd=1 barotropic=0"* ]] || fail "plain mhd fixture flags are wrong"

collapse_binary="$("$builder" resolve --physics "grav mhd eos=barotropic")"
[[ "$("$collapse_binary")" == *"mhd=1 barotropic=1"* ]] || fail "collapse fixture flags are wrong"

# Exercise runtest.sh from a retained temporary root. Symlinking the script and
# codes/ makes its relative adapter lookup work while keeping any test_cases/
# directories made before resolve-only exit out of the repository.
runtest_root="$artifact_root/runtest-root"
mkdir "$runtest_root"
ln -s "$repo_root/runtest.sh" "$runtest_root/runtest.sh"
ln -s "$repo_root/codes" "$runtest_root/codes"

check_runtest_resolution() {
    local label="$1" expected_set="$2" expected_key="$3" output
    local expected_alias alias_mapping
    shift 3
    if ! output="$(
        cd "$runtest_root"
        RUNTEST_RESOLVE_ONLY=1 GASOLINE_BUILD_MODE=reuse-only \
            bash ./runtest.sh "$@" 2>&1
    )"; then
        printf '%s\n' "$output" >&2
        fail "runtest resolve-only failed for $label"
    fi
    # RUNPHYSICS is a SET now, and it carries the eos axis: assert the whole
    # declaration rather than the flattened physics/eos pair it replaced.
    [[ "$output" == *"RUNPHYSICS=\"$expected_set\""* ]] ||
        fail "runtest requirements were wrong for $label"
    [[ "$output" == *"Gasoline build key: $expected_key"* ]] ||
        fail "runtest selected the wrong build for $label"
    alias_mapping="$GASOLINE_BUILD_ROOT/build-aliases.tsv"
    [ -f "$alias_mapping" ] || fail "runtest did not create the Gasoline build-alias map"
    expected_alias="$(awk -F '\t' -v key="$expected_key" '!/^#/ && $4 == key {print $1}' "$alias_mapping")"
    [[ "$expected_alias" =~ ^[A-Z]+$ ]] || fail "runtest did not map the build key for $label"
    [[ "$output" == *"Gasoline build alias: $expected_alias"* ]] ||
        fail "runtest did not report the build alias for $label"
    [[ "$output" == *"Gasoline resolve-only run name:"*"GASOLINE-$expected_alias"* ]] ||
        fail "runtest run name lacks the build alias for $label"
    [[ "$output" != *"GASOLINEcfg${expected_key:0:12}"* ]] ||
        fail "runtest retained the long build hash in the run name for $label"
    LAST_RUNTEST_OUTPUT="$output"
}

check_runtest_resolution sedov "hydro" "$hydro_key" sedov 64 16 3 0
check_runtest_resolution orzag "mhd" "$mhd_key" orzag 64 16 3 0
check_runtest_resolution cosmowave-grav+mhd "grav mhd" "$mhd_key" \
    -a 3 -b 1 -c 1 cosmowave 64 16 3 0
check_runtest_resolution mhdcollapse "grav mhd eos=barotropic" "$collapse_key" \
    mhdcollapse 64 16 3 0
check_runtest_resolution powerlawring-short-name "grav" "$hydro_key" \
    -a -0.5 -b 2 -c 0 -d 1000000 -e 0.05 -f 0 -g 0 -h 0.001 -i 0 \
    powerlawring 84 256 2 0
[[ "$LAST_RUNTEST_OUTPUT" == *"pb1e6"* ]] ||
    fail "powerlawring did not compact plasma beta to pb1e6"
[[ "$LAST_RUNTEST_OUTPUT" != *"pb1000000"* ]] ||
    fail "powerlawring retained the expanded plasma-beta tag"
[[ "$LAST_RUNTEST_OUTPUT" == *"Gasoline output-prefix length:"*"/247"* ]] ||
    fail "powerlawring did not pass the Gasoline filename preflight"

alias_mapping="$GASOLINE_BUILD_ROOT/build-aliases.tsv"
[ "$(awk -F '\t' '!/^#/ && NF {count++} END {print count+0}' "$alias_mapping")" -eq 3 ] ||
    fail "Gasoline alias mapping did not reuse the shared hydro/MHD builds"
[ "$("$aliaser" register "$(dirname "$hydro_binary")")" = A ] ||
    fail "Gasoline alias registration was not stable on reuse"
[[ "$("$aliaser" list "$GASOLINE_BUILD_ROOT")" == *$'A\tcfg'"${hydro_key:0:12}"* ]] ||
    fail "Gasoline alias list does not expose the original cache tag"

campaign_paths="$("$builder" all --define CAMPAIGN)"
[ "$(printf '%s\n' "$campaign_paths" | awk 'NF {count++} END {print count+0}')" -eq 3 ] ||
    fail "all did not build the three concrete campaign profiles"
while IFS= read -r campaign_binary; do
    [ -x "$campaign_binary" ] || fail "all returned a non-executable campaign artifact"
done <<<"$campaign_paths"

python3 - "$hydro_binary" <<'PY'
import hashlib
import json
import pathlib
import sys

binary = pathlib.Path(sys.argv[1])
manifest = json.loads((binary.parent / "manifest.json").read_text(encoding="utf-8"))
assert manifest["profile"] == "hydro-metal"
assert manifest["binary"]["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
assert "-DGASOLINE" in manifest["effective_defines"]
assert len(manifest["source"]["inputs"]) == 2
assert manifest["build"]["extra_build_flags_file"].endswith(
    "codes/gasoline/tests/fixture/empty_build_flags.sh"
)
PY

source_copy="$artifact_root/source-copy"
mkdir "$source_copy"
cp "$fixture/source/Makefile" "$source_copy/Makefile"
cp "$fixture/source/fake.c" "$source_copy/fake.c"
copy_key_before="$("$builder" explain --physics hydro --source "$source_copy" | field build_key)"
printf '\n/* source fingerprint mutation */\n' >>"$source_copy/fake.c"
copy_key_after="$("$builder" explain --physics hydro --source "$source_copy" | field build_key)"
[ "$copy_key_before" != "$copy_key_after" ] || fail "source edit did not change the build key"

secondary_copy="$artifact_root/secondary-copy"
mkdir "$secondary_copy"
cp "$fixture/secondary/marker.c" "$secondary_copy/marker.c"
secondary_key_before="$(GASOLINE_TEST_SECONDARY="$secondary_copy" "$builder" explain --physics hydro | field build_key)"
printf '\n/* secondary fingerprint mutation */\n' >>"$secondary_copy/marker.c"
secondary_key_after="$(GASOLINE_TEST_SECONDARY="$secondary_copy" "$builder" explain --physics hydro | field build_key)"
[ "$secondary_key_before" != "$secondary_key_after" ] ||
    fail "secondary source-input edit did not change the build key"

concurrent_root="$artifact_root/concurrent-builds"
"$builder" resolve --physics mhd --define CONCURRENT --build-root "$concurrent_root" \
    >"$artifact_root/concurrent-a.out" 2>"$artifact_root/concurrent-a.err" &
pid_a=$!
"$builder" resolve --physics mhd --define CONCURRENT --build-root "$concurrent_root" \
    >"$artifact_root/concurrent-b.out" 2>"$artifact_root/concurrent-b.err" &
pid_b=$!
wait "$pid_a" || fail "first concurrent resolver failed"
wait "$pid_b" || fail "second concurrent resolver failed"
cmp -s "$artifact_root/concurrent-a.out" "$artifact_root/concurrent-b.out" ||
    fail "concurrent resolvers returned different artifacts"

printf 'PASS: Gasoline resolver mapping, aliases, cache, provenance, source, lock, and runtest handoff checks\n'
printf 'Artifacts retained at %s\n' "$artifact_root"
