#!/usr/bin/env bash
# Non-destructive integration checks for the ChaNGa build resolver.
# The unique /tmp tree is intentionally retained for inspection.
set -euo pipefail
set -o noclobber

test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
changa_dir="$(cd "$test_dir/.." && pwd -P)"
repo_root="$(cd "$changa_dir/../.." && pwd -P)"
builder="$changa_dir/build_changa.sh"
fixture="$test_dir/fixture"
artifact_root="$(mktemp -d /tmp/changa-resolver-test.XXXXXXXX)"
# Isolate fixture builds from the live campaign's configure additions.
mkdir "$artifact_root/codes"
printf 'CHANGA_EXTRA_BUILD_FLAGS=()\n' >"$artifact_root/codes/extra_build_flags.sh"
export CODE_EXTRA_BUILD_FLAGS_FILE="$artifact_root/codes/extra_build_flags.sh"

export CHANGA_BUILD_CONFIG="$fixture/build_config.sh"
export CHANGA_SRC="$fixture/source"
export CHANGA_DIR="$CHANGA_SRC"   # what runtest.sh hands the builder
export CHANGA_CHARM_SRC="$fixture/charm"
export CHANGA_UTILITY_SRC="$fixture/utility"
export CHANGA_BUILD_ROOT="$artifact_root/builds"
export CHANGA_LOCK_ROOT="$artifact_root/locks"
export CHANGA_BUILD_MODE=auto

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

flags_ab="$("$builder" explain --physics mhd \
    --configure-flag --enable-extra-a --configure-flag --enable-extra-b | field build_key)"
flags_ba="$("$builder" explain --physics mhd \
    --configure-flag --enable-extra-b --configure-flag --enable-extra-a | field build_key)"
[ "$flags_ab" = "$flags_ba" ] || fail "user configure-flag order changed the build key"
flag_single="$("$builder" explain --physics mhd --configure-flag --enable-extra-a | field build_key)"
flag_duplicate="$("$builder" explain --physics mhd \
    --configure-flag --enable-extra-a --configure-flag --enable-extra-a | field build_key)"
[ "$flag_single" = "$flag_duplicate" ] || fail "duplicate configure flag changed the build key"

file_extra_explain="$(CODE_EXTRA_BUILD_FLAGS_FILE="$fixture/extra_build_flags.sh" \
    "$builder" explain --physics mhd)"
file_extra_key="$(field build_key <<<"$file_extra_explain")"
cli_extra_key="$("$builder" explain --physics mhd \
    --configure-flag --enable-fixture-extra | field build_key)"
[ "$file_extra_key" = "$cli_extra_key" ] ||
    fail "file and CLI forms of the same ChaNGa extra produced different keys"
[[ "$file_extra_explain" == *"--enable-fixture-extra"* ]] ||
    fail "CHANGA_EXTRA_BUILD_FLAGS was not included in explain output"

if CHANGA_BUILD_CONFIG="$changa_dir/build_config.sh" "$builder" supports --physics mhd; then
    fail "public ChaNGa profile accepted MHD"
fi
# Physics is additive now, so hydro+barotropic is a build in its own right:
# BAROTROPIC is an equation of state and carries no dependence on MHD.
[ "$("$builder" supports --physics hydro --eos barotropic)" = "hydro_barotropic" ] ||
    fail "hydro+barotropic did not resolve to the hydro_barotropic build"
if "$builder" explain --physics mhd --configure-flag --enable-mhd; then
    fail "profile-owned MHD flag was accepted as a user extra"
fi
if "$builder" explain --physics hydro --configure-flag --disable-fixture-base; then
    fail "conflicting common/user configure flags were accepted"
fi
if "$builder" explain --physics hydro --configure-flag --with-unsafe; then
    fail "unsupported configure option form was accepted"
fi

if reuse_only_output="$(CHANGA_BUILD_MODE=reuse-only "$builder" resolve \
        --physics hydro --configure-flag --enable-reuse-only-miss \
        --build-root "$artifact_root/reuse-only-miss" 2>&1)"; then
    fail "reuse-only mode built a cache miss"
else
    reuse_only_status=$?
fi
[ "$reuse_only_status" -eq 2 ] || fail "reuse-only miss returned status $reuse_only_status instead of 2"
[[ "$reuse_only_output" == *"ChaNGa cache miss"* ]] || fail "reuse-only miss did not explain the failure"
[ ! -e "$artifact_root/reuse-only-miss" ] || fail "reuse-only miss created a build root"

hydro_binary="$("$builder" resolve --physics hydro)"
[ -x "$hydro_binary" ] || fail "hydro resolve did not produce an executable"
hydro_again="$("$builder" resolve --physics grav)"
[ "$hydro_binary" = "$hydro_again" ] || fail "exact cache hit returned another path"
[[ "$("$hydro_binary")" == *"mhd=0 barotropic=0 extra=0"* ]] || fail "hydro fixture flags are wrong"

mhd_binary="$("$builder" resolve --physics mhd --configure-flag --enable-fixture-extra)"
[[ "$("$mhd_binary")" == *"mhd=1 barotropic=0 extra=1"* ]] || fail "mhd/user fixture flags are wrong"
file_mhd_binary="$(CODE_EXTRA_BUILD_FLAGS_FILE="$fixture/extra_build_flags.sh" \
    "$builder" resolve --physics mhd)"
[ "$file_mhd_binary" = "$mhd_binary" ] ||
    fail "file-based ChaNGa extra did not reuse the equivalent CLI artifact"
plain_mhd_binary="$("$builder" resolve --physics mhd)"
[[ "$("$plain_mhd_binary")" == *"mhd=1 barotropic=0 extra=0"* ]] || fail "plain mhd fixture flags are wrong"

collapse_binary="$("$builder" resolve --physics "grav mhd eos=barotropic")"
[[ "$("$collapse_binary")" == *"mhd=1 barotropic=1 extra=0"* ]] || fail "collapse fixture flags are wrong"

campaign_paths="$("$builder" all --configure-flag --enable-campaign)"
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
launcher = binary.parent / "charmrun"
assert manifest["adapter"] == "changa"
assert manifest["profile"] == "hydro"
assert manifest["binary"]["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
assert manifest["launcher"]["sha256"] == hashlib.sha256(launcher.read_bytes()).hexdigest()
assert "--enable-fixture-base" in manifest["effective_configure_flags"]
assert len(manifest["source"]["inputs"]) == 6
assert manifest["build"]["extra_build_flags_file"].endswith("codes/extra_build_flags.sh")
PY

source_copy="$artifact_root/source-copy"
mkdir "$source_copy"
cp "$fixture/source/configure" "$source_copy/configure"
cp "$fixture/source/Makefile" "$source_copy/Makefile"
cp "$fixture/source/fake.c" "$source_copy/fake.c"
chmod 0755 "$source_copy/configure"
source_key_before="$("$builder" explain --physics hydro --source "$source_copy" \
    --charm-source "$fixture/charm" --utility-source "$fixture/utility" | field build_key)"
printf '\n/* source fingerprint mutation */\n' >>"$source_copy/fake.c"
source_key_after="$("$builder" explain --physics hydro --source "$source_copy" \
    --charm-source "$fixture/charm" --utility-source "$fixture/utility" | field build_key)"
[ "$source_key_before" != "$source_key_after" ] || fail "ChaNGa source edit did not change the build key"

# Retained diagnostic copies are not compiler inputs; live untracked sources are.
mkdir "$source_copy/diagnostics"
printf '/* retained scratch source */\n' >"$source_copy/diagnostics/scratch.cpp"
scratch_key="$("$builder" explain --physics hydro --source "$source_copy" | field build_key)"
[ "$scratch_key" = "$source_key_after" ] || fail "untracked diagnostic copy changed the build key"
printf '\n/* scratch-only edit */\n' >>"$source_copy/diagnostics/scratch.cpp"
scratch_key_after="$("$builder" explain --physics hydro --source "$source_copy" | field build_key)"
[ "$scratch_key" = "$scratch_key_after" ] || fail "editing an untracked diagnostic copy changed the build key"
printf '/* new live source */\n' >"$source_copy/new.cpp"
live_key="$("$builder" explain --physics hydro --source "$source_copy" | field build_key)"
[ "$live_key" != "$scratch_key" ] || fail "new untracked live source did not change the build key"

# If diagnostics ever contains a tracked input, it must still be checked.
git -C "$source_copy" init -q
git -C "$source_copy" add -- diagnostics/scratch.cpp
tracked_key_before="$("$builder" explain --physics hydro --source "$source_copy" | field build_key)"
printf '\n/* tracked diagnostic input edit */\n' >>"$source_copy/diagnostics/scratch.cpp"
tracked_key_after="$("$builder" explain --physics hydro --source "$source_copy" | field build_key)"
[ "$tracked_key_before" != "$tracked_key_after" ] || fail "tracked diagnostic edit did not change the build key"

utility_copy="$artifact_root/utility-copy"
mkdir "$utility_copy" "$utility_copy/structures"
cp "$fixture/utility/structures/marker.c" "$utility_copy/structures/marker.c"
utility_key_before="$("$builder" explain --physics hydro --utility-source "$utility_copy" | field build_key)"
printf '\n/* utility fingerprint mutation */\n' >>"$utility_copy/structures/marker.c"
utility_key_after="$("$builder" explain --physics hydro --utility-source "$utility_copy" | field build_key)"
[ "$utility_key_before" != "$utility_key_after" ] || fail "utility source edit did not change the build key"

charm_copy="$artifact_root/charm-copy"
mkdir "$charm_copy"
cp -R "$fixture/charm/." "$charm_copy"
chmod 0755 "$charm_copy/bin/charmc" "$charm_copy/bin/charmrun"
charm_key_before="$("$builder" explain --physics hydro --charm-source "$charm_copy" | field build_key)"
printf '\n/* Charm fingerprint mutation */\n' >>"$charm_copy/include/marker.h"
charm_key_after="$("$builder" explain --physics hydro --charm-source "$charm_copy" | field build_key)"
[ "$charm_key_before" != "$charm_key_after" ] || fail "selected Charm++ edit did not change the build key"

# Exercise runtest.sh from a retained temporary root. Resolve-only exits before
# IC creation or simulation launch, so this verifies the semantic handoff only.
runtest_root="$artifact_root/runtest-root"
mkdir "$runtest_root"
ln -s "$repo_root/runtest.sh" "$runtest_root/runtest.sh"
ln -s "$repo_root/codes" "$runtest_root/codes"

check_runtest_resolution() {
    local label="$1" expected_set="$2" expected_key="$3" output expected_alias
    shift 3
    if ! output="$(
        cd "$runtest_root"
        RUNTEST_RESOLVE_ONLY=1 CHANGA_BUILD_MODE=reuse-only \
            bash ./runtest.sh "$@" 2>&1
    )"; then
        printf '%s\n' "$output" >&2
        fail "runtest resolve-only failed for $label"
    fi
    # RUNPHYSICS is a SET now, and it carries the eos axis: assert the whole
    # declaration rather than the flattened physics/eos pair it replaced.
    [[ "$output" == *"RUNPHYSICS=\"$expected_set\""* ]] ||
        fail "runtest requirements were wrong for $label"
    [[ "$output" == *"ChaNGa build key: $expected_key"* ]] ||
        fail "runtest selected the wrong build for $label"
    expected_alias="$(awk -F '\t' -v key="$expected_key" '!/^#/ && $4 == key {print $1}' \
        "$CHANGA_BUILD_ROOT/build-aliases.tsv")"
    [[ "$expected_alias" =~ ^[A-Z]+$ ]] || fail "runtest did not map the build key for $label"
    [[ "$output" == *"ChaNGa resolve-only run name:"*"CHANGA-$expected_alias"* ]] ||
        fail "runtest run name lacks the build alias for $label"
}

check_runtest_resolution sedov "hydro" "$hydro_key" sedov 64 16 3 1
check_runtest_resolution orzag "mhd" "$mhd_key" orzag 64 16 3 1
check_runtest_resolution cosmowave-grav+mhd "grav mhd" "$mhd_key" \
    -a 3 -b 1 -c 1 cosmowave 64 16 3 1
check_runtest_resolution mhdcollapse "grav mhd eos=barotropic" "$collapse_key" \
    mhdcollapse 64 16 3 1

concurrent_root="$artifact_root/concurrent-builds"
"$builder" resolve --physics mhd --configure-flag --enable-concurrent \
    --build-root "$concurrent_root" >"$artifact_root/concurrent-a.out" 2>"$artifact_root/concurrent-a.err" &
pid_a=$!
"$builder" resolve --physics mhd --configure-flag --enable-concurrent \
    --build-root "$concurrent_root" >"$artifact_root/concurrent-b.out" 2>"$artifact_root/concurrent-b.err" &
pid_b=$!
wait "$pid_a" || fail "first concurrent resolver failed"
wait "$pid_b" || fail "second concurrent resolver failed"
cmp -s "$artifact_root/concurrent-a.out" "$artifact_root/concurrent-b.out" ||
    fail "concurrent resolvers returned different artifacts"

printf 'PASS: ChaNGa resolver mapping, file extras, cache, provenance, dependencies, lock, and runtest handoff checks\n'
printf 'Artifacts retained at %s\n' "$artifact_root"
