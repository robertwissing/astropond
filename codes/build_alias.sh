#!/usr/bin/env bash
# Assign short, stable display aliases to immutable managed build artifacts.
#
# The full content key remains authoritative in each artifact and run manifest.
# This helper only maintains an append-only human-facing map in the build root
# that holds the artifact (<build_root>/<profile>-<short_key>), so run names can
# use <CODE>-A, <CODE>-B, ... without losing provenance. Every code keeps its own
# map, because every code has its own build root.
set -euo pipefail
umask 022

alias_die() {
    printf 'FATAL: %s\n' "$*" >&2
    exit 1
}

alias_usage() {
    cat <<'EOF'
Usage:
  build_alias.sh register ARTIFACT_DIRECTORY
  build_alias.sh list BUILD_ROOT
  build_alias.sh path BUILD_ROOT

`register` verifies ARTIFACT_DIRECTORY/build-key, then returns its stable
letter alias from the append-only map BUILD_ROOT/build-aliases.tsv, where
BUILD_ROOT is the directory containing ARTIFACT_DIRECTORY.
EOF
}

mapping_root() {
    [ -d "$1" ] || alias_die "build-alias root does not exist: $1"
    cd "$1" && pwd -P
}

alias_from_index() {
    local index="$1"
    awk -v n="$index" 'BEGIN {
        if (n < 1 || n != int(n)) exit 1
        value = ""
        while (n > 0) {
            remainder = (n - 1) % 26
            value = sprintf("%c", 65 + remainder) value
            n = int((n - 1) / 26)
        }
        print value
    }'
}

command_name="${1:-}"
case "$command_name" in
    list|path)
        [ "$#" -eq 2 ] || { alias_usage >&2; exit 1; }
        build_root="$(mapping_root "$2")"
        mapping_file="$build_root/build-aliases.tsv"
        if [ "$command_name" = path ]; then
            printf '%s\n' "$mapping_file"
            exit 0
        fi
        [ -f "$mapping_file" ] || alias_die "no build-alias map: $mapping_file"
        cat "$mapping_file"
        ;;
    register)
        [ "$#" -eq 2 ] || { alias_usage >&2; exit 1; }
        [ -d "$2" ] || alias_die "artifact directory does not exist: $2"
        artifact_dir="$(cd "$2" && pwd -P)"
        key_file="$artifact_dir/build-key"
        [ -f "$key_file" ] || alias_die "artifact has no build-key: $artifact_dir"
        build_key="$(<"$key_file")"
        [[ "$build_key" =~ ^[0-9a-f]{64}$ ]] ||
            alias_die "invalid build key in $key_file"

        short_key="${build_key:0:12}"
        cache_tag="cfg$short_key"
        cache_directory="$(basename "$artifact_dir")"
        [[ "$cache_directory" == *-"$short_key" ]] ||
            alias_die "artifact directory does not match its build key: $artifact_dir"
        build_root="$(mapping_root "$(dirname "$artifact_dir")")"
        mapping_file="$build_root/build-aliases.tsv"
        lock_file="$build_root/.build-aliases.lock"

        command -v flock >/dev/null 2>&1 || alias_die "flock is required for build aliases"
        exec {lock_fd}>>"$lock_file"
        flock "$lock_fd"

        if [ ! -e "$mapping_file" ]; then
            (
                set -o noclobber
                printf '# alias\tcache_tag\tcache_directory\tfull_build_key\n' >"$mapping_file"
            ) || alias_die "could not create build-alias map: $mapping_file"
        fi

        malformed_line="$(awk -F '\t' '
            !/^#/ && NF && (NF != 4 || $1 !~ /^[A-Z]+$/ || $2 !~ /^cfg[0-9a-f]+$/) {
                print NR
                exit
            }
        ' "$mapping_file")"
        [ -z "$malformed_line" ] ||
            alias_die "malformed entry at $mapping_file:$malformed_line"

        existing_record="$(awk -F '\t' -v key="$build_key" '
            !/^#/ && $4 == key {print $1 "\t" $2 "\t" $3}
        ' "$mapping_file")"
        existing_count="$(printf '%s\n' "$existing_record" | awk 'NF {count++} END {print count+0}')"
        [ "$existing_count" -le 1 ] ||
            alias_die "build key occurs more than once in $mapping_file: $build_key"

        if [ "$existing_count" -eq 1 ]; then
            IFS=$'\t' read -r build_alias recorded_tag recorded_directory <<<"$existing_record"
            [ "$recorded_tag" = "$cache_tag" ] && [ "$recorded_directory" = "$cache_directory" ] ||
                alias_die "existing alias disagrees with artifact $artifact_dir"
        else
            collision="$(awk -F '\t' -v tag="$cache_tag" -v directory="$cache_directory" '
                !/^#/ && ($2 == tag || $3 == directory) {print $1; exit}
            ' "$mapping_file")"
            [ -z "$collision" ] ||
                alias_die "cache identity already maps to alias $collision in $mapping_file"

            alias_index="$(awk -F '\t' '!/^#/ && NF {count++} END {print count+1}' "$mapping_file")"
            build_alias="$(alias_from_index "$alias_index")" ||
                alias_die "could not allocate a build alias"
            while awk -F '\t' -v wanted="$build_alias" '!/^#/ && $1 == wanted {found=1} END {exit !found}' "$mapping_file"; do
                alias_index=$((alias_index + 1))
                build_alias="$(alias_from_index "$alias_index")" ||
                    alias_die "could not allocate a build alias"
            done
            printf '%s\t%s\t%s\t%s\n' \
                "$build_alias" "$cache_tag" "$cache_directory" "$build_key" >>"$mapping_file"
        fi

        flock -u "$lock_fd"
        exec {lock_fd}>&-
        printf '%s\n' "$build_alias"
        ;;
    -h|--help|help|"")
        alias_usage
        ;;
    *)
        alias_usage >&2
        alias_die "unknown command: $command_name"
        ;;
esac
