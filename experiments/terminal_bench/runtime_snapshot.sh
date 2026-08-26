#!/bin/sh
# Task-agnostic post-agent runtime evidence for Linux images without Python.
# Keep this POSIX-sh-only: it is the last-resort collector for minimal images.
# Process environments are deliberately never read.

output=
if [ "$#" -eq 2 ] && [ "$1" = "--output" ]; then
    output=$2
else
    echo "usage: runtime_snapshot.sh --output PATH" >&2
    exit 2
fi

case "$output" in
    /*) ;;
    *) echo "output must be absolute" >&2; exit 2 ;;
esac

rm -rf -- "$output" || exit 1
mkdir -p "$output/processes" "$output/proc-net-before" \
    "$output/proc-net-after" "$output/system" || exit 1

copy_proc_file() {
    source=$1
    destination=$2
    if [ -r "$source" ]; then
        cp "$source" "$destination" 2>/dev/null || : >"$destination.unreadable"
    else
        : >"$destination.unreadable"
    fi
}

copy_link() {
    source=$1
    destination=$2
    if readlink "$source" >"$destination" 2>/dev/null; then
        :
    else
        rm -f -- "$destination"
        : >"$destination.unreadable"
    fi
}

capture_net() {
    phase=$1
    for name in tcp tcp6 udp udp6 unix; do
        copy_proc_file "/proc/net/$name" "$output/proc-net-$phase/$name.txt"
    done
}

capture_net before
process_count=0
for proc_dir in /proc/[0-9]*; do
    [ -d "$proc_dir" ] || continue
    pid=${proc_dir#/proc/}
    case "$pid" in *[!0-9]*|'') continue ;; esac
    target="$output/processes/$pid"
    mkdir -p "$target/fd" "$target/ns" || continue
    copy_proc_file "$proc_dir/stat" "$target/stat"
    copy_proc_file "$proc_dir/status" "$target/status"
    copy_proc_file "$proc_dir/cmdline" "$target/cmdline"
    copy_proc_file "$proc_dir/comm" "$target/comm"
    copy_link "$proc_dir/exe" "$target/exe.link"
    copy_link "$proc_dir/cwd" "$target/cwd.link"
    copy_link "$proc_dir/root" "$target/root.link"
    for namespace in cgroup ipc mnt net pid pid_for_children time \
        time_for_children user uts; do
        copy_link "$proc_dir/ns/$namespace" "$target/ns/$namespace.link"
    done
    for fd_path in "$proc_dir"/fd/*; do
        [ -e "$fd_path" ] || [ -L "$fd_path" ] || continue
        fd=${fd_path##*/}
        copy_link "$fd_path" "$target/fd/$fd.link"
    done
    process_count=$((process_count + 1))
done
capture_net after

copy_proc_file /proc/sys/kernel/random/boot_id "$output/system/boot_id.txt"
copy_proc_file /proc/uptime "$output/system/uptime.txt"
copy_proc_file /proc/version "$output/system/proc_version.txt"
copy_link /proc/self/ns/net "$output/system/self_net_namespace.link"
copy_link /proc/1/ns/net "$output/system/pid1_net_namespace.link"
uname -a >"$output/system/uname.txt" 2>/dev/null || \
    : >"$output/system/uname.txt.unreadable"
hostname >"$output/system/hostname.txt" 2>/dev/null || \
    : >"$output/system/hostname.txt.unreadable"
cp "$0" "$output/collector.sh" 2>/dev/null || :

cat >"$output/manifest.json" <<EOF
{
  "schema": "gv-post-run-runtime-snapshot/v1",
  "status": "complete",
  "collector_runtime": "posix-sh-fallback",
  "capture_boundary": "after solver return, before official verifier",
  "purpose": "task-agnostic runtime evidence; no verdict is computed",
  "privacy": {"process_environments_captured": false},
  "process_count": $process_count,
  "network": {
    "before": "proc-net-before/{tcp,tcp6,udp,udp6,unix}.txt",
    "after": "proc-net-after/{tcp,tcp6,udp,udp6,unix}.txt"
  },
  "process_layout": "processes/PID/{stat,status,cmdline,comm,*.link,fd/*.link,ns/*.link}"
}
EOF

exit 0
