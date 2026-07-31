#!/usr/bin/env bash
set -euo pipefail

# Minimal model of WG's synchronous coordinator tick.  The first run proves that
# an unbounded maintenance subprocess freezes last_tick even though the daemon
# remains alive.  The second run shows the required bounded behavior.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
state="$tmp/last_tick"
printf '0\n' >"$state"

tick_unbounded() {
    local n=$1
    # Models Command::output() in worktree maintenance.  A real reproducer can
    # replace this with a wedged `git merge-base` on a slow filesystem.
    exec setsid sh -c 'trap "" TERM; while :; do sleep 60; done'
    printf '%s\n' "$n" >"$state"
}

tick_unbounded 1 &
daemon_pid=$!
sleep 0.2
before=$(cat "$state")
kill -0 "$daemon_pid"
sleep 0.2
after=$(cat "$state")
if [[ "$before" != 0 || "$after" != 0 ]]; then
    echo "FAIL: unbounded tick unexpectedly advanced" >&2
    exit 1
fi
echo "PASS frozen-tick: daemon_pid=$daemon_pid alive, last_tick=$after"
kill -KILL -- "-$daemon_pid" 2>/dev/null || true
wait "$daemon_pid" 2>/dev/null || true

# GNU timeout sends TERM, waits one second, then kills the entire command it
# supervises.  Production WG should implement the equivalent with a fresh
# process group and TERM->KILL escalation rather than depending on this script.
if timeout --signal=TERM --kill-after=1 0.2 \
    sh -c 'trap "" TERM; while :; do sleep 60; done'; then
    echo "FAIL: bounded child unexpectedly succeeded" >&2
    exit 1
else
    rc=$?
    [[ "$rc" == 124 || "$rc" == 137 ]]
fi
printf '2\n' >"$state"
echo "PASS bounded-tick: timed-out maintenance was killed, last_tick=$(cat "$state")"
