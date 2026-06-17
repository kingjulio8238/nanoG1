#!/bin/bash
# Build the policy inference shim (libnanog1policy.{so,dylib}) for deploy_g1.py.
# Needs the engine fork for puffernet.h — run `bash setup.sh` once first.
set -e
cd "$(dirname "$0")/.."

[ -f vendor/PufferLib/src/puffernet.h ] || { echo "puffernet.h missing — run: bash setup.sh"; exit 1; }

case "$(uname -s)" in
  Darwin) EXT=dylib ;;
  *)      EXT=so ;;
esac
OUT="deploy/libnanog1policy.$EXT"

clang -O2 -shared -fPIC -I vendor/PufferLib/src \
  deploy/nanog1_policy.c -o "$OUT" -lm
echo "Built: $OUT"
echo "Now:   python deploy/deploy_g1.py --net <robot-net-iface>   (e.g. eth0)"
