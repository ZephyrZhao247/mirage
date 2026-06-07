#!/usr/bin/env bash
set -o pipefail
# test-on-gpu's worktree sync (rsync --delete from the shared-FS worktree, whose
# deps/cutlass & deps/json are empty submodule stubs) wipes the build deps. The
# native core.so is already compiled, but the runtime megakernel nvcc-compile needs
# cutlass headers — so restore them from the main checkout before running.
[ -d deps/cutlass/include ] || rsync -a /mnt/shared/zepengz/projects/mirage/deps/cutlass/ deps/cutlass/
[ -f deps/json/CMakeLists.txt ] || rsync -a /mnt/shared/zepengz/projects/mirage/deps/json/ deps/json/
# Clean any stale trace artifacts from a previous run in cwd.
rm -f dsv3_prof.perfetto-trace dsv3_prof.csv
timeout 1500 python demo/deepseek_v3/demo_new.py \
  --model-path /mnt/shared/models/DeepSeek-V3 \
  --layers 0-3 \
  --max-num-batched-requests 1 \
  --trace-name dsv3_prof \
  --output-dir output/dsv3_prof
rc=$?
echo "=== demo_new.py exit code: $rc ==="
pkill -9 -f "[d]emo_new" 2>/dev/null || true
echo "=== trace artifacts in cwd ==="
ls -la dsv3_prof.perfetto-trace dsv3_prof.csv 2>&1 || true
echo "=== output dir ==="
ls -la output/dsv3_prof 2>&1 || true
exit $rc
