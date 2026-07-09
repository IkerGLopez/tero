#!/usr/bin/env bash
set -e

# using v0.14.1 since when using v0.15.1 we get "Failed to import from vllm._C ... _C.abi.so" and then "AttributeError: '_OpNamespace' '_C_utils' object has no attribute 'init_cpu_threads_env'" when starting vllm serve.
[ -d var/vllm/.git ] || git clone -b v0.14.1 https://github.com/vllm-project/vllm.git var/vllm
cd var/vllm

# C++: 0 < M <= 8 is parsed as (0 < M) <= 8; fix to (0 < M) && (M <= 8) so Clang on macOS accepts it
sed -i.bak 's/static_assert(0 < M <= 8);/static_assert(0 < M \&\& M <= 8);/g' csrc/cpu/cpu_attn_vec.hpp 2>/dev/null || true
sed -i.bak 's/static_assert(0 < M <= 16);/static_assert(0 < M \&\& M <= 16);/g' csrc/cpu/cpu_attn_vec16.hpp 2>/dev/null || true

if [ ! -x .venv/bin/vllm ]; then
  ([ -d .venv ] || uv venv --seed .venv)
  . .venv/bin/activate
  uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
  uv pip install -e ".[audio]"
else
  . .venv/bin/activate
fi

cleanup() {
  [ -n "$CHAT_PID" ] && kill "$CHAT_PID" 2>/dev/null || true
  [ -n "$EMBED_PID" ] && kill "$EMBED_PID" 2>/dev/null || true
  [ -n "$TRANSCRIBE_PID" ] && kill "$TRANSCRIBE_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype auto --api-key test-token --port 8001 --enable-auto-tool-choice --tool-call-parser hermes &
CHAT_PID=$!
sleep 2
vllm serve nomic-ai/nomic-embed-text-v1 --api-key test-token --trust-remote-code --port 8002 &
EMBED_PID=$!

wait

