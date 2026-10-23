# Image contract

What the recipes in this repository assume about the container image, and how to move them to a newer build.

## The image

| Property | Value |
|----------|-------|
| Image | `ghcr.io/xnerv/qwen3.8-flash-next-gb10-vllm` |
| Pinned in the recipes | `sha-3415ef7`, i.e. `@sha256:5e715243f91577a152519c9aec8946a730ab5509afc33f8db7d808701921e6b5` |
| Platform | `linux/arm64` only (DGX Spark / GB10) |
| Built from | [XNerv/qwen3.8-Flash-DGX](https://github.com/XNerv/qwen3.8-Flash-DGX) at commit `3415ef77ffe0b6ead92d514896eda8390f53ac80`, a fork of [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) |
| Base | `vllm/vllm-openai` v0.32.0 (image label `qwen38.base=v0.32`) |
| `ENTRYPOINT` | `["vllm", "serve"]`, no `CMD` |
| Weights | not included, sparkrun downloads them |

Recipes pin the digest, not `latest`, so a recipe that works today keeps working when a new image is published.

## Why `executor_config.entrypoint: ""`

sparkrun runs every workload as `docker run <options> <image> bash -c <command>`. With the image's own
`ENTRYPOINT ["vllm", "serve"]`, that `bash -c ...` would be parsed as `vllm serve` arguments and the server would never
start. Every recipe clears the entrypoint and the generated command starts with `vllm serve` itself.

## Who runs the container

The recipes do not set `executor_config.user`, so sparkrun's default applies: the container runs as the invoking
user (`--user $(id -u):$(id -g)`), with `/etc/passwd` and `/etc/group` mounted read-only and `HOME=/tmp`. Nothing here
needs root. The `user` key is also gated behind `--trust` for third-party registries, so leaving it out keeps the recipes
runnable without extra flags.

The vLLM, Triton, TorchInductor and FlashInfer caches are placed by sparkrun's vLLM runtime in its own persistent
runtime-cache directory, so the recipes set no cache variables.

## Environment the image reads

These come from the upstream `scripts/serve.sh`, which is the source of truth for this image.

| Variable | Recipe value | Meaning |
|----------|--------------|---------|
| `VLLM_PLE_MMAP` | `1` | Serve the 48 GiB n-gram (PLE) table from NVMe through `mmap`. This is what makes the model fit. |
| `VLLM_PLE_MMAP_DIR` | `{launch.model_path}` | Directory the table is read from. See [the local directory requirement](#the-ple-table-needs-a-local-directory). |
| `VLLM_PLE_MMAP_WORKERS` | `32` | Threads used for the mmap gather. |
| `VLLM_PLE_MMAP_PREWARM` | `0` | `1` streams the table once at boot to warm the page cache. |
| `VLLM_PLE_MMAP_MADVISE` | `random` | `madvise` mode on the table: `random` (no readahead) or `normal`. |
| `VLLM_PLE_MMAP_FAST_ROWS` | `0` | Gathers up to this many unique rows run inline on one thread. `0` sends every gather to the worker pool. |
| `VLLM_QSA_DET_TOPK` | `1` | Use the deterministic top-k kernel for sparse attention (identical output at temperature 0). |
| `VLLM_QSA_DET_LIB` | `/opt/llm/kernel-det/_C_det.so` | Path of that kernel inside the image. |
| `VLLM_QSA_EXACT_TOPK` | `0` | `1` switches to an exact `torch.topk` fallback: also deterministic, but 20-40% slower on long prefills. Wins over `VLLM_QSA_DET_TOPK`. |
| `VLLM_MTP_DRAFT_VOCAB` | `/opt/llm/draft_vocab_65536.npy` | The MTP drafter scores only the 65,536 most frequent tokens. |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | FlashInfer sampler. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` (500k recipe only) | Lets vLLM accept a `max_model_len` above the checkpoint's native 262144. |
| `PYTORCH_CUDA_ALLOC_CONF` | empty | Blanks sparkrun's GB10 default (`expandable_segments:True`). The upstream v0.32 profile was validated with no allocator override; `expandable_segments` was a v0.31 workaround there. |

## The PLE table needs a local directory

The image's mmap patch (`src/vllm_ple_mmap.py`) reads the table from a local directory: it takes `model_config.model`
or, when set, `VLLM_PLE_MMAP_DIR`, and refuses to start with `PLE mmap: table path ... is not a local directory` otherwise.
The upstream `serve.sh` avoids the problem by passing the snapshot path as the model. sparkrun serves the checkpoint by its
Hugging Face id (`vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 --revision ...`), which is not a directory, so the recipes set
`VLLM_PLE_MMAP_DIR` to sparkrun's `{launch.model_path}`. That template resolves, inside the container, to the prepared
snapshot, for example
`/cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47`.
It needs the exact commit in `model_revision`, which the recipes pin.

## Why `model_revision` is pinned

The container runs with `HF_HUB_OFFLINE=1` (set by sparkrun, as the upstream `serve.sh` also does), and sparkrun downloads a
pinned model by commit, which writes no `refs/` entry in the cache. sparkrun therefore passes `--revision <sha>` to vLLM;
without a pin the engine would fail with `LocalEntryNotFoundError` after the weights have synced. The same pin is what lets
`{launch.model_path}` find the snapshot, so do not remove `model_revision` from a recipe.

## Compilation config

`cudagraph_mode` is `PIECEWISE`. The PLE gather is a CPU operation plus a pageable host-to-device copy, so it has to stay
outside CUDA graphs: `vllm::ple_mmap_lookup_ids` is declared as a splitting op, next to the default attention ops of
vLLM v0.32. These op names belong to the v0.32 build of the image. A different vLLM base can rename them, so check
upstream `scripts/serve.sh` before reusing this list with another image.

## Moving to a newer image

1. Publish or pick the new build and read the upstream `scripts/serve.sh` at that commit. Compare the environment
   variables, the `SPLIT` list and the `vllm serve` flags with the recipes. Anything that changed has to change here.
2. Check the base label: `docker image inspect -f '{{index .Config.Labels "qwen38.base"}}' <image>` must still describe the
   vLLM release the recipes were written for.
3. Replace the `container:` digest in `qwen3.8-flash-next-nvfp4-vllm.yaml` (the other recipes inherit it) and update the
   comment above it and the table at the top of this file.
4. Run `sparkrun recipe validate --strict` on every recipe and `sparkrun run <recipe> --dry-run`.
5. Boot at least the base recipe on a DGX Spark and record the result in [VALIDATION.md](VALIDATION.md).
