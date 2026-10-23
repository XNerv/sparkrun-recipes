# Validation log

What has been checked for these recipes, on what, and what is still open. Update this file after every boot test on
real hardware.

## Checked

Date: 2026-10-23. Tool: `sparkrun` 0.4.1 on Windows, installed in a throwaway Python 3.12 virtual environment.

| Check | Result |
|-------|--------|
| `sparkrun recipe validate --strict` on all three recipes | pass |
| `sparkrun run <recipe> --dry-run -H 127.0.0.1` generates the serve command | pass, flags match the upstream `serve.sh` (see below) |
| `include:` merges `defaults` and `env` from the base recipe into both variants | pass |
| Image `sha-3415ef7` exists in GHCR and has the digest pinned in the recipes | pass (anonymous pull of the manifest) |
| Image config: `ENTRYPOINT ["vllm","serve"]`, no `CMD`, label `qwen38.base=v0.32` | confirmed from the published manifest |
| Checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` is public and pinned to `fc694b54fb0174e0913e6adf86691ef85a4ead47` | confirmed through the Hugging Face API |
| Model download covers every file the recipe needs | pass, by reading the `sparkrun` 0.4.1 source and the pinned checkpoint (see below) |

Model download, checked against the `sparkrun` 0.4.1 source and the pinned revision:

- There is a single model repo. The MTP drafter and the PLE table are in the same checkpoint
  (`model-fp8-mtp-ple.safetensors`, about 54 GB), so no second model is needed. `config.json` has no reference to
  another repo, which matters because sparkrun runs with `HF_HUB_OFFLINE=1`.
- `model.safetensors.index.json` maps 299,545 tensors to 11 files: the 10 shards plus `model-fp8-mtp-ple.safetensors`.
  The sparkrun checkpoint check (all files in the `weight_map` present) therefore covers the PLE file too.
- `sparkrun` calls `huggingface_hub.snapshot_download(revision=<model_revision>)` with no `allow_patterns` for
  non-GGUF models, so the whole repo is fetched (about 124 GiB). It keeps Xet enabled, and `hf-xet` is installed with
  it, which the large Xet files need. Make sure the cache disk has room for the full repo.
- Not run: the actual 124 GiB download.

The generated command for the base recipe carries the same flags as the upstream script: `--enable-prefix-caching`,
`--enable-chunked-prefill`, `--enable-prompt-tokens-details`, `--no-enable-flashinfer-autotune`, `--kv-cache-dtype auto`,
the MTP `--speculative-config`, the `--compilation-config` with `PIECEWISE` and the splitting ops, plus `--revision`.

Notes on the `sparkrun show` estimate:

- Weights are taken from `metadata.model_vram: 75` (GB), because the PLE table lives on NVMe.
- For the 500k recipe the KV demand uses `max_model_len` (11.6 GiB), but the "model limit" line still reports the
  checkpoint's native 262,144: the estimator does not apply `hf_overrides`.

## Comparison with the upstream `scripts/serve.sh`

Compared on 2026-10-23 against `scripts/serve.sh` of [XNerv/qwen3.8-Flash-DGX](https://github.com/XNerv/qwen3.8-Flash-DGX)
(the image's source) and the image's patch code under `src/`.

Identical, checked by script: the 19 `splitting_ops`, `cudagraph_mode`, the YaRN `hf_overrides` JSON, every environment
variable and the defaults (`GPU_MEM` 0.80, `SEQS` 8, `MTP` 2, `CTX` 262144, deterministic top-k, draft vocabulary, madvise
`random`, `FAST_ROWS` 0). In the 500k recipe the drafter's `max_model_len` equals `CTX`.

Fixed after this comparison:

- `VLLM_PLE_MMAP_DIR` was missing. The mmap patch needs a local directory and `serve.sh` gets one by passing the snapshot
  path; sparkrun passes a Hugging Face id, so the boot would have failed. See [IMAGE-CONTRACT.md](IMAGE-CONTRACT.md#the-ple-table-needs-a-local-directory).
- sparkrun's GB10 default `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is now blanked, to match the upstream v0.32
  profile, which sets no allocator override.

Different on purpose, because sparkrun owns that part of the launch:

| `serve.sh` | These recipes |
|------------|---------------|
| `docker run --gpus all --ipc=host --shm-size 16g --restart unless-stopped`, port `18300` | sparkrun's executor: shareable IPC, 32 GB `/dev/shm`, host network, port `8000` |
| `HF_HOME=/hf`, `HF_HUB_OFFLINE=1`, snapshot path as the model | Equivalent: sparkrun mounts the cache at `/cache/huggingface`, sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` itself (checked in the sparkrun 0.4.1 source), and serves the id with the pinned `--revision` |
| `--cpuset-cpus 5-9,15-19` | not available in sparkrun (listed in the README's known limits) |
| `EFFORT_ALIAS` patched chat template, bind-mounted | not done, so `reasoning_effort: high` returns `400` |
| `MODE=hybrid` with a locally prepared `-fp8hybrid` snapshot | not covered: the conversion scripts are not inside the image |
| `COMPILE_CACHE`, `PROM_MULTIPROC`, `LOG_REQUESTS`, `KV_CACHE_MEM` | not set; the compile caches are kept by sparkrun's runtime cache |

The [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX) repository is an older snapshot of the
same upstream (27 files against 173 here) with its own GPU-gather kernel. Its recipes cannot be reused as they are: the
environment names differ (`VLLM_QSA_EXACT_TOPK=1` there, `VLLM_QSA_DET_TOPK` here), and so do the splitting ops
(`qwen3_8_flash_next_*` and `ple_mmap_lookup` there, `qwen4_exp_*` and `ple_mmap_lookup_ids` here). Its mmap code has the same
local-directory check and its recipes set no `VLLM_PLE_MMAP_DIR`; whether they boot through the Hugging Face id on that image
was not verified.

## Not checked yet (needs a DGX Spark)

Nobody has booted these recipes on a GB10 yet. Until that is done, treat them as unverified.

- [ ] Boot of `qwen3.8-flash-next-nvfp4-vllm` ends with `Application startup complete`. In the log, look for
      `PLE mmap: layer ..., shards, ... on disk`: it proves `VLLM_PLE_MMAP_DIR` pointed at the snapshot.
- [ ] No `PYTORCH_CUDA_ALLOC_CONF` in the container (`docker exec <container> printenv PYTORCH_CUDA_ALLOC_CONF` prints an empty
      line). If boots stall in `allocate_kv_cache` (the v0.31 symptom upstream), set `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [ ] The container runs as the invoking user, not root: `docker inspect <container> --format '{{.Config.User}}'` shows a
      numeric `uid:gid`. Also check that the image files are readable by that user:
      `docker run --rm --user $(id -u):$(id -g) --entrypoint ls ghcr.io/xnerv/qwen3.8-flash-next-gb10-vllm:sha-3415ef7 -l /opt/llm`.
- [ ] Hugging Face download of the NVIDIA checkpoint through sparkrun. One of its shards is about 50 GiB; the upstream
      script downloads it through Xet, and it is not known whether sparkrun's download path does.
- [ ] `/v1/models` and a chat completion respond; prefix-cache hit on a repeated prompt; two identical greedy requests
      return identical text.
- [ ] 500k recipe boots and serves a long prompt.
- [ ] Concurrent recipe boots with `max_num_seqs: 16` and the bf16 recurrent state.
- [ ] Real `model_vram` after boot, to replace the 75 GB estimate if it differs.
- [ ] Boot time and KV pool size on this box, for the README.
