 <p align="center">
  <img src="assets/banner.svg" alt="sparkrun-recipes" width="640" />
</p>

<h1 align="center">⚡ sparkrun-recipes</h1>

<p align="center">
  <b>A 176-billion-parameter model. One desktop box. One command.</b><br />
  Ready-to-run <a href="https://sparkrun.dev">sparkrun</a> recipes for <b>Qwen3.8-Flash-Next</b> on a single <b>NVIDIA DGX Spark (GB10)</b>.
</p>

<p align="center">
  <a href="https://sparkrun.dev"><img src="https://img.shields.io/badge/sparkrun-registry-1e40af" alt="sparkrun registry" /></a>
  <img src="https://img.shields.io/badge/platform-DGX%20Spark%20%7C%20GB10-76b900" alt="DGX Spark" />
  <img src="https://img.shields.io/badge/arch-arm64-informational" alt="arm64" />
  <img src="https://img.shields.io/badge/runtime-vLLM-blueviolet" alt="vLLM" />
  <a href="https://github.com/XNerv/sparkrun-recipes/actions/workflows/validate.yml"><img src="https://github.com/XNerv/sparkrun-recipes/actions/workflows/validate.yml/badge.svg" alt="validate" /></a>
</p>

---

## The pitch

Qwen3.8-Flash-Next is a **~176B-parameter model** (125B main + 51B n-gram, ~6B active).The NVFP4 checkpoint is reported to occupy approximately **134 GB** on disk, while the DGX Spark provides **128 GB** of unified memory. Loading the entire checkpoint into memory would exceed the system's unified-memory capacity, leaving no room for the operating system, KV cache, or other runtime allocations.

The upstream implementation addresses this by serving the large n-gram embedding table directly from NVMe through `mmap`, rather than loading the entire table into memory. The upstream project reports approximately **80 GB** of resident model weights with this approach. Actual memory usage depends on the model, runtime, workload and system configuration.

This repository packages that setup as **sparkrun recipes**, so you can launch the configured workload without maintaining a long `docker run` command.

```text
your terminal ──▶ sparkrun ──▶ ghcr.io/xnerv/qwen3.8-flash-next-gb10-vllm ──▶ Qwen3.8-Flash-Next
```

> **Validation status:** The recipes pass `sparkrun recipe validate --strict` and render the expected serving command. They have **not yet been verified through a complete model startup and inference run on GB10 hardware**. See [docs/VALIDATION.md](docs/VALIDATION.md) for the validation scope and outstanding checks.

## Quick start

You need a DGX Spark or another compatible GB10 system, Docker with the NVIDIA container runtime, sufficient free NVMe space for the model files, and [sparkrun](https://sparkrun.dev) installed.

```bash
# 1. Add this registry
sparkrun registry add https://github.com/XNerv/sparkrun-recipes.git

# 2. List its recipes
sparkrun list @xnerv

# 3. Inspect the base recipe before launching
sparkrun show @xnerv/qwen3.8-flash-next-nvfp4-vllm

# 4. Launch the model
sparkrun run @xnerv/qwen3.8-flash-next-nvfp4-vllm
```

The first launch downloads the model files and may take some time. Subsequent launches can reuse the local cache.

Once the server reports `Application startup complete`, the OpenAI-compatible API is expected to be available on port `8000`, as configured by the recipe:

```bash
curl http://<spark-host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hi from a desktop supercomputer."}]}'
```

Replace `<spark-host>` with the hostname or IP address of your machine.

## The menu

| **Recipe**                                 | **Context**            | **Intended use**                                                                                      | **Profile**                                      |
| :----------------------------------------- | :--------------------- | :---------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| `qwen3.8-flash-next-nvfp4-vllm`            | 262144 tokens (native) | **Recommended default!** General-purpose chat, RAG and agentic workloads.                             | Balanced configuration for general-purpose use   |
| `qwen3.8-flash-next-nvfp4-ctx512k-vllm`    | 524288 tokens (YaRN)   | Long-document analysis, large codebase review and multi-document synthesis.                           | Extended context with potential speed and quality trade-offs |
| `qwen3.8-flash-next-nvfp4-concurrent-vllm` | 262144 tokens (native) | Multi-client serving, API gateways and concurrent agent workloads.                                   | Tuned for higher concurrency, supporting up to 16 concurrent requests |

> **IMPORTANT — SINGLE-NODE DEPLOYMENT ONLY**
>
> All recipes require a single-node deployment. The `mmap` design requires the embedding table to reside on the serving node's local NVMe storage. Multi-node deployment is not supported in this recipes.

## What the recipes configure

- **Prefix caching:** Uses the upstream image's patched implementation. Validate cache-hit behavior against the exact image before relying on it in production.
- **Deterministic decoding:** Configures greedy decoding for repeatable results under the same runtime and execution conditions.
- **MTP speculative decoding:** Uses the model's draft head for speculative decoding.
- **Tool calling and reasoning parsers:** Configures the `qwen3_coder` and `qwen3` parsers.
- **Pinned artifacts:** References a fixed model revision and an immutable container image digest rather than a floating `latest` tag.
- **Minimal recipe overrides:** Keeps model and runtime settings in the base recipe, with derived recipes changing only their intended parameters.

## Where things live

```text
.sparkrun/
  registry.yaml                 Registry manifest
recipes/
  qwen3.8-flash-next/
    vllm/                       Model recipes
docs/
  IMAGE-CONTRACT.md             Image contract and update procedure
  VALIDATION.md                 Validation scope and outstanding checks
.github/
  workflows/
    validate.yml                Recipe validation on pushes and pull requests
```

The container image is built from [XNerv/qwen3.8-Flash-DGX](https://github.com/XNerv/qwen3.8-Flash-DGX), a fork of [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), and is published as `ghcr.io/xnerv/qwen3.8-flash-next-gb10-vllm` for `linux/arm64`.

See [docs/IMAGE-CONTRACT.md](docs/IMAGE-CONTRACT.md) for the image's responsibilities and the procedure for updating its pinned digest.

## Tweaking a recipe

Override supported defaults at launch without editing the recipe:

```bash
# Override the API port and maximum concurrent sequences
sparkrun run @xnerv/qwen3.8-flash-next-nvfp4-vllm \
  -o port=9000 \
  -o max_num_seqs=4

# Preview the generated serving command without launching the workload
sparkrun run @xnerv/qwen3.8-flash-next-nvfp4-vllm --dry-run
```

For a permanent variation, create a derived recipe, use `include:` to inherit the base configuration, and change only the settings you need.

## Known limitations

- **Single-node deployment:** These recipes are configured for one GB10 host.
- **KV cache dtype:** The KV cache is configured as `BF16`. FP8 KV cache is not enabled because the QSA (Qwen Sparse Attention) layers do not support the required path.
- **CPU affinity:** Upstream pins the container to performance cores with `--cpuset-cpus`; sparkrun does not expose an equivalent setting.

## Contributing

Recipes live under `recipes/<family>/<runtime>/` and follow the [sparkrun recipe format](https://sparkrun.dev/recipes/format/).

Before opening a pull request, validate the recipe and inspect the generated command:

```bash
sparkrun recipe validate --strict recipes/<family>/<runtime>/<recipe>.yaml
sparkrun run recipes/<family>/<runtime>/<recipe>.yaml --dry-run
```

CI runs recipe validation on pushes and pull requests. These checks do not replace a full startup and inference test on the target hardware.

## Credits

This repository builds on the work of the upstream project and the sparkrun community.

- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX): upstream vLLM implementation and `mmap` approach.
- [@jschmied](https://github.com/jschmied) and the contributors credited upstream: deterministic top-k kernel and related fixes.
- [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX): sparkrun recipes for the same upstream project, which helped shape this layout.
- [sparkrun](https://github.com/spark-arena/sparkrun) and the [Spark Arena](https://spark-arena.com) team: recipe orchestration and ecosystem.

## License

The recipes are released under the [MIT License](LICENSE). Model weights and the container image have their own licenses; review them before use.
