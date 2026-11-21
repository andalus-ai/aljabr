# Aljabr model conversion CLI

This module provides a thin command-line adapter over the AMIR conversion SPIs.
It does not implement format parsing, compatibility policy, tensor conversion or
payload writing; those responsibilities remain in the registered Aljabr providers.

## Commands

Inspect compatibility without writing an output file:

```bash
./gradlew :tools:aljabr-convert-cli:run --args='plan --input ./model.safetensors --from safetensors --to gguf'
```

Execute a conversion:

```bash
./gradlew :tools:aljabr-convert-cli:run --args='convert --input ./model.safetensors --from safetensors --to gguf --output ./model.gguf'
```

A PARTIAL plan is never executed unless the caller explicitly acknowledges the
losses:

```bash
./gradlew :tools:aljabr-convert-cli:run --args='convert --input ./model.safetensors --from safetensors --to gguf --output ./model.gguf --acknowledge-partial'
```

Target-specific options use namespaced keys and may be repeated:

```bash
--option gguf.alignment=64
```

The source and target formats are explicit. The CLI intentionally does not guess
formats from file extensions. Use `plan` first when compatibility is uncertain.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Success (including a successful plan) |
| 2 | Invalid command-line arguments |
| 3 | Unsupported conversion, missing provider, or missing PARTIAL acknowledgement |
| 4 | Conversion I/O failure |

The executable discovers providers through Java `ServiceLoader`. Include the
desired format-reader, target-capability and conversion-executor artifacts on
the runtime classpath. The current converter module supplies the built-in GGUF
and Safetensors providers.
