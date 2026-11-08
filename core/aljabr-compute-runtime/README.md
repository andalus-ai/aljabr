# Aljabr Compute Runtime

This module is the runtime owner for provider-backed tensor compute. It depends on
`aljabr-spi-compute`; the foundational tensor module does not depend on this
module or on the provider SPI.

## Usage

```java
try (ComputeRuntime runtime = ComputeRuntime.open()) {
    Tensor input = Tensor.zeros(runtime.backend(), 1, 16);
    // Use tensors created with this backend while the runtime remains open.
}
```

Select a provider explicitly by its stable provider id:

```java
try (ComputeRuntime runtime = ComputeRuntime.open("andalus/cpu")) {
    Tensor input = Tensor.ones(runtime.backend(), 4);
}
```

The default selection is `auto`, which ranks available devices as GPU,
accelerator, CPU, then other. Set `-Daljabr.compute.provider=andalus/cpu`
to select a provider for the process. Explicit selection fails rather than
silently falling back to another provider.

The runtime owns discovered provider instances and the selected device session.
Keep it open while tensors using its backend are in use; close it when that
work is complete. Tensor storage remains governed by the tensor factory and
backend implementation. This initial runtime exposes one selected backend;
multi-device scheduling and per-operation dispatch are separate concerns.
