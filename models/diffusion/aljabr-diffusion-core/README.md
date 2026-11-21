# Aljabr Diffusion Core

Family-neutral diffusion orchestration primitives.

The initial core provides an immutable, explicitly supplied family-provider
registry. It does not perform global registration, load model weights, select
hardware backends or implement a serving lifecycle. Applications may construct
the registry from providers discovered through `ServiceLoader` or explicit
dependency injection.

Future additions must remain family-neutral and reuse Aljabr tensor, model and
backend APIs rather than introducing a parallel compute stack.
