# Aljabr Diffusion Scheduler API

Family-neutral scheduler metadata and inference schedule contracts.

The API distinguishes discrete timestep, continuous sigma and continuous
flow-time domains, and makes denoiser prediction semantics explicit
(`EPSILON`, `VELOCITY`, `SAMPLE`, `FLOW`). It does not assume that all
schedulers use alpha-cumprod equations.

Tensor update execution and multistep scheduler state are intentionally not
part of this first foundation. Those contracts must be designed against Aljabr's
actual tensor ownership and backend execution APIs before migrating DDIM.
