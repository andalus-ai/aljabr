# Aljabr Diffusion DDIM Scheduler

Backend-independent DDIM schedule construction and scalar reverse-step coefficients.

The module depends only on the diffusion scheduler API. It does not depend on
Tensor, a compute backend, a model family, AQLI, or Tafkir.

DDIMScheduler exposes the generic scheduler contract for epsilon-prediction
inference coordinates and an algorithm-specific DDIMInferenceSchedule for the
cumulative-alpha values and step coefficients required by DDIM execution.
Keeping those values in the DDIM-specific type avoids pretending that the
generic schedule-coordinate contract contains algorithm-specific state.

The legacy scheduler in models/aljabr-model-common remains available for
source and binary compatibility during migration.
