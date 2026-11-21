# Aljabr Diffusion Euler Scheduler

Backend-independent deterministic Euler schedule construction over continuous sigma coordinates.

The scheduler uses a Karras-style rho schedule and exposes the current and
next sigma for each reverse step. The final transition targets sigma zero.
Tensor update execution and model-output conversion remain in a later
execution layer; this module does not depend on tensors or compute backends.
