package tech.kayys.aljabr.backend.cpu;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.testing.TensorViewContract;

/** Applies the shared tensor-view contract to the CPU provider. */
class CpuTensorViewContractTest extends TensorViewContract {

    @Override
    protected ComputeBackend createBackend() {
        return new CpuBackend();
    }
}
