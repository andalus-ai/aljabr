package tech.kayys.aljabr.backend.metal.cache;

import java.lang.foreign.ValueLayout;
import java.util.Optional;

import tech.kayys.aljabr.cache.KVCacheBufferReadView;
import tech.kayys.aljabr.cache.KVCacheBufferRegion;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheMemoryDomain;
import tech.kayys.aljabr.cache.testing.KVCacheBufferProviderContractTest;
import tech.kayys.aljabr.metal.binding.MetalBinding;

class MetalKVCacheBufferProviderContractTest extends KVCacheBufferProviderContractTest {
    private Optional<MetalBinding> availableBinding() {
        if (!MetalBinding.initialize()) return Optional.empty();
        MetalBinding binding = MetalBinding.getInstance();
        binding.init();
        return binding.supportsKVBufferAllocation() ? Optional.of(binding) : Optional.empty();
    }

    @Override
    protected boolean isProviderAvailable() {
        return availableBinding().isPresent();
    }

    @Override
    protected Provider openProvider() {
        MetalBinding binding = availableBinding().orElseThrow();
        MetalKVCacheStorage storage = new MetalKVCacheStorage(
                new KVCacheLayout(5, new int[] {1, 2}, new int[] {2, 3}), binding);
        return new Provider(storage, "metal", KVCacheMemoryDomain.DEVICE,
                (lease, region) -> MetalKVCacheBufferAccess.withBuffer(
                        lease, region, view -> view.readOnlyContents()
                                .toArray(ValueLayout.JAVA_FLOAT)));
    }
}
