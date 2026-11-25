package tech.kayys.aljabr.safetensor.loader;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.chronos2.Chronos2CheckpointMapper;
import tech.kayys.aljabr.models.chronos2.Chronos2Config;
import tech.kayys.aljabr.safetensor.chronos2.Chronos2SafetensorBindings;

class Chronos2SafetensorBindingsTest {
    private static Chronos2Config config() {
        return new Chronos2Config(8, 2, 2, 2, 1, 8, 4, 2, 8, 1, 1,
                1e-6, 10000d, true, false, List.of(0.1d, 0.9d));
    }

    @Test
    void bindsOriginalTensorViewsWithoutMaterializingPayloads() {
        Chronos2CheckpointMapper mapper = new Chronos2CheckpointMapper();
        Map<String, SafetensorTensor> tensors = new LinkedHashMap<>();
        try (Arena arena = Arena.ofConfined()) {
            mapper.expectedTensors(config()).forEach((name, spec) -> {
                long elements = 1;
                for (long dimension : spec.shape()) elements = Math.multiplyExact(elements, dimension);
                long bytes = Math.multiplyExact(elements, SafetensorDType.F32.byteSize());
                SafetensorTensorInfo info = SafetensorTensorInfo.of(
                        name, SafetensorDType.F32, spec.shape(), 0, bytes);
                tensors.put(name, new SafetensorTensor(info, arena.allocate(bytes)));
            });

            Chronos2SafetensorBindings.Binding binding =
                    new Chronos2SafetensorBindings().bind(config(), tensors, tensors.size());
            String parameter = "encoder.block.0.layer.0.self_attention.q.weight";
            assertSame(tensors.get(parameter), binding.tensorForParameter(parameter));
            org.junit.jupiter.api.Assertions.assertEquals(parameter, binding.checkpointNameForParameter(parameter));
        } finally {
            tensors.values().forEach(SafetensorTensor::close);
        }
    }

    @Test
    void rejectsUnexpectedDtypeBeforeBinding() {
        Chronos2CheckpointMapper mapper = new Chronos2CheckpointMapper();
        Map<String, SafetensorTensor> tensors = new LinkedHashMap<>();
        try (Arena arena = Arena.ofConfined()) {
            mapper.expectedTensors(config()).forEach((name, spec) -> {
                long elements = 1;
                for (long dimension : spec.shape()) elements = Math.multiplyExact(elements, dimension);
                SafetensorDType dtype = name.equals("shared.weight") ? SafetensorDType.F16 : SafetensorDType.F32;
                long bytes = Math.multiplyExact(elements, dtype.byteSize());
                SafetensorTensorInfo info = SafetensorTensorInfo.of(name, dtype, spec.shape(), 0, bytes);
                tensors.put(name, new SafetensorTensor(info, arena.allocate(bytes)));
            });

            assertThrows(IllegalArgumentException.class,
                    () -> new Chronos2SafetensorBindings().bind(config(), tensors, tensors.size()));
        } finally {
            tensors.values().forEach(SafetensorTensor::close);
        }
    }
}
