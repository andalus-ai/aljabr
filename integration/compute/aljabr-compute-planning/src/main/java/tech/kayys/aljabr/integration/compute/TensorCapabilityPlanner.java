package tech.kayys.aljabr.integration.compute;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorLayout;
import tech.kayys.aljabr.spi.compute.ProviderSession;
import tech.kayys.aljabr.spi.provider.OperationSupportDescriptor;

/**
 * Builds provider-neutral operation constraints from actual tensor descriptors.
 *
 * <p>This integration belongs above both the tensor API and provider SPI:
 * tensors remain independent of provider discovery, while provider sessions
 * receive consistent dtype, shape, and layout metadata. This class only asks
 * for a support description. It does not select a backend, execute an
 * operation, or apply fallback policy.</p>
 */
public final class TensorCapabilityPlanner {

    private TensorCapabilityPlanner() {
    }

    /**
     * Queries the session for support of an operation over the supplied inputs.
     *
     * <p>Every input is described under {@code input.&lt;index&gt;.*}. For
     * compatibility with providers that currently consume conventional
     * two-input keys, the first input is also exposed as {@code dtype},
     * {@code rank}, etc., and the second as {@code rhs_dtype} and
     * {@code rhs_rank}. Operation-specific constraints (for example
     * {@code normalized_numel}, {@code eps}, or {@code training}) are merged
     * without being allowed to overwrite derived tensor metadata.</p>
     *
     * @param session open provider session to query
     * @param operationId stable provider-neutral operation identifier
     * @param inputs ordered, non-null tensor inputs
     * @param operationConstraints operation arguments that affect capability
     * @return the provider's support description
     */
    public static OperationSupportDescriptor query(
            ProviderSession session,
            String operationId,
            List<? extends Tensor> inputs,
            Map<String, String> operationConstraints) {
        Objects.requireNonNull(session, "session");
        if (operationId == null || operationId.isBlank()) {
            throw new IllegalArgumentException("operationId must not be blank");
        }
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(operationConstraints, "operationConstraints");

        Map<String, String> constraints = new LinkedHashMap<>();
        for (int index = 0; index < inputs.size(); index++) {
            Tensor tensor = Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
            addTensorMetadata(constraints, "input." + index + ".", tensor);
        }
        if (!inputs.isEmpty()) {
            addPrimaryAliases(constraints, inputs.get(0));
        }
        if (inputs.size() > 1) {
            Tensor rhs = inputs.get(1);
            constraints.put("rhs_dtype", rhs.dtype().name());
            constraints.put("rhs_rank", Integer.toString(rhs.shape().rank()));
        }

        operationConstraints.forEach((key, value) -> {
            if (key == null || key.isBlank()) {
                throw new IllegalArgumentException("operation constraint keys must not be blank");
            }
            if (value == null) {
                throw new IllegalArgumentException("operation constraint values must not be null: " + key);
            }
            if (constraints.containsKey(key)) {
                throw new IllegalArgumentException(
                        "operation constraint conflicts with derived tensor metadata: " + key);
            }
            constraints.put(key, value);
        });

        return session.operationSupport(operationId, Map.copyOf(constraints));
    }

    private static void addTensorMetadata(
            Map<String, String> constraints,
            String prefix,
            Tensor tensor) {
        TensorLayout layout = Objects.requireNonNull(tensor.layout(), "tensor.layout()");
        var shape = Objects.requireNonNull(tensor.shape(), "tensor.shape()");
        if (!layout.shape().equals(shape)) {
            throw new IllegalArgumentException("Tensor layout shape does not match tensor shape");
        }
        constraints.put(prefix + "dtype", tensor.dtype().name());
        constraints.put(prefix + "rank", Integer.toString(shape.rank()));
        constraints.put(prefix + "shape", join(shape.dims()));
        constraints.put(prefix + "numel", Long.toString(shape.numel()));
        constraints.put(prefix + "strides", join(layout.strides()));
        constraints.put(prefix + "contiguous", Boolean.toString(layout.isContiguous()));
        constraints.put(prefix + "storage_offset", Long.toString(layout.storageOffset()));
        constraints.put(prefix + "device", tensor.device().name());
    }

    private static void addPrimaryAliases(Map<String, String> constraints, Tensor tensor) {
        TensorLayout layout = tensor.layout();
        var shape = tensor.shape();
        constraints.put("dtype", tensor.dtype().name());
        constraints.put("rank", Integer.toString(shape.rank()));
        constraints.put("shape", join(shape.dims()));
        constraints.put("numel", Long.toString(shape.numel()));
        constraints.put("strides", join(layout.strides()));
        constraints.put("contiguous", Boolean.toString(layout.isContiguous()));
        constraints.put("storage_offset", Long.toString(layout.storageOffset()));
        constraints.put("device", tensor.device().name());
    }

    private static String join(long[] values) {
        StringBuilder result = new StringBuilder();
        for (long value : values) {
            if (!result.isEmpty()) {
                result.append(',');
            }
            result.append(value);
        }
        return result.toString();
    }
}
