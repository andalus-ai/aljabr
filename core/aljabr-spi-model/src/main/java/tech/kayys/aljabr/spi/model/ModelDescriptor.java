package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.spec.*;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.model.ModelFormat;

import java.net.URI;
import java.util.Map;
/**
 * Immutable record representing modeldescriptor data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelDescriptor(
                String id,
                String format, // gguf, onnx, safetensors, triton, etc
                URI source,
                Map<String, String> metadata) {
}
