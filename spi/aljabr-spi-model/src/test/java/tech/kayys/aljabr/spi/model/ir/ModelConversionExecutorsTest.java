package tech.kayys.aljabr.spi.model.ir;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelConversionExecutorsTest {

    @Test
    void findsExecutorForMatchingFormatPair() {
        ModelConversionExecutor executor = executor("safetensors", "gguf");

        assertSame(executor, ModelConversionExecutors.find(
                List.of(executor), "SAFETENSORS", "GGUF").orElseThrow());
        assertTrue(ModelConversionExecutors.find(List.of(executor), "gguf", "safetensors").isEmpty());
    }

    @Test
    void rejectsAmbiguousExecutorRegistration() {
        var first = executor("safetensors", "gguf");
        var second = executor("SAFETENSORS", "GGUF");

        assertThrows(IllegalStateException.class,
                () -> ModelConversionExecutors.find(List.of(first, second), "safetensors", "gguf"));
    }

    @Test
    void requestRequiresAbsoluteOutputAndNamespacedOptions() {
        assertThrows(IllegalArgumentException.class,
                () -> new ConversionRequest(URI.create("relative/model.safetensors"), false, Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> new ConversionRequest(URI.create("file:///model.safetensors"), false,
                        Map.of("compression", "none")));
    }

    private static ModelConversionExecutor executor(String source, String target) {
        return new ModelConversionExecutor() {
            @Override
            public boolean supports(String sourceFormat, String targetFormat) {
                return source.equalsIgnoreCase(sourceFormat) && target.equalsIgnoreCase(targetFormat);
            }

            @Override
            public ConversionExecutionResult execute(
                    ModelDescriptor model, ConversionPlan plan, ConversionRequest request) throws IOException {
                throw new UnsupportedOperationException("test provider");
            }
        };
    }
}
