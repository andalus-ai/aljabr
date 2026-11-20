package tech.kayys.aljabr.converter.ir;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorHeader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensorInfo;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SafetensorModelDocumentAdapterTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void inspectsSingleFileWithoutLoadingTensorPayloads() throws Exception {
        Path model = Files.createDirectory(temporaryDirectory.resolve("tiny-model"));
        Path file = Files.createFile(model.resolve("model.safetensors"));
        Files.writeString(model.resolve("config.json"), """
                {
                  "model_type": "llama",
                  "architectures": ["LlamaForCausalLM"],
                  "vendor_extension": {"keep": true}
                }
                """);

        var info = SafetensorTensorInfo.of("layers.0.weight", SafetensorDType.F16,
                new long[]{2, 3}, 16L, 28L);
        var header = SafetensorHeader.of(128L, Map.of(info.name(), info), Map.of("format", "pt"));
        var loader = new HeaderOnlyLoader(Map.of(file.toAbsolutePath().normalize(), header));

        var document = new SafetensorModelDocumentAdapter(loader, new ObjectMapper()).inspect(file);

        assertEquals("tiny-model", document.id());
        assertEquals("LlamaForCausalLM", document.architecture());
        assertEquals("safetensors", document.sourceFormat());
        assertEquals("pt", document.metadata().get("safetensors.format"));
        assertTrue(document.extensions().get("huggingface.config.json").contains("vendor_extension"));
        assertEquals(1, document.tensors().size());
        var tensor = document.tensors().get(0);
        assertEquals("F16", tensor.dataType());
        assertEquals(java.util.List.of(2L, 3L), tensor.shape());
        assertEquals(16L, tensor.payload().offset());
        assertEquals(12L, tensor.payload().length());
        assertEquals(PayloadReference.OffsetBasis.DATA_SECTION, tensor.payload().offsetBasis());
        assertEquals(1, loader.headerCalls);
        assertEquals(0, loader.payloadCalls);
    }

    @Test
    void inspectsIndexedShardsAndRetainsEachTensorSource() throws Exception {
        Path model = Files.createDirectory(temporaryDirectory.resolve("sharded-model"));
        Path first = Files.createFile(model.resolve("model-00001-of-00002.safetensors"));
        Path second = Files.createFile(model.resolve("model-00002-of-00002.safetensors"));
        Files.writeString(model.resolve("config.json"), """
                {"model_type":"mistral"}
                """);
        Files.writeString(model.resolve("model.safetensors.index.json"), """
                {
                  "metadata": {"total_size": 16},
                  "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors"
                  }
                }
                """);

        var embedding = SafetensorTensorInfo.of("embedding.weight", SafetensorDType.F32,
                new long[]{2}, 0L, 8L);
        var lmHead = SafetensorTensorInfo.of("lm_head.weight", SafetensorDType.F32,
                new long[]{2}, 8L, 16L);
        var headers = new LinkedHashMap<Path, SafetensorHeader>();
        headers.put(first.toAbsolutePath().normalize(),
                SafetensorHeader.of(64L, Map.of(embedding.name(), embedding), Map.of("format", "pt")));
        headers.put(second.toAbsolutePath().normalize(),
                SafetensorHeader.of(64L, Map.of(lmHead.name(), lmHead), Map.of("format", "pt")));
        var loader = new HeaderOnlyLoader(headers);

        var document = new SafetensorModelDocumentAdapter(loader, new ObjectMapper()).inspect(model);

        assertEquals("mistral", document.architecture());
        assertEquals(2, document.tensors().size());
        assertEquals(first.toAbsolutePath().normalize().toUri(), document.tensors().get(0).payload().source());
        assertEquals(second.toAbsolutePath().normalize().toUri(), document.tensors().get(1).payload().source());
        assertEquals(2, loader.headerCalls);
        assertEquals(0, loader.payloadCalls);
    }

    private static final class HeaderOnlyLoader extends SafetensorFFMLoader {
        private final Map<Path, SafetensorHeader> headers;
        private int headerCalls;
        private int payloadCalls;

        private HeaderOnlyLoader(Map<Path, SafetensorHeader> headers) {
            this.headers = headers;
        }

        @Override
        public SafetensorHeader loadHeaderOnly(Path filePath) {
            headerCalls++;
            SafetensorHeader header = headers.get(filePath.toAbsolutePath().normalize());
            if (header == null) {
                throw new IllegalArgumentException("Unexpected file: " + filePath);
            }
            return header;
        }

        @Override
        public tech.kayys.aljabr.safetensor.loader.SafetensorLoadResult load(Path filePath) {
            payloadCalls++;
            throw new AssertionError("AMIR inspection must not load tensor payloads");
        }
    }
}
