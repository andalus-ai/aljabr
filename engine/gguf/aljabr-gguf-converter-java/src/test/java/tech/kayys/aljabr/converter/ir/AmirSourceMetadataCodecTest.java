package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AmirSourceMetadataCodecTest {

    @Test
    void restoresNamespacedMetadataAndTensorExtensions() {
        String tensor = encode("weight/主");
        String extension = encode("vendor.attribute");
        var decoded = AmirSourceMetadataCodec.decode(Map.of(
                "general.architecture", "llama",
                "aljabr.source.metadata.vendor.metadata/key", "metadata-value",
                "aljabr.source.extension.vendor.extension", "extension-value",
                "aljabr.source.tensor." + tensor + "." + extension, "tensor-value",
                AmirTypedMetadataCodec.PREFIX + "dGVzdA", "{\\"kind\\":\\"STRING\\",\\"value\\":\\"typed\\"}"),
                List.of(new TensorDescriptor("weight/主", "F32", List.of(2L),
                        "contiguous", null, null)));

        assertEquals(Map.of("general.architecture", "llama",
                "vendor.metadata/key", "metadata-value"), decoded.metadata());
        assertEquals(Map.of("vendor.extension", "extension-value"), decoded.extensions());
        assertEquals("tensor-value",
                decoded.tensors().get(0).extensions().get("vendor.attribute"));
    }

    @Test
    void rejectsCollisionBetweenRestoredAndNativeMetadataKeys() {
        String prefix = "aljabr.source.metadata.";
        assertThrows(IllegalArgumentException.class, () -> AmirSourceMetadataCodec.decode(
                Map.of("vendor.key", "native", prefix + "vendor.key", "restored"), List.of()));
    }

    @Test
    void rejectsTensorExtensionsForUnknownTensorNames() {
        String tensor = encode("missing");
        String extension = encode("vendor.attribute");
        assertThrows(IllegalArgumentException.class, () -> AmirSourceMetadataCodec.decode(
                Map.of("aljabr.source.tensor." + tensor + "." + extension, "value"), List.of()));
    }

    private static String encode(String value) {
        return Base64.getUrlEncoder().withoutPadding()
                .encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }
}
