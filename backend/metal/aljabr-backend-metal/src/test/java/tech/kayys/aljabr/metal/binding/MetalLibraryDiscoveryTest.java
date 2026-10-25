package tech.kayys.aljabr.metal.binding;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalLibraryDiscoveryTest {
    @Test
    void rejectsNativeLibrariesFromPreviousProjectNames() {
        assertTrue(MetalLibraryDiscovery.isLegacyLibraryName(
                Path.of("/tmp/libalkhawarizm_metal.dylib")));
        assertTrue(MetalLibraryDiscovery.isLegacyLibraryName(
                Path.of("/tmp/libaqli_metal.dylib")));
        assertFalse(MetalLibraryDiscovery.isLegacyLibraryName(
                Path.of("/tmp/libaljabr_metal.dylib")));
    }
}
