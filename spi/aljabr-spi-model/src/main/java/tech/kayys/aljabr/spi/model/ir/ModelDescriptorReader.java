package tech.kayys.aljabr.spi.model.ir;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Reads model structure and metadata into AMIR without loading tensor payloads.
 *
 * <p>Implementations must keep tensor bytes external and describe them using
 * {@link PayloadReference}. A returned descriptor therefore remains usable
 * after the reader has closed its parsing resources.
 */
public interface ModelDescriptorReader {

    /** Stable source-format identifier, such as {@code GGUF} or {@code SAFETENSORS}. */
    String format();

    /** Whether this reader accepts the supplied source format identifier. */
    default boolean supports(String sourceFormat) {
        return format().equalsIgnoreCase(sourceFormat);
    }

    /**
     * Inspects a local model file without materializing tensor payloads.
     *
     * @param source existing regular model file
     * @return immutable AMIR descriptor referencing the source file
     * @throws IOException if the file cannot be inspected
     */
    ModelDescriptor read(Path source) throws IOException;
}
