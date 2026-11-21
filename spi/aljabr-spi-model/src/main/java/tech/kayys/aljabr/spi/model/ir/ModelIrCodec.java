package tech.kayys.aljabr.spi.model.ir;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Set;

/**
 * SPI for reading and writing versioned AMIR documents in a concrete
 * serialization format.
 *
 * <p>Implementations must validate schema compatibility before exposing a
 * decoded document. They must preserve document extension keys and values
 * exactly when reading and writing a document they support. Unknown
 * format-specific fields should be retained in document or model extensions
 * rather than silently discarded.
 *
 * <p>Codecs do not own or close caller-provided streams. Implementations should
 * stream where practical and must not eagerly resolve model asset references.
 */
public interface ModelIrCodec {

    /** Stable, lowercase identifier for this serialization format. */
    String formatId();

    /** Media types accepted or emitted by this codec. */
    Set<String> mediaTypes();

    /** Reads one AMIR document without closing the supplied stream. */
    ModelIrDocument read(InputStream input) throws IOException;

    /** Writes one AMIR document without closing the supplied stream. */
    void write(ModelIrDocument document, OutputStream output) throws IOException;
}
