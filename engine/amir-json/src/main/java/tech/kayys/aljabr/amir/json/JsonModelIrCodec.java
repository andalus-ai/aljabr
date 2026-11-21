package tech.kayys.aljabr.amir.json;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.fasterxml.jackson.core.StreamWriteFeature;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jdk8.Jdk8Module;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.spi.model.ir.ModelIrCodec;
import tech.kayys.aljabr.spi.model.ir.ModelIrDocument;
import tech.kayys.aljabr.spi.model.ir.ModelIrSchemaVersion;

/**
 * Strict JSON serialization for versioned AMIR documents.
 *
 * <p>The codec binds JSON directly to the typed AMIR records. Unknown fields
 * are rejected instead of being silently discarded; producers that need
 * forward-compatible vendor data must place it in the document or descriptor
 * extension maps. The mapper is immutable after construction and safe to share
 * between threads.
 */
public final class JsonModelIrCodec implements ModelIrCodec {

    public static final String FORMAT_ID = "amir-json";
    public static final String MEDIA_TYPE = "application/vnd.aljabr.amir+json";

    private static final Set<String> MEDIA_TYPES = Set.of(MEDIA_TYPE, "application/json");

    private final ObjectMapper mapper;

    public JsonModelIrCodec() {
        JsonFactory factory = JsonFactory.builder()
                .disable(StreamReadFeature.AUTO_CLOSE_SOURCE)
                .disable(StreamWriteFeature.AUTO_CLOSE_TARGET)
                .build();

        this.mapper = com.fasterxml.jackson.databind.json.JsonMapper.builder(factory)
                .addModule(new Jdk8Module())
                .enable(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY)
                .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .build();
    }

    @Override
    public String formatId() {
        return FORMAT_ID;
    }

    @Override
    public Set<String> mediaTypes() {
        return MEDIA_TYPES;
    }

    @Override
    public ModelIrDocument read(InputStream input) throws IOException {
        Objects.requireNonNull(input, "input must not be null");
        ModelIrDocument document = mapper.readValue(input, ModelIrDocument.class);
        if (document == null) {
            throw new IOException("AMIR JSON document must not be null");
        }
        ModelIrSchemaVersion.CURRENT.requireCompatibleWith(document.schemaVersion());
        return document;
    }

    @Override
    public void write(ModelIrDocument document, OutputStream output) throws IOException {
        Objects.requireNonNull(document, "document must not be null");
        Objects.requireNonNull(output, "output must not be null");
        ModelIrSchemaVersion.CURRENT.requireCompatibleWith(document.schemaVersion());
        mapper.writeValue(output, document);
    }
}
