package tech.kayys.aljabr.spi.model.ir;

/**
 * Stable, format-independent identity for a model.
 *
 * <p>The identifier is opaque to AMIR. Namespace and revision are optional
 * qualifiers and must not be inferred from a source filename or serialization
 * format.
 *
 * @param id stable model identifier; non-blank
 * @param namespace optional publisher or organization namespace
 * @param revision optional immutable revision, tag, or version identifier
 */
public record ModelIdentity(String id, String namespace, String revision) {

    public ModelIdentity {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("id must not be blank");
        }
        namespace = normalize(namespace);
        revision = normalize(revision);
    }

    /** Creates an identity containing only the supplied stable identifier. */
    public static ModelIdentity of(String id) {
        return new ModelIdentity(id, null, null);
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value;
    }
}
