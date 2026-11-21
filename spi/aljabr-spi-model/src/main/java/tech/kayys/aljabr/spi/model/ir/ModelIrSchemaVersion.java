package tech.kayys.aljabr.spi.model.ir;

/**
 * Semantic version of the AMIR document schema, independent of Aljabr releases.
 *
 * <p>Compatibility is major-version based. A reader may preserve fields it does
 * not understand when reading a document with the same major version.
 */
public record ModelIrSchemaVersion(int major, int minor, int patch)
        implements Comparable<ModelIrSchemaVersion> {

    /** Initial stable AMIR document schema. */
    public static final ModelIrSchemaVersion V1_0_0 = new ModelIrSchemaVersion(1, 0, 0);
    /** Schema version emitted by {@link ModelIrDocument#current(ModelDescriptor)}. */
    public static final ModelIrSchemaVersion CURRENT = V1_0_0;

    public ModelIrSchemaVersion {
        if (major < 0 || minor < 0 || patch < 0) {
            throw new IllegalArgumentException("schema version components must not be negative");
        }
    }

    /** True when both versions belong to the same compatibility line. */
    public boolean isCompatibleWith(ModelIrSchemaVersion other) {
        return other != null && major == other.major;
    }

    /** Requires the supplied version to belong to this version's major line. */
    public void requireCompatibleWith(ModelIrSchemaVersion other) {
        if (!isCompatibleWith(other)) {
            throw new UnsupportedModelIrVersionException(this, other);
        }
    }

    @Override
    public int compareTo(ModelIrSchemaVersion other) {
        int result = Integer.compare(major, other.major);
        if (result == 0) result = Integer.compare(minor, other.minor);
        if (result == 0) result = Integer.compare(patch, other.patch);
        return result;
    }

    @Override
    public String toString() {
        return major + "." + minor + "." + patch;
    }
}
