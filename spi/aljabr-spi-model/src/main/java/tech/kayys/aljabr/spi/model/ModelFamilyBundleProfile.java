package tech.kayys.aljabr.spi.model;

/**
 * Recognised bundle-profile tiers for a model family.
 *
 * <p>
 * The tier is declared as a string value ({@code "core"}, {@code "optional"},
 * {@code "experimental"}) inside the descriptor metadata map and parsed at
 * validation time.
 * @author bhangun
 */
public enum ModelFamilyBundleProfile {
    /** Shipped by default; always included in the standard distribution. */
    CORE("core"),
    OPTIONAL("optional"),
    METADATA_ONLY("metadata_only"),
    EXPERIMENTAL("experimental");

    private final String key;

    ModelFamilyBundleProfile(String key) {
        this.key = key;
    }

    /** The lowercase string value used in descriptor metadata. */
    public String key() {
        return key;
    }

    /**
     * Parse a metadata string value into a {@code ModelFamilyBundleProfile}.
     *
     * @param value the raw string from descriptor metadata, e.g. {@code "optional"}
     * @return the matching profile, or {@code null} if unrecognised
     */
    public static ModelFamilyBundleProfile fromKey(String value) {
        if (value == null)
            return null;
        for (ModelFamilyBundleProfile p : values()) {
            if (p.key.equalsIgnoreCase(value))
                return p;
        }
        return null;
    }
}
