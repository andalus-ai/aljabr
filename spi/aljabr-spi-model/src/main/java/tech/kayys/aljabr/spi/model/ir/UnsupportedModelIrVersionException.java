package tech.kayys.aljabr.spi.model.ir;

/** Raised when an AMIR reader cannot safely interpret a document schema version. */
public final class UnsupportedModelIrVersionException extends IllegalArgumentException {
    private final ModelIrSchemaVersion readerVersion;
    private final ModelIrSchemaVersion documentVersion;

    public UnsupportedModelIrVersionException(
            ModelIrSchemaVersion readerVersion, ModelIrSchemaVersion documentVersion) {
        super("AMIR schema " + documentVersion + " is incompatible with reader schema " + readerVersion);
        this.readerVersion = readerVersion;
        this.documentVersion = documentVersion;
    }

    public ModelIrSchemaVersion readerVersion() {
        return readerVersion;
    }

    public ModelIrSchemaVersion documentVersion() {
        return documentVersion;
    }
}
