package tech.kayys.aljabr.spi.provider;

/** Backend-neutral category for a provider-reported execution device. */
public enum DeviceKind {
    CPU,
    GPU,
    ACCELERATOR,
    OTHER
}