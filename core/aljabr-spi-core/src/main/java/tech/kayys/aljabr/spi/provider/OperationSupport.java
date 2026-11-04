package tech.kayys.aljabr.spi.provider;

/** Describes how an operation is supported by a queried backend/session. */
public enum OperationSupport {
    /** Executed by the queried backend implementation. */
    NATIVE,
    /** Deliberately delegated to another identified backend. */
    DELEGATED,
    /** Implemented through a semantically equivalent decomposition. */
    EMULATED,
    /** Known not to be supported for the queried constraints. */
    UNSUPPORTED,
    /** No reliable support claim is currently available. */
    UNKNOWN
}