package tech.kayys.aljabr.plugin.kernel;

/** Base checked exception for kernel execution failures. */
public class KernelException extends Exception {
    public KernelException(String message) { super(message); }
    public KernelException(String message, Throwable cause) { super(message, cause); }
}
