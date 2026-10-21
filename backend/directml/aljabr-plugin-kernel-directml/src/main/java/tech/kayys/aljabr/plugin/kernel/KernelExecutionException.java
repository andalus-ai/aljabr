package tech.kayys.aljabr.plugin.kernel;

/** Thrown when the kernel plugin fails during execution. */
public class KernelExecutionException extends KernelException {

    private final String platform;
    private final String operationName;

    public KernelExecutionException(String platform, String operationName, String message) {
        super("[" + platform + "] " + operationName + ": " + message);
        this.platform = platform;
        this.operationName = operationName;
    }

    public String getPlatform() {
        return platform;
    }

    public String getOperationName() {
        return operationName;
    }
}
