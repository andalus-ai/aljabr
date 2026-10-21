package tech.kayys.aljabr.plugin.kernel;

/**
 * SPI — result wrapper returned by kernel execution.
 */
public final class KernelResult<T> {
    private final T value;
    private final boolean success;

    private KernelResult(T value, boolean success) {
        this.value = value;
        this.success = success;
    }

    public static <T> KernelResult<T> success(T value) {
        return new KernelResult<>(value, true);
    }

    public static <T> KernelResult<T> failure() {
        return new KernelResult<>(null, false);
    }

    public T getValue() {
        return value;
    }

    public boolean isSuccess() {
        return success;
    }
}
