package tech.kayys.aljabr.spi.exception;

import tech.kayys.aljabr.error.ErrorCode;
import java.util.HashMap;
import java.util.Map;

/**
 * Base exception for inference-related errors in the Aljabr math foundation.
 * Framework-agnostic: works with both conventional and reactive (Mutiny/RxJava)
 * calling code.
 * @author bhangun
 */
public class InferenceException extends RuntimeException {

    private final ErrorCode errorCode;
    private final Map<String, String> context = new HashMap<>();

    public InferenceException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public InferenceException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public InferenceException(String message) {
        super(message);
        this.errorCode = null;
    }

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = null;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    public InferenceException addContext(String key, String value) {
        context.put(key, value);
        return this;
    }

    public InferenceException addContext(String key, int value) {
        context.put(key, String.valueOf(value));
        return this;
    }

    public InferenceException addContext(String key, Object value) {
        context.put(key, value != null ? value.toString() : "null");
        return this;
    }

    public Map<String, String> getContext() {
        return new HashMap<>(context);
    }
}
