package tech.kayys.aljabr.ml.hub;

/**
 * Exception thrown when model hub operations fail.
 * @author bhangun
 */
public class HubException extends RuntimeException {

    public HubException(String message) {
        super(message);
    }

    public HubException(String message, Throwable cause) {
        super(message, cause);
    }
}
