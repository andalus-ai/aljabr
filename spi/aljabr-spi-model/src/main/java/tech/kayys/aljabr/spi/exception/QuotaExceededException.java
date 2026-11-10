package tech.kayys.aljabr.spi.exception;

import tech.kayys.aljabr.error.ErrorCode;
/**
 * 
 * Exception or error class for handling quotaexceeded.
 *
 * @author bhangun
 * @since 0.1.0
 */
public class QuotaExceededException extends InferenceException {

    public QuotaExceededException(String message) {
        super(ErrorCode.QUOTA_EXCEEDED, message);
    }

    public QuotaExceededException(ErrorCode errorCode, String message) {
        super(errorCode, message);
    }
}
