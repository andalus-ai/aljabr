package tech.kayys.aljabr.linalg;

/** Raised when a matrix cannot be factored at the requested numerical tolerance. */
public final class SingularMatrixException extends ArithmeticException {
    public SingularMatrixException(String message) { super(message); }
}
