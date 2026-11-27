package tech.kayys.aljabr.backend.cpu;

/**
 * IEEE-754 conversion helpers for scalar CPU kernels.
 *
 * <p>Conversions are kept in one place so kernels do not accidentally treat
 * the 16-bit storage representations as integer arithmetic.</p>
 */
final class CpuFloatingPointConversions {

    private CpuFloatingPointConversions() {
    }

    static float halfToFloat(short value) {
        int half = Short.toUnsignedInt(value);
        int sign = (half & 0x8000) << 16;
        int exponent = (half >>> 10) & 0x1f;
        int mantissa = half & 0x03ff;
        int bits;
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                int adjustedExponent = -14;
                while ((mantissa & 0x0400) == 0) {
                    mantissa <<= 1;
                    adjustedExponent--;
                }
                mantissa &= 0x03ff;
                bits = sign | ((adjustedExponent + 127) << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1f) {
            bits = sign | 0x7f800000 | (mantissa << 13);
        } else {
            bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
        }
        return Float.intBitsToFloat(bits);
    }

    static short floatToHalf(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = (bits >>> 23) & 0xff;
        int mantissa = bits & 0x7fffff;

        if (exponent == 0xff) {
            int payload = mantissa == 0 ? 0 : Math.max(1, mantissa >>> 13);
            return (short) (sign | 0x7c00 | payload);
        }

        int halfExponent = exponent - 127 + 15;
        if (halfExponent >= 0x1f) {
            return (short) (sign | 0x7c00);
        }
        if (halfExponent <= 0) {
            if (halfExponent < -10) {
                return (short) sign;
            }
            mantissa |= 0x800000;
            int shift = 14 - halfExponent;
            int rounded = mantissa + (1 << (shift - 1)) - 1 + ((mantissa >>> shift) & 1);
            return (short) (sign | (rounded >>> shift));
        }

        int rounded = mantissa + 0x0fff + ((mantissa >>> 13) & 1);
        if ((rounded & 0x800000) != 0) {
            rounded = 0;
            halfExponent++;
            if (halfExponent >= 0x1f) {
                return (short) (sign | 0x7c00);
            }
        }
        return (short) (sign | (halfExponent << 10) | (rounded >>> 13));
    }

    static float bfloat16ToFloat(short value) {
        return Float.intBitsToFloat(Short.toUnsignedInt(value) << 16);
    }

    static short floatToBfloat16(float value) {
        int bits = Float.floatToRawIntBits(value);
        // Preserve NaN payloads rather than rounding a NaN into infinity.
        if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x007fffff) != 0) {
            return (short) ((bits >>> 16) | 0x0040);
        }
        int roundingBias = 0x7fff + ((bits >>> 16) & 1);
        return (short) ((bits + roundingBias) >>> 16);
    }
}
