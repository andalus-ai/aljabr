package tech.kayys.aljabr.gguf.core;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class GGUFConstants {
    private GGUFConstants() {}

    public static final int GGML_TYPE_F32  = 0;
    public static final int GGML_TYPE_F16  = 1;
    public static final int GGML_TYPE_Q4_0 = 2;
    public static final int GGML_TYPE_Q4_1 = 3;
    public static final int GGML_TYPE_Q5_0 = 6;
    public static final int GGML_TYPE_Q5_1 = 7;
    public static final int GGML_TYPE_Q8_0 = 8;
    public static final int GGML_TYPE_Q8_1 = 9;
    public static final int GGML_TYPE_Q2_K = 10;
    public static final int GGML_TYPE_Q3_K = 11;
    public static final int GGML_TYPE_Q4_K = 12;
    public static final int GGML_TYPE_Q5_K = 13;
    public static final int GGML_TYPE_Q6_K = 14;
    public static final int GGML_TYPE_Q8_K = 15;
    public static final int GGML_TYPE_TQ1_0 = 34;
    public static final int GGML_TYPE_TQ2_0 = 35;
    public static final int GGML_TYPE_MXFP4 = 39;
    public static final int GGML_TYPE_NVFP4 = 40;
    public static final int GGML_TYPE_Q1_0 = 41;

    public static long tensorBytes(int type, long n) {
        return GgmlType.fromId(type).bytesFor(n);
    }
}
