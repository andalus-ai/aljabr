package tech.kayys.aljabr.llama;

/**
 * Minimal native CPU-feature smoke test.
 *
 * Use jextract's public downcall wrapper rather than reaching into its private
 * implementation details (MethodHandles may change between jextract versions).
 */
public final class Main {
    private Main() {
    }

    public static void main(String[] args) throws Throwable {
        int hasAvx = llama_h.ggml_cpu_has_avx();
        System.out.println("CPU has AVX: " + (hasAvx != 0));
    }
}
