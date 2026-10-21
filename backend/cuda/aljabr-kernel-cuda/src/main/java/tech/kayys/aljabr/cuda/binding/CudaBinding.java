package tech.kayys.aljabr.cuda.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM-based binding directly to NVIDIA CUDA Runtime (libcudart) and cuBLAS
 * (libcublas).
 */
public class CudaBinding {

    private static final Logger LOG = Logger.getLogger(CudaBinding.class);
    private static volatile CudaBinding instance;

    // CUDA Runtime API
    private static final String FN_CUDA_MALLOC = "cudaMalloc";
    private static final String FN_CUDA_MALLOC_MANAGED = "cudaMallocManaged";
    private static final String FN_CUDA_FREE = "cudaFree";
    private static final String FN_CUDA_MEMCPY = "cudaMemcpy";
    private static final String FN_CUDA_GET_DEVICE_COUNT = "cudaGetDeviceCount";
    private static final String FN_CUDA_SET_DEVICE = "cudaSetDevice";

    // cuBLAS API
    private static final String FN_CUBLAS_CREATE = "cublasCreate_v2";
    private static final String FN_CUBLAS_DESTROY = "cublasDestroy_v2";
    private static final String FN_CUBLAS_SGEMM = "cublasSgemm_v2";

    // Custom Kernels
    private static final String FN_SILU = "aljabr_cuda_silu";
    private static final String FN_GELU = "aljabr_cuda_gelu";
    private static final String FN_LAYERNORM = "aljabr_cuda_layernorm";
    private static final String FN_LAYERNORM_ROWS = "aljabr_cuda_layernorm_rows";
    private static final String FN_SOFTMAX = "aljabr_cuda_softmax";

    // cudaMemcpyKind
    public static final int cudaMemcpyHostToHost = 0;
    public static final int cudaMemcpyHostToDevice = 1;
    public static final int cudaMemcpyDeviceToHost = 2;
    public static final int cudaMemcpyDeviceToDevice = 3;

    // cublasOperation_t
    public static final int CUBLAS_OP_N = 0;
    public static final int CUBLAS_OP_T = 1;
    public static final int CUBLAS_OP_C = 2;

    private final SymbolLookup cudartLookup;
    private final SymbolLookup cublasLookup;
    private final SymbolLookup customLookup;
    private final Map<String, MethodHandle> handles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private CudaBinding(SymbolLookup cudart, SymbolLookup cublas, SymbolLookup custom) {
        this.cudartLookup = cudart;
        this.cublasLookup = cublas;
        this.customLookup = custom;
        this.nativeAvailable = (cudart != null && cublas != null);
        if (nativeAvailable) {
            bindAll();
        }
    }

    public static boolean initialize() {
        if (instance != null)
            return instance.nativeAvailable;
        try {
            // Attempt to load system libraries
            System.loadLibrary("cudart");
            System.loadLibrary("cublas");

            SymbolLookup custom = null;
            try {
                System.loadLibrary("aljabr_cuda");
                custom = SymbolLookup.loaderLookup();
            } catch (Throwable e) {
                LOG.warn("CudaBinding: Custom libaljabr_cuda.so not found, custom kernels disabled.");
            }

            SymbolLookup lk = SymbolLookup.loaderLookup();
            instance = new CudaBinding(lk, lk, custom != null ? custom : lk);
            LOG.info("CudaBinding loaded libcudart and libcublas via system library path.");
            return true;
        } catch (Throwable e) {
            LOG.warnf("CudaBinding: system libraries not found (%s). Attempting explicit paths.", e.getMessage());
            try {
                SymbolLookup cudart = SymbolLookup.libraryLookup(Path.of("/usr/local/cuda/lib64/libcudart.so"),
                        Arena.global());
                SymbolLookup cublas = SymbolLookup.libraryLookup(Path.of("/usr/local/cuda/lib64/libcublas.so"),
                        Arena.global());
                instance = new CudaBinding(cudart, cublas, cudart); // fallback for custom is empty/cudart
                LOG.info("CudaBinding loaded libcudart and libcublas via explicit paths.");
                return true;
            } catch (Throwable e2) {
                LOG.warnf("CudaBinding: explicit libraries not found (%s). CPU fallback active.", e2.getMessage());
                instance = new CudaBinding(null, null, null);
                return false;
            }
        }
    }

    public static void initializeFallback() {
        if (instance != null)
            return;
        instance = new CudaBinding(null, null, null);
        LOG.info("CudaBinding: CPU fallback mode");
    }

    public static CudaBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("CudaBinding not initialized — call initialize() first");
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public int cudaGetDeviceCount() {
        if (!nativeAvailable)
            return 0;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment countPtr = a.allocate(ValueLayout.JAVA_INT);
            int status = (int) invoke(FN_CUDA_GET_DEVICE_COUNT, countPtr);
            if (status != 0)
                return 0;
            return countPtr.get(ValueLayout.JAVA_INT, 0);
        }
    }

    public int cudaSetDevice(int deviceId) {
        if (!nativeAvailable)
            return 0;
        return (int) invoke(FN_CUDA_SET_DEVICE, deviceId);
    }

    public MemorySegment cudaMalloc(long bytes) {
        if (!nativeAvailable) {
            return Arena.ofAuto().allocate(bytes, 64);
        }
        try (Arena a = Arena.ofConfined()) {
            MemorySegment ptrPtr = a.allocate(ValueLayout.ADDRESS);
            int status = (int) invoke(FN_CUDA_MALLOC, ptrPtr, bytes);
            if (status != 0)
                throw new RuntimeException("cudaMalloc failed with code " + status);
            return ptrPtr.get(ValueLayout.ADDRESS, 0);
        }
    }

    public MemorySegment cudaMallocManaged(long bytes, int flags) {
        if (!nativeAvailable) {
            return Arena.ofAuto().allocate(bytes, 64);
        }
        try (Arena a = Arena.ofConfined()) {
            MemorySegment ptrPtr = a.allocate(ValueLayout.ADDRESS);
            int status = (int) invoke(FN_CUDA_MALLOC_MANAGED, ptrPtr, bytes, flags);
            if (status != 0)
                throw new RuntimeException("cudaMallocManaged failed with code " + status);
            return ptrPtr.get(ValueLayout.ADDRESS, 0);
        }
    }

    public void cudaFree(MemorySegment ptr) {
        if (!nativeAvailable)
            return;
        invoke(FN_CUDA_FREE, ptr);
    }

    public void cudaMemcpy(MemorySegment dst, MemorySegment src, long count, int kind) {
        if (!nativeAvailable) {
            dst.copyFrom(src);
            return;
        }
        int status = (int) invoke(FN_CUDA_MEMCPY, dst, src, count, kind);
        if (status != 0)
            throw new RuntimeException("cudaMemcpy failed with code " + status);
    }

    public MemorySegment cublasCreate() {
        if (!nativeAvailable)
            return MemorySegment.NULL;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment handlePtr = a.allocate(ValueLayout.ADDRESS);
            int status = (int) invoke(FN_CUBLAS_CREATE, handlePtr);
            if (status != 0)
                throw new RuntimeException("cublasCreate failed with code " + status);
            return handlePtr.get(ValueLayout.ADDRESS, 0);
        }
    }

    public void cublasDestroy(MemorySegment handle) {
        if (!nativeAvailable || handle.equals(MemorySegment.NULL))
            return;
        invoke(FN_CUBLAS_DESTROY, handle);
    }

    public int cublasSgemm(MemorySegment handle, int transa, int transb,
            int m, int n, int k,
            float alpha,
            MemorySegment A, int lda,
            MemorySegment B, int ldb,
            float beta,
            MemorySegment C, int ldc) {
        if (!nativeAvailable)
            throw new UnsupportedOperationException("cuBLAS not available");

        try (Arena a = Arena.ofConfined()) {
            MemorySegment alphaPtr = a.allocateFrom(ValueLayout.JAVA_FLOAT, alpha);
            MemorySegment betaPtr = a.allocateFrom(ValueLayout.JAVA_FLOAT, beta);

            return (int) invoke(FN_CUBLAS_SGEMM, handle, transa, transb,
                    m, n, k, alphaPtr, A, lda, B, ldb, betaPtr, C, ldc);
        }
    }

    public int silu(MemorySegment out, MemorySegment x, int N) {
        if (!nativeAvailable || !handles.containsKey(FN_SILU))
            return -1;
        return (int) invoke(FN_SILU, out, x, N);
    }

    public int gelu(MemorySegment out, MemorySegment x, int N) {
        if (!nativeAvailable || !handles.containsKey(FN_GELU))
            return -1;
        return (int) invoke(FN_GELU, out, x, N);
    }

    public int layerNorm(MemorySegment out, MemorySegment x, MemorySegment weight, MemorySegment bias, int N,
            float eps) {
        if (!nativeAvailable || !handles.containsKey(FN_LAYERNORM))
            return -1;
        return (int) invoke(FN_LAYERNORM, out, x, weight, bias, N, eps);
    }

    public int layerNormRows(MemorySegment out, MemorySegment x, MemorySegment weight, MemorySegment bias, int rows,
            int N, float eps) {
        if (!nativeAvailable || !handles.containsKey(FN_LAYERNORM_ROWS))
            return -1;
        return (int) invoke(FN_LAYERNORM_ROWS, out, x, weight, bias, rows, N, eps);
    }

    public int softmax(MemorySegment out, MemorySegment x, int rows, int N) {
        if (!nativeAvailable || !handles.containsKey(FN_SOFTMAX))
            return -1;
        return (int) invoke(FN_SOFTMAX, out, x, rows, N);
    }

    // ── FFM binding ───────────────────────────────────────────────────────────

    private void bindAll() {
        bind(cudartLookup, FN_CUDA_MALLOC, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        bind(cudartLookup, FN_CUDA_MALLOC_MANAGED, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
        bind(cudartLookup, FN_CUDA_FREE, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS));
        bind(cudartLookup, FN_CUDA_MEMCPY, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
        bind(cudartLookup, FN_CUDA_GET_DEVICE_COUNT, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS));
        bind(cudartLookup, FN_CUDA_SET_DEVICE, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        bind(cublasLookup, FN_CUBLAS_CREATE, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS));
        bind(cublasLookup, FN_CUBLAS_DESTROY, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS));
        bind(cublasLookup, FN_CUBLAS_SGEMM, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        if (customLookup != null) {
            bind(customLookup, FN_SILU, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            bind(customLookup, FN_GELU, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            bind(customLookup, FN_LAYERNORM, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));
            bind(customLookup, FN_LAYERNORM_ROWS, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));
            bind(customLookup, FN_SOFTMAX, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        }
    }

    private void bind(SymbolLookup lookup, String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> sym = lookup.find(name);
        if (sym.isPresent()) {
            handles.put(name, Linker.nativeLinker().downcallHandle(sym.get(), descriptor));
            LOG.debugf("CudaBinding: bound %s", name);
        } else {
            LOG.warnf("CudaBinding: symbol not found — %s", name);
        }
    }

    private Object invoke(String name, Object... args) {
        MethodHandle mh = handles.get(name);
        if (mh == null)
            throw new IllegalStateException("Unbound: " + name);
        try {
            return mh.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new RuntimeException("CudaBinding." + name + " failed", t);
        }
    }

    public String deviceName(int deviceId) {
        return "CUDA Device " + deviceId; // Simplified as cudaGetDeviceProperties requires struct mapping
    }
}
