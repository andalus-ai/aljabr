package tech.kayys.aljabr.mlx.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Optional;

public class MlxBinding {
    private static final Linker LINKER = Linker.nativeLinker();
    private static SymbolLookup LOOKUP;

    private static MethodHandle INIT;
    private static MethodHandle ARRAY_FROM_FLOAT;
    private static MethodHandle MATMUL;
    private static MethodHandle EVAL;
    private static MethodHandle GET_DATA;
    private static MethodHandle FREE;

    public static void load(Path libraryPath) {
        LOOKUP = SymbolLookup.libraryLookup(libraryPath, Arena.global());

        INIT = LINKER.downcallHandle(
                LOOKUP.find("aljabr_mlx_init").get(),
                FunctionDescriptor.ofVoid());

        ARRAY_FROM_FLOAT = LINKER.downcallHandle(
                LOOKUP.find("aljabr_mlx_array_from_float").get(),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT));

        MATMUL = LINKER.downcallHandle(
                LOOKUP.find("aljabr_mlx_matmul").get(),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        EVAL = LINKER.downcallHandle(
                LOOKUP.find("aljabr_mlx_eval").get(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        GET_DATA = LINKER.downcallHandle(
                LOOKUP.find("aljabr_mlx_array_get_data").get(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        FREE = LINKER.downcallHandle(
                LOOKUP.find("aljabr_mlx_array_free").get(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    }

    public static void init() {
        try {
            INIT.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static MemorySegment arrayFromFloat(Arena arena, float[] data, long[] shape) {
        MemorySegment dataSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
        MemorySegment shapeSegment = arena.allocateFrom(ValueLayout.JAVA_LONG, shape);
        try {
            return (MemorySegment) ARRAY_FROM_FLOAT.invokeExact(dataSegment, shapeSegment, shape.length);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static MemorySegment matmul(MemorySegment a, MemorySegment b) {
        try {
            return (MemorySegment) MATMUL.invokeExact(a, b);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static void eval(MemorySegment array) {
        try {
            EVAL.invokeExact(array);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static void getData(MemorySegment array, float[] out) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, out);
            GET_DATA.invokeExact(array, outSegment);
            MemorySegment.copy(outSegment, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static void free(MemorySegment array) {
        try {
            FREE.invokeExact(array);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }
}
