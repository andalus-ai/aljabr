package tech.kayys.aljabr.safetensor.bert;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.models.bert.MiniLmL6V2;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

class MiniLmL6V2BundleLifecycleTest {

    @Test
    void closeIsIdempotentAndEncodeAfterCloseIsRejected() {
        CpuBackend backend = new CpuBackend();
        MiniLmL6V2Bundle bundle = new MiniLmL6V2Bundle(
                MiniLmL6V2.create(backend), tokenizer());
        try {
            assertDoesNotThrow(bundle::close);
            assertDoesNotThrow(bundle::close);
            assertThrows(IllegalStateException.class, () -> bundle.encode(List.of()));
            // The bundle owns the encoder, not the caller-provided compute backend.
            Tensor probe = TensorFactory.zeros(backend, 1);
            probe.release();
        } finally {
            bundle.close();
            backend.close();
        }
    }

    private static Tokenizer tokenizer() {
        return new Tokenizer() {
            @Override public long[] encode(String text, EncodeOptions options) { return new long[0]; }
            @Override public String decode(long[] tokens, DecodeOptions options) { return ""; }
            @Override public int vocabSize() { return 30_522; }
            @Override public int bosTokenId() { return 101; }
            @Override public int eosTokenId() { return 102; }
            @Override public int padTokenId() { return 0; }
            @Override public int[] allStopTokenIds() { return new int[] {102}; }
        };
    }
}
