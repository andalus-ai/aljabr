package tech.kayys.aljabr.safetensor.bert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;

/**
 * Opt-in smoke test against an unmodified Hugging Face all-MiniLM-L6-v2 snapshot.
 * Set ALJABR_MINILM_CHECKPOINT to a local snapshot directory.
 * This verifies real loading and embedding invariants, not numerical parity
 * against Sentence Transformers reference vectors.
 */
class MiniLmL6V2OfficialCheckpointIT {
    @Test
    void loadsOfficialCheckpointAndProducesNormalized384DimensionalEmbeddings() throws Exception {
        String checkpoint = System.getenv("ALJABR_MINILM_CHECKPOINT");
        assumeTrue(checkpoint != null && !checkpoint.isBlank(),
                "Set ALJABR_MINILM_CHECKPOINT to run this integration test");

        CpuBackend backend = new CpuBackend();
        try (MiniLmL6V2Bundle bundle = new MiniLmL6V2BundleLoader(
                new ObjectMapper(), new BertSafetensorLoader(new SafetensorShardLoader()))
                .load(Path.of(checkpoint), backend)) {
            List<String> sentences = List.of(
                    "The quick brown fox jumps over the lazy dog.",
                    "A completely different sentence about numerical computing.");
            float[][] embeddings = bundle.encode(sentences);
            assertEquals(2, embeddings.length);
            for (float[] embedding : embeddings) {
                assertEquals(384, embedding.length);
                double squaredNorm = 0.0;
                for (float value : embedding) {
                    assertTrue(Float.isFinite(value), "Embedding contains a non-finite value");
                    squaredNorm += (double) value * value;
                }
                assertEquals(1.0, Math.sqrt(squaredNorm), 1.0e-3,
                        "Sentence embedding should be L2 normalized");
            }
            float[][] repeated = bundle.encode(sentences);
            for (int row = 0; row < embeddings.length; row++) {
                for (int column = 0; column < embeddings[row].length; column++) {
                    assertEquals(embeddings[row][column], repeated[row][column], 1.0e-5f,
                            "Repeated inference should be deterministic");
                }
            }
        } finally {
            backend.close();
        }
    }
}
