package tech.kayys.aljabr.diffusion.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.Map;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationException;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerPromptConditioningPreprocessorTest {

    @Test
    void resolvesAndTokenizesPromptWithConfiguredBoundaryTokens() {
        var tokenizer = new StubTokenizer();
        var adapter = new TokenizerPromptConditioningPreprocessor(
                tokenizer,
                input -> "a diffusion prompt",
                new PromptTokenizationOptions(true, true));
        var input = new DiffusionConditioningInput(
                "positive", DiffusionConditioningInput.Modality.PROMPT, "prompt-store:positive");

        var result = (PromptTokenizationData) adapter.prepare(
                input, DiffusionConditioningPreparationContext.defaults());

        assertEquals("a diffusion prompt", tokenizer.encodedText);
        assertTrue(tokenizer.options.addBos);
        assertTrue(tokenizer.options.addEos);
        assertArrayEquals(new long[]{1, 2, 3}, result.tokenIds());
        assertEquals("positive", result.inputId());
        assertEquals(DiffusionConditioningInput.Modality.PROMPT, result.modality());
    }

    @Test
    void tokenIdsAreDefensivelyCopied() {
        long[] source = {4, 5};
        var result = new PromptTokenizationData("prompt", source);
        source[0] = 99;
        long[] returned = result.tokenIds();
        returned[1] = 88;

        assertArrayEquals(new long[]{4, 5}, result.tokenIds());
    }

    @Test
    void rejectsNonPromptModality() {
        var adapter = new TokenizerPromptConditioningPreprocessor(
                new StubTokenizer(), input -> "unused");
        var image = new DiffusionConditioningInput(
                "image", DiffusionConditioningInput.Modality.IMAGE, "asset:image");

        assertFalse(adapter.supports(image));
        assertThrows(DiffusionConditioningPreparationException.class,
                () -> adapter.prepare(image, DiffusionConditioningPreparationContext.defaults()));
    }

    @Test
    void wrapsSourceResolutionFailuresWithInputProvenance() {
        var adapter = new TokenizerPromptConditioningPreprocessor(
                new StubTokenizer(), input -> { throw new IllegalStateException("missing prompt"); });
        var input = new DiffusionConditioningInput(
                "negative", DiffusionConditioningInput.Modality.PROMPT, "prompt-store:negative");

        var failure = assertThrows(DiffusionConditioningPreparationException.class,
                () -> adapter.prepare(input, DiffusionConditioningPreparationContext.defaults()));

        assertEquals("negative", failure.inputId());
        assertEquals(DiffusionConditioningInput.Modality.PROMPT, failure.modality());
        assertInstanceOf(IllegalStateException.class, failure.getCause());
    }

    private static final class StubTokenizer implements Tokenizer {
        private String encodedText;
        private EncodeOptions options;

        @Override
        public long[] encode(String text, EncodeOptions options) {
            this.encodedText = text;
            this.options = options;
            return new long[]{1, 2, 3};
        }

        @Override public String decode(long[] tokens, DecodeOptions options) { return ""; }
        @Override public int vocabSize() { return 16; }
        @Override public int bosTokenId() { return 1; }
        @Override public int eosTokenId() { return 2; }
        @Override public int padTokenId() { return 0; }
        @Override public int[] allStopTokenIds() { return new int[]{2}; }
    }
}
