package tech.kayys.aljabr.diffusion.video;
import static org.junit.jupiter.api.Assertions.*;
import java.time.Instant;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningPreparationContext;
class VideoFrameConditioningPreprocessorTest {
    private static final Instant TIME = Instant.parse("2026-11-27T10:15:30.250Z");
    private static DiffusionConditioningInput input(int index, String time) {
        return new DiffusionConditioningInput("frame-" + index, DiffusionConditioningInput.Modality.VIDEO_FRAME,
                "opaque:frame/" + index, index, Map.of(VideoFrameConditioningPreprocessor.TIMESTAMP_METADATA_KEY, time));
    }
    private static DecodedVideoFrame frame(float... pixels) {
        return new DecodedVideoFrame(1, 1, pixels.length, DecodedVideoFrame.Layout.HWC, pixels);
    }
    @Test void preservesTemporalProvenanceAndPixels() {
        var p = new VideoFrameConditioningPreprocessor(ignored -> frame(0.1f, 0.2f, 0.3f));
        var data = (VideoFrameConditioningData) p.prepare(input(4, TIME.toString()), DiffusionConditioningPreparationContext.defaults());
        assertEquals("frame-4", data.inputId()); assertEquals(4, data.sequenceIndex()); assertEquals(TIME, data.timestamp());
        assertEquals(1, data.width()); assertEquals(3, data.channels());
        assertArrayEquals(new float[] {0.1f, 0.2f, 0.3f}, data.pixels());
    }
    @Test void delegatesTransformAndContext() {
        var context = new DiffusionConditioningPreparationContext(Map.of("resize", "512"));
        var expected = frame(1f, 2f);
        var p = new VideoFrameConditioningPreprocessor(ignored -> frame(0f), (in, decoded, ctx) -> {
            assertEquals("frame-2", in.id()); assertSame(context, ctx); assertEquals(0f, decoded.pixels()[0]); return expected;
        });
        var data = (VideoFrameConditioningData) p.prepare(input(2, TIME.toString()), context);
        assertArrayEquals(new float[] {1f, 2f}, data.pixels());
    }
    @Test void rejectsMissingAndInvalidTimestamps() {
        var p = new VideoFrameConditioningPreprocessor(ignored -> frame(0f));
        var missing = new DiffusionConditioningInput("missing", DiffusionConditioningInput.Modality.VIDEO_FRAME, "opaque", 0, Map.of());
        assertThrows(RuntimeException.class, () -> p.prepare(missing, DiffusionConditioningPreparationContext.defaults()));
        assertThrows(RuntimeException.class, () -> p.prepare(input(0, "invalid"), DiffusionConditioningPreparationContext.defaults()));
    }
    @Test void rejectsOtherModalities() {
        var p = new VideoFrameConditioningPreprocessor(ignored -> frame(0f));
        var image = new DiffusionConditioningInput("image", DiffusionConditioningInput.Modality.IMAGE, "opaque:image");
        assertFalse(p.supports(image));
        assertThrows(RuntimeException.class, () -> p.prepare(image, DiffusionConditioningPreparationContext.defaults()));
    }
    @Test void decodedPixelsAreDefensivelyCopiedAndFinite() {
        float[] pixels = {1f, 2f}; var decoded = frame(pixels); pixels[0] = 9f;
        assertArrayEquals(new float[] {1f, 2f}, decoded.pixels());
        float[] exposed = decoded.pixels(); exposed[0] = 8f;
        assertArrayEquals(new float[] {1f, 2f}, decoded.pixels());
        assertThrows(IllegalArgumentException.class, () -> frame(Float.NaN));
    }
    @Test void rejectsMalformedFrameShape() {
        assertThrows(IllegalArgumentException.class, () -> new DecodedVideoFrame(2, 2, 3, DecodedVideoFrame.Layout.HWC, new float[3]));
    }
}
