package tech.kayys.aljabr.diffusion.video;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

class VideoFrameSequenceChunkerTest {
    private static final Instant START = Instant.parse("2026-11-27T10:00:00Z");

    private static VideoFrameSequence sequence(int count) {
        List<VideoFrameConditioningData> frames = IntStream.range(0, count)
                .mapToObj(index -> new VideoFrameConditioningData(
                        "video", index * 3, START.plusMillis(index * 40L),
                        new DecodedVideoFrame(1, 1, 1, DecodedVideoFrame.Layout.HWC,
                                new float[] {index})))
                .toList();
        return VideoFrameSequence.of(frames);
    }

    private static List<VideoFrameSequence> collect(Iterable<VideoFrameSequence> chunks) {
        List<VideoFrameSequence> collected = new ArrayList<>();
        chunks.forEach(collected::add);
        return List.copyOf(collected);
    }

    @Test
    void emitsContiguousChunksAndPreservesOriginalProvenance() {
        var chunks = VideoFrameSequenceChunker.chunks(
                sequence(5), 2, VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL);

        var collected = collect(chunks);

        assertEquals(List.of(2, 2, 1), collected.stream()
                .map(VideoFrameSequence::frameCount).toList());
        assertEquals(List.of(0, 6, 12), collected.stream()
                .map(chunk -> chunk.firstFrame().sequenceIndex()).toList());
        assertEquals(START.plusMillis(160), collected.get(2).lastFrame().timestamp());
    }

    @Test
    void dropsIncompleteFinalChunkWhenRequested() {
        var chunks = collect(VideoFrameSequenceChunker.chunks(
                sequence(5), 2, VideoFrameSequenceChunker.TailPolicy.DROP_PARTIAL));

        assertEquals(2, chunks.size());
        assertTrue(chunks.stream().allMatch(chunk -> chunk.frameCount() == 2));
    }

    @Test
    void canConsumeOneChunkAtATime() {
        var iterator = VideoFrameSequenceChunker.chunks(
                sequence(5), 2, VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL).iterator();

        assertTrue(iterator.hasNext());
        assertEquals(0, iterator.next().firstFrame().sequenceIndex());
        assertEquals(6, iterator.next().firstFrame().sequenceIndex());
        assertEquals(12, iterator.next().firstFrame().sequenceIndex());
        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Test
    void eachIteratorStartsAtTheBeginning() {
        var chunks = VideoFrameSequenceChunker.chunks(
                sequence(4), 2, VideoFrameSequenceChunker.TailPolicy.DROP_PARTIAL);

        assertEquals(0, chunks.iterator().next().firstFrame().sequenceIndex());
        assertEquals(0, chunks.iterator().next().firstFrame().sequenceIndex());
    }

    @Test
    void dropsAllFramesWhenNoCompleteChunkCanBeFormed() {
        var chunks = VideoFrameSequenceChunker.chunks(
                sequence(1), 2, VideoFrameSequenceChunker.TailPolicy.DROP_PARTIAL);

        assertFalse(chunks.iterator().hasNext());
    }

    @Test
    void rejectsNonPositiveChunkSize() {
        assertThrows(IllegalArgumentException.class, () -> VideoFrameSequenceChunker.chunks(
                sequence(3), 0, VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL));
    }
}
