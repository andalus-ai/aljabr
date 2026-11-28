package tech.kayys.aljabr.diffusion.video;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

class VideoFrameBatchStreamTest {
    private static final Instant START = Instant.parse("2026-11-27T10:00:00Z");

    private static VideoFrameSequence sequence(int count) {
        List<VideoFrameConditioningData> frames = IntStream.range(0, count)
                .mapToObj(index -> new VideoFrameConditioningData(
                        "video", index, START.plusMillis(index * 40L),
                        new DecodedVideoFrame(1, 1, 1, DecodedVideoFrame.Layout.HWC,
                                new float[] {index})))
                .toList();
        return VideoFrameSequence.of(frames);
    }

    @Test
    void mapsChunksOnlyWhenConsumerRequestsNextValue() {
        AtomicInteger packed = new AtomicInteger();
        Iterable<Integer> batches = VideoFrameBatchStream.mapChunks(
                sequence(5), 2,
                VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL,
                chunk -> {
                    packed.incrementAndGet();
                    return chunk.frameCount();
                });

        assertEquals(0, packed.get());
        var iterator = batches.iterator();
        assertEquals(0, packed.get());
        assertTrue(iterator.hasNext());
        assertEquals(0, packed.get());

        assertEquals(2, iterator.next());
        assertEquals(1, packed.get());
        assertEquals(2, iterator.next());
        assertEquals(2, packed.get());
        assertEquals(1, iterator.next());
        assertEquals(3, packed.get());
        assertFalse(iterator.hasNext());
        assertEquals(3, packed.get());
    }

    @Test
    void preservesSourceChunkOrderAndTailPolicy() {
        List<Integer> starts = new ArrayList<>();
        Iterable<Integer> batches = VideoFrameBatchStream.mapChunks(
                sequence(5), 2,
                VideoFrameSequenceChunker.TailPolicy.DROP_PARTIAL,
                chunk -> {
                    starts.add(chunk.firstFrame().sequenceIndex());
                    return chunk.frameCount();
                });

        assertEquals(List.of(2, 2), collect(batches));
        assertEquals(List.of(0, 2), starts);
    }

    @Test
    void createsIndependentRepeatableIterators() {
        AtomicInteger packed = new AtomicInteger();
        Iterable<Integer> batches = VideoFrameBatchStream.mapChunks(
                sequence(4), 2,
                VideoFrameSequenceChunker.TailPolicy.DROP_PARTIAL,
                chunk -> packed.incrementAndGet());

        assertEquals(List.of(1, 2), collect(batches));
        assertEquals(List.of(3, 4), collect(batches));
    }

    @Test
    void propagatesPackerFailuresWithoutHidingCause() {
        IllegalStateException failure = new IllegalStateException("packing failed");
        Iterable<Object> batches = VideoFrameBatchStream.mapChunks(
                sequence(2), 1,
                VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL,
                chunk -> {
                    throw failure;
                });

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class, () -> batches.iterator().next());
        assertSame(failure, thrown);
    }

    @Test
    void rejectsNullBatchValues() {
        Iterable<Object> batches = VideoFrameBatchStream.mapChunks(
                sequence(1), 1,
                VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL,
                chunk -> null);

        NullPointerException thrown = assertThrows(
                NullPointerException.class, () -> batches.iterator().next());
        assertTrue(thrown.getMessage().contains("batchPacker"));
    }

    @Test
    void rejectsNullPacker() {
        assertThrows(NullPointerException.class, () -> VideoFrameBatchStream.mapChunks(
                sequence(1), 1,
                VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL,
                null));
    }

    private static <T> List<T> collect(Iterable<T> values) {
        List<T> collected = new ArrayList<>();
        values.forEach(collected::add);
        return List.copyOf(collected);
    }
}
