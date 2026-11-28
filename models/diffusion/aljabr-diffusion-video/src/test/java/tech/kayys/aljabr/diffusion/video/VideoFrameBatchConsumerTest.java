package tech.kayys.aljabr.diffusion.video;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

class VideoFrameBatchConsumerTest {
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

    private static class TestBatch implements AutoCloseable {
        private final int start;
        private boolean closed;
        private final Exception closeFailure;

        private TestBatch(int start) {
            this(start, null);
        }

        private TestBatch(int start, Exception closeFailure) {
            this.start = start;
            this.closeFailure = closeFailure;
        }

        @Override
        public void close() throws Exception {
            closed = true;
            if (closeFailure != null) {
                throw closeFailure;
            }
        }
    }

    @Test
    void closesEachBatchBeforeRequestingTheNext() throws Exception {
        AtomicReference<TestBatch> previous = new AtomicReference<>();
        List<Integer> processed = new ArrayList<>();
        Iterable<TestBatch> batches = VideoFrameBatchStream.mapChunks(
                sequence(5), 2,
                VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL,
                chunk -> {
                    TestBatch prior = previous.get();
                    if (prior != null) {
                        assertTrue(prior.closed, "previous batch must be closed before packing next");
                    }
                    TestBatch next = new TestBatch(chunk.firstFrame().sequenceIndex());
                    previous.set(next);
                    return next;
                });

        long count = VideoFrameBatchConsumer.consume(batches, batch -> {
            assertFalse(batch.closed);
            processed.add(batch.start);
        });

        assertEquals(3, count);
        assertEquals(List.of(0, 2, 4), processed);
        assertTrue(previous.get().closed);
    }

    @Test
    void closesCurrentBatchAndStopsWhenProcessingFails() {
        TestBatch batch = new TestBatch(0);
        IllegalStateException failure = new IllegalStateException("consumer failed");

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> VideoFrameBatchConsumer.consume(
                        List.of(batch), ignored -> {
                            throw failure;
                        }));

        assertSame(failure, thrown);
        assertTrue(batch.closed);
    }

    @Test
    void attachesCloseFailureToProcessingFailure() {
        IOException closeFailure = new IOException("close failed");
        TestBatch batch = new TestBatch(0, closeFailure);
        IllegalStateException processingFailure = new IllegalStateException("processing failed");

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> VideoFrameBatchConsumer.consume(
                        List.of(batch), ignored -> {
                            throw processingFailure;
                        }));

        assertSame(processingFailure, thrown);
        assertArrayEquals(new Throwable[] {closeFailure}, thrown.getSuppressed());
        assertTrue(batch.closed);
    }

    @Test
    void propagatesCloseFailureAndDoesNotRequestAnotherBatch() {
        IOException closeFailure = new IOException("close failed");
        TestBatch first = new TestBatch(0, closeFailure);
        AtomicReference<Integer> requested = new AtomicReference<>(0);
        Iterable<TestBatch> source = () -> new java.util.Iterator<>() {
            private int index;

            @Override
            public boolean hasNext() {
                return index < 2;
            }

            @Override
            public TestBatch next() {
                requested.set(requested.get() + 1);
                index++;
                return index == 1 ? first : new TestBatch(1);
            }
        };

        IOException thrown = assertThrows(
                IOException.class,
                () -> VideoFrameBatchConsumer.consume(source, ignored -> {
                }));

        assertSame(closeFailure, thrown);
        assertTrue(first.closed);
        assertEquals(1, requested.get());
    }


    @Test
    void rejectsNullSourceProcessorAndBatch() {
        assertThrows(NullPointerException.class,
                () -> VideoFrameBatchConsumer.consume(null, ignored -> {
                }));
        assertThrows(NullPointerException.class,
                () -> VideoFrameBatchConsumer.consume(List.of(), null));
        assertThrows(NullPointerException.class,
                () -> VideoFrameBatchConsumer.consume(
                        Arrays.asList((TestBatch) null), ignored -> {
                        }));
    }

    @Test
    void reportsProgressOnlyAfterBatchHasBeenClosed() throws Exception {
        List<VideoBatchProgress> progress = new ArrayList<>();
        TestBatch first = new TestBatch(0);
        TestBatch second = new TestBatch(1);
        VideoBatchProcessingContext context = VideoBatchProcessingContext.create(
                () -> false, OptionalLong.of(2), progress::add);

        long count = VideoFrameBatchConsumer.consume(
                List.of(first, second), ignored -> { }, context);

        assertEquals(2, count);
        assertTrue(first.closed);
        assertTrue(second.closed);
        assertEquals(List.of(
                new VideoBatchProgress(1, OptionalLong.of(2)),
                new VideoBatchProgress(2, OptionalLong.of(2))), progress);
    }

    @Test
    void cancellationStopsBeforeRequestingOrPackingTheNextBatch() throws Exception {
        AtomicBoolean cancelled = new AtomicBoolean();
        AtomicReference<Integer> packed = new AtomicReference<>(0);
        List<VideoBatchProgress> progress = new ArrayList<>();
        VideoBatchProcessingContext context = VideoBatchProcessingContext.create(
                cancelled::get, OptionalLong.empty(), update -> {
                    progress.add(update);
                    cancelled.set(true);
                });
        Iterable<TestBatch> source = VideoFrameBatchStream.mapChunks(
                sequence(4), 2, VideoFrameSequenceChunker.TailPolicy.INCLUDE_PARTIAL,
                chunk -> {
                    packed.set(packed.get() + 1);
                    return new TestBatch(chunk.firstFrame().sequenceIndex());
                });

        long count = VideoFrameBatchConsumer.consume(source, batch -> { }, context);

        assertEquals(1, count);
        assertEquals(1, packed.get(), "cancellation must be checked before requesting the next chunk");
        assertEquals(List.of(new VideoBatchProgress(1, OptionalLong.empty())), progress);
    }

    @Test
    void cancellationDuringProcessingClosesCurrentBatchWithoutProgress() {
        AtomicBoolean cancelled = new AtomicBoolean();
        List<VideoBatchProgress> progress = new ArrayList<>();
        TestBatch batch = new TestBatch(0);
        VideoBatchProcessingContext context = VideoBatchProcessingContext.create(
                cancelled::get, OptionalLong.of(1), progress::add);

        assertThrows(CancellationException.class,
                () -> VideoFrameBatchConsumer.consume(
                        List.of(batch), ignored -> cancelled.set(true), context));

        assertTrue(batch.closed);
        assertTrue(progress.isEmpty());
    }

    @Test
    void rejectsInvalidKnownTotalAndNullContext() {
        assertThrows(IllegalArgumentException.class,
                () -> VideoBatchProcessingContext.create(
                        () -> false, OptionalLong.of(-1), ignored -> { }));
        assertThrows(NullPointerException.class,
                () -> VideoFrameBatchConsumer.consume(List.of(), ignored -> { }, null));
    }

    @Test
    void returnsZeroForAnEmptySource() throws Exception {
        assertEquals(0, VideoFrameBatchConsumer.consume(List.of(), ignored -> {
        }));
    }
}
