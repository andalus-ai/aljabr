# Aljabr Diffusion Video Conditioning

Optional reusable per-frame conditioning integration for the generic diffusion conditioning SPI.

## Frame preparation

- One VIDEO_FRAME input produces one VideoFrameConditioningData result.
- Each input carries a non-negative sequenceIndex and required timestamp metadata parsed as an ISO-8601 instant.
- Decoding and preprocessing are injected. Codec, source resolution, temporal sampling, resize/crop, normalization, and color-space policy remain application-owned.
- Decoded pixels are immutable heap-owned float arrays with explicit HWC/CHW layout and finite-value validation.

## Lazy sequence chunking

`VideoFrameSequenceChunker` exposes contiguous frame chunks through a repeatable, pull-based `Iterable`.

- Each iterator creates one non-owning sequence view when the consumer requests it; it does not materialize a list of all chunks.
- `INCLUDE_PARTIAL` retains the final non-empty short chunk; `DROP_PARTIAL` emits only complete chunks.
- Each iterator starts independently at the first source frame, and chunks preserve original frame objects, indices, and timestamps.
- Pull-based iteration lets a downstream consumer control how quickly it requests more work. It does not create threads, schedule execution, or provide asynchronous backpressure.
- Chunks do not own or close frame resources; the original prepared conditioning set remains the owner and must stay open while chunks are consumed.

## Sequence view

VideoFrameSequence groups prepared frame results into a validated temporal view.

- Caller order is retained; the sequence does not sort or resample.
- Sequence indices and presentation timestamps must both be strictly increasing.
- Sparse sequence indices are allowed, supporting intentionally sampled frames.
- Duration is elapsed presentation time between the first and last timestamps, not an estimated frame rate.
- The sequence is non-owning. PreparedDiffusionConditioningSet remains the single owner and must stay open while the sequence is in use.
- Mixed prepared conditioning data can be filtered to a video-only view; every VIDEO_FRAME entry must use VideoFrameConditioningData.

## Sequence windowing

VideoFrameSequenceWindowing creates immutable, non-owning windows for bounded-memory or chunked processing.

- Window size and start stride are expressed in frame counts; both must be positive.
- A stride smaller than the window size creates overlap; a larger stride intentionally skips frames between windows.
- INCLUDE_PARTIAL retains a final non-empty short window; DROP_PARTIAL emits only complete windows and may return an empty list.
- Windows preserve original frame instances, sequence indices, timestamps, and source ownership. They do not copy pixels, decode, interpolate, or alter temporal metadata.
- Windowing is independent of temporal sampling and can be applied to an already sampled sequence.

No FFmpeg, ImageIO, Python, PyTorch, ComfyUI, tensor backend, or device-selection dependency is introduced.

## Lazy batch mapping

`VideoFrameBatchStream.mapChunks` composes bounded chunking with an injected batch-packing function.

- The mapper is invoked only when the consumer calls `next()`; checking `hasNext()` does not pack or prefetch.
- Each output is handed to the consumer immediately. The stream does not retain previous chunks or produced batch values.
- The consumer owns each returned value. If a packed value owns resources, the consumer must close it after use.
- Iterators are independent and repeatable; each starts from the first frame.
- The source conditioning owner must remain open while chunks are being mapped. A mapper that needs independent lifetime should copy or retain the required data into its returned value.
- The mapper is injected, so this layer does not require a specific packed-batch class, tensor backend, device, codec, or scheduler.

This is synchronous pull-based flow control. It does not create worker threads, prefetch, or provide asynchronous backpressure.

## Bounded-memory batch consumption

`VideoFrameBatchConsumer.consume` processes resource-owning batch values one at a time.

- Each batch is closed with try-with-resources before the source iterator is asked for another value.
- Processing failure still closes the current batch; a close failure is attached as a suppressed exception.
- A close failure after successful processing stops consumption and is propagated; later batches are not requested.
- The method returns the number of batches successfully processed and closed.
- The consumer owns each produced batch, but not the source sequence or its prepared conditioning set.

Compose it with `VideoFrameBatchStream.mapChunks` to pack and consume one bounded chunk at a time. Keep the source conditioning owner open for the duration of the call. The processor must not retain a batch beyond its callback unless it explicitly copies or retains the needed data into independently owned storage.

## Batch processing controls

`VideoBatchProcessingContext` supplies optional synchronous cancellation and progress reporting to `VideoFrameBatchConsumer.consume`.

- Cancellation is polled before requesting the next batch, before processing, and after the processor callback. It does not interrupt an active callback.
- A batch is always closed when cancellation is observed while it is being processed. Cancellation observed before successful completion does not increment the completed count or emit a progress event.
- Progress is emitted after successful processing and close, with the completed batch count and an `OptionalLong` total. Unknown totals remain unknown; the consumer does not pre-count or materialize a lazy source.
- Progress-listener failures stop consumption after the current batch has been closed.
- The context is synchronous and per invocation. It does not introduce executors, scheduling, threads, or asynchronous prefetch.
