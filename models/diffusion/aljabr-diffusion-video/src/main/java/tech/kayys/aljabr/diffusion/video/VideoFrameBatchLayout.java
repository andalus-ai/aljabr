package tech.kayys.aljabr.diffusion.video;

/** Contiguous memory layout for a packed temporal video-frame batch. */
public enum VideoFrameBatchLayout {
    /** [time, height, width, channel] */
    THWC,
    /** [time, channel, height, width] */
    TCHW
}
