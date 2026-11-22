package tech.kayys.aljabr.cache;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KVCacheCompressionContractTest {

    @Test
    void contextDescribesPositionMajorLayerBlockGeometry() {
        KVCacheCompressionContext context = new KVCacheCompressionContext(
                3, 12, 4, 8, 64, KVCacheTensorKind.KEY);

        assertEquals(3, context.layer());
        assertEquals(12, context.firstPosition());
        assertEquals(4, context.tokenCount());
        assertEquals(8, context.kvHeads());
        assertEquals(64, context.headDimension());
        assertEquals(KVCacheTensorKind.KEY, context.tensorKind());
        assertEquals(2_048, context.elementCount());
    }

    @Test
    void contextRejectsInvalidGeometryAndIntegerOverflow() {
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheCompressionContext(-1, 0, 1, 1, 1, KVCacheTensorKind.KEY));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheCompressionContext(0, -1, 1, 1, 1, KVCacheTensorKind.KEY));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheCompressionContext(0, 0, 0, 1, 1, KVCacheTensorKind.KEY));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheCompressionContext(0, 0, 1, 0, 1, KVCacheTensorKind.KEY));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheCompressionContext(0, 0, 1, 1, 0, KVCacheTensorKind.KEY));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheCompressionContext(
                        0, 0, Integer.MAX_VALUE, Integer.MAX_VALUE, 2, KVCacheTensorKind.VALUE));
        assertThrows(NullPointerException.class,
                () -> new KVCacheCompressionContext(0, 0, 1, 1, 1, null));
    }

    @Test
    void encodedBlockDefensivelyCopiesPayloadAndMetadata() {
        byte[] payload = {1, 2, 3};
        byte[] metadata = {4, 5};
        KVCacheEncodedBlock block = new KVCacheEncodedBlock(
                "example.codec", 2, 16, payload, metadata);

        payload[0] = 99;
        metadata[0] = 99;
        assertArrayEquals(new byte[] {1, 2, 3}, block.payload());
        assertArrayEquals(new byte[] {4, 5}, block.metadata());

        byte[] returnedPayload = block.payload();
        byte[] returnedMetadata = block.metadata();
        returnedPayload[1] = 88;
        returnedMetadata[1] = 88;

        assertArrayEquals(new byte[] {1, 2, 3}, block.payload());
        assertArrayEquals(new byte[] {4, 5}, block.metadata());
        assertEquals("example.codec", block.codecId());
        assertEquals(2, block.codecVersion());
        assertEquals(16, block.elementCount());
        assertEquals(3, block.payloadSizeBytes());
        assertEquals(2, block.metadataSizeBytes());
    }

    @Test
    void encodedBlockRejectsMissingOrInvalidIdentityAndPayload() {
        assertThrows(NullPointerException.class,
                () -> new KVCacheEncodedBlock(null, 1, 1, new byte[] {1}, new byte[0]));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheEncodedBlock(" ", 1, 1, new byte[] {1}, new byte[0]));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheEncodedBlock("codec", 0, 1, new byte[] {1}, new byte[0]));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheEncodedBlock("codec", 1, 0, new byte[] {1}, new byte[0]));
        assertThrows(IllegalArgumentException.class,
                () -> new KVCacheEncodedBlock("codec", 1, 1, new byte[0], new byte[0]));
        assertThrows(NullPointerException.class,
                () -> new KVCacheEncodedBlock("codec", 1, 1, null, new byte[0]));
        assertThrows(NullPointerException.class,
                () -> new KVCacheEncodedBlock("codec", 1, 1, new byte[] {1}, null));
    }
}
