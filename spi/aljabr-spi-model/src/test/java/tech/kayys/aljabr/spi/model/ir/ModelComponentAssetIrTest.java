package tech.kayys.aljabr.spi.model.ir;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

class ModelComponentAssetIrTest {

    private static GraphDescriptor graph() {
        GraphValue input = new GraphValue("tokens", TensorType.of(
                TensorDataType.I64, TensorShape.of(1, 8)));
        GraphValue output = new GraphValue("embeddings", TensorType.of(
                TensorDataType.F32, TensorShape.of(1, 8, 64)));
        OperationDescriptor operation = new OperationDescriptor(
                "embed", "embedding.lookup", Map.of("indices", List.of("tokens")),
                Map.of("values", "embeddings"), Map.of(), Map.of());
        return GraphDescriptor.of("text-embedding", List.of(input, output), List.of(operation),
                new GraphSignature(Map.of("token_ids", "tokens"), Map.of("hidden_states", "embeddings")));
    }

    @Test
    void modelsExternalAssetsWithoutLoadingPayloads() {
        AssetReference vocabulary = new AssetReference(
                "vocab", "hf://example/model/vocab.json",
                Optional.of("application/json"), OptionalLong.of(4096),
                Optional.of("abc123"), Optional.of("sha256"), Map.of("origin", "hub"));

        assertEquals("hf://example/model/vocab.json", vocabulary.location());
        assertEquals(4096L, vocabulary.sizeBytes().orElseThrow());
        assertThrows(IllegalArgumentException.class, () -> new AssetReference(
                "bad", "file:///tmp/vocab", Optional.empty(), OptionalLong.of(-1),
                Optional.empty(), Optional.empty(), Map.of()));
        assertThrows(IllegalArgumentException.class, () -> new AssetReference(
                "bad-checksum", "file:///tmp/vocab", Optional.empty(), OptionalLong.empty(),
                Optional.of("abc"), Optional.empty(), Map.of()));
    }

    @Test
    void componentCollectionsRequireUniqueIdsAndValidReferences() {
        ModelComponent tokenizer = ModelComponent.of("tokenizer", ModelComponentKind.TOKENIZER);
        assertThrows(IllegalArgumentException.class, () -> new ComponentCollection(List.of(tokenizer, tokenizer)));
        assertThrows(IllegalArgumentException.class, () -> new AssetCollection(List.of(
                AssetReference.of("same", "asset:a"), AssetReference.of("same", "asset:b"))));
    }

    @Test
    void bindsTokenizerAssetsAndPortsToGraphSignatures() {
        AssetCollection assets = new AssetCollection(List.of(
                AssetReference.of("vocab", "hf://example/model/vocab.json"),
                AssetReference.of("merges", "hf://example/model/merges.txt")));
        ModelComponent tokenizer = new ModelComponent(
                "tokenizer", ModelComponentKind.TOKENIZER,
                Map.of("add_bos", ConfigurationValue.of(true),
                        "normalization", ConfigurationValue.of("NFC")),
                List.of("vocab", "merges"),
                List.of(new ComponentGraphBinding("text-embedding",
                        Map.of("token_ids", "token_ids"),
                        Map.of("hidden_states", "hidden_states"))),
                Map.of("vendor.example/tokenizer_mode", "byte-level"));

        ModelDescriptor model = new ModelDescriptor(
                ModelIdentity.of("multimodal-sample"), "decoder", "native",
                Map.of(), Map.of(), List.of())
                .withGraphCollection(new GraphCollection(List.of(graph())))
                .withAssetCollection(assets)
                .withComponentCollection(new ComponentCollection(List.of(tokenizer)));

        assertEquals(ModelComponentKind.TOKENIZER,
                model.componentCollection().orElseThrow().components().getFirst().kind());
        assertEquals("vocab", model.componentCollection().orElseThrow()
                .components().getFirst().assetIds().getFirst());
    }

    @Test
    void rejectsUnknownAssetGraphAndGraphPorts() {
        ModelComponent missingAsset = new ModelComponent(
                "tokenizer", ModelComponentKind.TOKENIZER, Map.of(),
                List.of("missing"), List.of(), Map.of());
        ModelDescriptor noAssets = new ModelDescriptor(
                ModelIdentity.of("sample"), "decoder", "native", Map.of(), Map.of(), List.of());
        assertThrows(IllegalArgumentException.class, () -> noAssets
                .withComponentCollection(new ComponentCollection(List.of(missingAsset))));

        ModelComponent missingGraph = new ModelComponent(
                "processor", ModelComponentKind.IMAGE_PROCESSOR, Map.of(), List.of(),
                List.of(new ComponentGraphBinding("missing-graph", Map.of(), Map.of())), Map.of());
        assertThrows(IllegalArgumentException.class, () -> noAssets
                .withComponentCollection(new ComponentCollection(List.of(missingGraph))));

        ModelComponent missingPort = new ModelComponent(
                "tokenizer", ModelComponentKind.TOKENIZER, Map.of(), List.of(),
                List.of(new ComponentGraphBinding("text-embedding",
                        Map.of("tokens", "not-an-input"), Map.of())), Map.of());
        assertThrows(IllegalArgumentException.class, () -> noAssets
                .withGraphCollection(new GraphCollection(List.of(graph())))
                .withComponentCollection(new ComponentCollection(List.of(missingPort))));
    }

    @Test
    void copyMethodsPreserveAllExistingAndNewIrLayers() {
        ModelDescriptor model = new ModelDescriptor(
                ModelIdentity.of("sample"), "decoder", "native", Map.of(), Map.of(), List.of())
                .withGraphCollection(new GraphCollection(List.of(graph())))
                .withAssetCollection(new AssetCollection(List.of(AssetReference.of("vocab", "asset:vocab"))))
                .withComponentCollection(new ComponentCollection(List.of(
                        new ModelComponent("tokenizer", ModelComponentKind.TOKENIZER, Map.of(),
                                List.of("vocab"), List.of(), Map.of()))));

        ModelDescriptor renamed = model.withIdentity(ModelIdentity.of("renamed"));
        assertTrue(renamed.graphCollection().isPresent());
        assertTrue(renamed.assetCollection().isPresent());
        assertTrue(renamed.componentCollection().isPresent());
        assertEquals("renamed", renamed.identity().id());
    }
}
