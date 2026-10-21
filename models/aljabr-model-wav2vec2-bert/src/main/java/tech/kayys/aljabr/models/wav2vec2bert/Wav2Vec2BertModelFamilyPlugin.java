package tech.kayys.aljabr.models.wav2vec2bert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Wav2Vec2BertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "wav2vec2_bert",
                "Wav2Vec2-BERT",
                List.of("wav2vec2_bert", "wav2vec2-bert"),
                List.of("Wav2Vec2BertForCTC", "Wav2Vec2BertForSequenceClassification",
                        "Wav2Vec2BertForAudioFrameClassification", "Wav2Vec2BertForXVector",
                        "Wav2Vec2BertProcessor"),
                List.of(
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "tokenizer", "wav2vec2_bert_processor",
                        "direct_safetensor", "pending_audio_conformer_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("wav2vec2-bert-processor"));
    }
}
