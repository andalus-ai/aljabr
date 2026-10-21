package tech.kayys.aljabr.models.bert;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BertModelFamilyPluginTest {

    @Test
    void bertPluginSatisfiesSharedModelFamilyContract() {
        BertModelFamilyPlugin plugin = new BertModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "bert model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void bertFixtureMatchesDescriptorAndTokenizer() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new BertModelFamilyPlugin(),
                fixture("bert"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "bert fixture should match descriptor and tokenizer claims");
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                BertModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
