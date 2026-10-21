package tech.kayys.aljabr.models.falcon;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FalconModelFamilyPluginTest {

    @Test
    void falconPluginSatisfiesSharedModelFamilyContract() {
        FalconModelFamilyPlugin plugin = new FalconModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "falcon model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void falconFixtureMatchesDescriptorAndTokenizer() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new FalconModelFamilyPlugin(),
                fixture("falcon"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "falcon fixture should match descriptor and tokenizer claims");
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                FalconModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
