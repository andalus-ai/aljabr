package tech.kayys.aljabr.models.flava;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyFixtureValidator;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FlavaModelFamilyPluginTest {

    @Test
    void flavaPluginSatisfiesSharedModelFamilyContract() {
        FlavaModelFamilyPlugin plugin = new FlavaModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "flava model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void flavaFixtureMatchesDescriptorAndTokenizer() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new FlavaModelFamilyPlugin(),
                fixture("flava"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "flava fixture should match descriptor and tokenizer claims");
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                FlavaModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
