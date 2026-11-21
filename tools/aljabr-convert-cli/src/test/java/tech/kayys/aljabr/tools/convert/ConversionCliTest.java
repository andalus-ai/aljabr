package tech.kayys.aljabr.tools.convert;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.ModelConversionFacade;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConversionCliTest {

    @Test
    void helpDoesNotRequireProviderDiscovery() {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        int status = ConversionCli.run(
                new String[]{"--help"},
                new PrintStream(output, true, StandardCharsets.UTF_8),
                System.err,
                null);

        assertEquals(0, status);
        assertTrue(output.toString(StandardCharsets.UTF_8).contains("aljabr-convert plan"));
    }

    @Test
    void rejectsUnknownCommandBeforeAccessingFacade() {
        ByteArrayOutputStream errors = new ByteArrayOutputStream();
        int status = ConversionCli.run(
                new String[]{"inspect"},
                System.out,
                new PrintStream(errors, true, StandardCharsets.UTF_8),
                null);

        assertEquals(2, status);
        assertTrue(errors.toString(StandardCharsets.UTF_8).contains("unknown command"));
    }

    @Test
    void rejectsUnnamespacedOptionsBeforeAccessingFacade() {
        ByteArrayOutputStream errors = new ByteArrayOutputStream();
        int status = ConversionCli.run(
                new String[]{"convert", "--input", "in.safetensors", "--from", "safetensors",
                        "--to", "gguf", "--output", "out.gguf", "--option", "alignment=32"},
                System.out,
                new PrintStream(errors, true, StandardCharsets.UTF_8),
                null);

        assertEquals(2, status);
        assertTrue(errors.toString(StandardCharsets.UTF_8).contains("must be namespaced"));
    }
}
