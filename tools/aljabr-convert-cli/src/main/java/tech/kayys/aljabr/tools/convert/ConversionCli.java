package tech.kayys.aljabr.tools.convert;

import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport.Diagnostic;
import tech.kayys.aljabr.spi.model.ir.ConversionExecutionResult;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelConversionFacade;
import tech.kayys.aljabr.spi.model.ir.PreparedConversion;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Thin command-line adapter for AMIR model conversion.
 *
 * <p>Format discovery, compatibility assessment and payload transfer are
 * delegated to the registered Aljabr providers.
 */
public final class ConversionCli {

    private ConversionCli() {
    }

    public static void main(String[] args) {
        int status = run(args, System.out, System.err, ModelConversionFacade.discover());
        if (status != 0) {
            System.exit(status);
        }
    }

    static int run(String[] args, PrintStream out, PrintStream err, ModelConversionFacade facade) {
        if (args.length == 1 && isHelp(args[0])) {
            usage(out);
            return 0;
        }
        try {
            Arguments parsed = Arguments.parse(args);
            PreparedConversion prepared = facade.prepare(
                    parsed.input(), parsed.sourceFormat(), parsed.targetFormat());
            ConversionPlan plan = prepared.plan();
            printPlan(out, plan);

            if (!plan.isConvertible()) {
                err.println("Conversion is unsupported for "
                        + plan.sourceFormat() + " -> " + plan.targetFormat());
                return 3;
            }
            if (parsed.command().equals("plan")) {
                return 0;
            }

            ConversionRequest request = new ConversionRequest(
                    parsed.output().toUri(), parsed.acknowledgePartial(), parsed.options());
            ConversionExecutionResult result = facade.execute(prepared, request);
            out.println("Conversion completed");
            out.println("Output: " + result.output());
            out.println("Tensors: " + result.tensorCount());
            out.println("Payload bytes: " + result.payloadBytes());
            if (!result.untransferredMetadataKeys().isEmpty()) {
                out.println("Untransferred metadata: " + String.join(", ", result.untransferredMetadataKeys()));
            }
            return 0;
        } catch (IllegalArgumentException e) {
            err.println("Invalid arguments: " + e.getMessage());
            usage(err);
            return 2;
        } catch (IllegalStateException e) {
            err.println("Conversion rejected: " + e.getMessage());
            return 3;
        } catch (IOException e) {
            err.println("Conversion I/O failed: " + e.getMessage());
            return 4;
        }
    }

    private static void printPlan(PrintStream out, ConversionPlan plan) {
        out.println("Source format: " + plan.sourceFormat());
        out.println("Target format: " + plan.targetFormat());
        out.println("Fidelity: " + plan.compatibility().fidelity());
        for (Diagnostic diagnostic : plan.compatibility().diagnostics()) {
            out.println("[" + diagnostic.severity() + "] " + diagnostic.code()
                    + ": " + diagnostic.message());
        }
    }

    private static boolean isHelp(String value) {
        return "--help".equals(value) || "-h".equals(value) || "help".equals(value);
    }

    private static void usage(PrintStream out) {
        out.println("""
                Aljabr model conversion

                Usage:
                  aljabr-convert plan --input <path> --from <format> --to <format>
                  aljabr-convert convert --input <path> --from <format> --to <format>
                      --output <path> [--acknowledge-partial] [--option <namespace.key=value> ...]

                Commands:
                  plan       Read model metadata and print compatibility; do not write output.
                  convert    Execute a supported conversion.

                Formats are explicit; the CLI does not infer them from file extensions.
                Partial-fidelity conversions require --acknowledge-partial.
                """);
    }

    private record Arguments(
            String command,
            Path input,
            String sourceFormat,
            String targetFormat,
            Path output,
            boolean acknowledgePartial,
            Map<String, String> options) {

        private static Arguments parse(String[] args) {
            if (args == null || args.length == 0) {
                throw new IllegalArgumentException("a command is required");
            }
            String command = args[0];
            if (!command.equals("plan") && !command.equals("convert")) {
                throw new IllegalArgumentException("unknown command: " + command);
            }

            Map<String, String> values = new LinkedHashMap<>();
            Map<String, String> options = new LinkedHashMap<>();
            boolean acknowledgePartial = false;
            for (int i = 1; i < args.length; i++) {
                String arg = args[i];
                if (arg.equals("--acknowledge-partial")) {
                    acknowledgePartial = true;
                } else if (arg.equals("--option")) {
                    if (++i >= args.length) {
                        throw new IllegalArgumentException("--option requires namespace.key=value");
                    }
                    String option = args[i];
                    int separator = option.indexOf('=');
                    if (separator <= 0 || separator == option.length() - 1) {
                        throw new IllegalArgumentException("invalid option: " + option);
                    }
                    String key = option.substring(0, separator);
                    String value = option.substring(separator + 1);
                    if (!key.contains(".")) {
                        throw new IllegalArgumentException("option key must be namespaced: " + key);
                    }
                    if (options.putIfAbsent(key, value) != null) {
                        throw new IllegalArgumentException("duplicate option: " + key);
                    }
                } else if (arg.startsWith("--")) {
                    if (++i >= args.length) {
                        throw new IllegalArgumentException(arg + " requires a value");
                    }
                    String key = arg.substring(2);
                    if (!key.equals("input") && !key.equals("from")
                            && !key.equals("to") && !key.equals("output")) {
                        throw new IllegalArgumentException("unknown option: " + arg);
                    }
                    if (values.putIfAbsent(key, args[i]) != null) {
                        throw new IllegalArgumentException("duplicate option: " + arg);
                    }
                } else {
                    throw new IllegalArgumentException("unexpected argument: " + arg);
                }
            }

            Path input = requiredPath(values, "input");
            String from = required(values, "from");
            String to = required(values, "to");
            Path output = command.equals("convert") ? requiredPath(values, "output") : null;
            if (command.equals("plan") && values.containsKey("output")) {
                throw new IllegalArgumentException("--output is only valid for convert");
            }
            if (command.equals("plan") && acknowledgePartial) {
                throw new IllegalArgumentException("--acknowledge-partial is only valid for convert");
            }
            if (command.equals("plan") && !options.isEmpty()) {
                throw new IllegalArgumentException("--option is only valid for convert");
            }
            return new Arguments(command, input, from, to, output, acknowledgePartial, Map.copyOf(options));
        }

        private static Path requiredPath(Map<String, String> values, String key) {
            return Path.of(required(values, key)).toAbsolutePath().normalize();
        }

        private static String required(Map<String, String> values, String key) {
            String value = values.get(key);
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException("--" + key + " is required");
            }
            return value;
        }
    }
}
