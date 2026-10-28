// Run with: java scripts/BenchmarkAqli.java --model hf:unsloth/gemma-4-12b-it-GGUF
// Or:       jbang scripts/BenchmarkAqli.java --model hf:unsloth/gemma-4-12b-it-GGUF
//
// JDK 25 source-file mode; standard library only.
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class BenchmarkAqli {
    private static final Pattern RATE = Pattern.compile(
            "(?i)(?:tok(?:en)?s?/s|tokens per second)\\s*[:=]?\\s*([0-9]+(?:\\.[0-9]+)?)");

    public static void main(String[] argv) throws Exception {
        Map<String, String> options = parseArgs(argv);
        String cli = options.getOrDefault("cli", "aqli");
        String model = required(options, "model");
        String prompt = options.getOrDefault("prompt", "who are you");
        int maxTokens = positiveInt(options, "max-tokens", 50);
        int runs = positiveInt(options, "runs", 5);
        int warmup = nonNegativeInt(options, "warmup", 1);
        Path outDir = Path.of(options.getOrDefault("out-dir", "build/metal-benchmark"))
                .toAbsolutePath().normalize();
        boolean noJava = options.containsKey("no-java");
        boolean disableMetal = options.containsKey("disable-metal");
        if (options.containsKey("help")) {
            usage();
            return;
        }

        Path logsDir = outDir.resolve("logs");
        Files.createDirectories(logsDir);
        Path csvPath = outDir.resolve("runs.csv");
        List<String> command = new ArrayList<>(List.of(cli, "run", "--model", model,
                "--no-banner", "--prompt", prompt, "--max-tokens", Integer.toString(maxTokens)));
        if (!noJava) command.add("--java");

        System.out.println("Command: " + shellQuote(command));
        System.out.printf(Locale.ROOT, "Warm-ups: %d; measured runs: %d%n", warmup, runs);
        System.out.println("Logs and CSV: " + outDir);

        List<Map<String, String>> rows = new ArrayList<>();
        int total = warmup + runs;
        for (int index = 0; index < total; index++) {
            boolean isWarmup = index < warmup;
            int number = isWarmup ? index + 1 : index - warmup + 1;
            String label = isWarmup ? "warmup" : "run";
            Path logPath = logsDir.resolve(String.format(Locale.ROOT, "%s-%02d.log", label, number));
            ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
            if (disableMetal) {
                String current = builder.environment().getOrDefault("JAVA_TOOL_OPTIONS", "");
                builder.environment().put("JAVA_TOOL_OPTIONS",
                        (current + " -Daljabr.metal.disable=true").trim());
            }
            long start = System.nanoTime();
            Process process = builder.start();
            byte[] output;
            try (var stream = process.getInputStream()) {
                output = stream.readAllBytes();
            }
            int exitCode = process.waitFor();
            double elapsedSeconds = (System.nanoTime() - start) / 1_000_000_000.0;
            String text = new String(output, StandardCharsets.UTF_8);
            Files.writeString(logPath, text, StandardCharsets.UTF_8);
            System.out.printf(Locale.ROOT, "%s %d/%d: %.3fs (exit %d) -> %s%n",
                    label, number, isWarmup ? warmup : runs, elapsedSeconds, exitCode, logPath);
            if (exitCode != 0) {
                System.err.println(tail(text, 6000));
                System.err.println("Benchmark stopped: command failed; inspect " + logPath);
                System.exit(exitCode == 0 ? 1 : exitCode);
            }
            if (!isWarmup) {
                Matcher matcher = RATE.matcher(text);
                String rate = "";
                while (matcher.find()) rate = matcher.group(1);
                Map<String, String> row = new LinkedHashMap<>();
                row.put("timestamp_utc", Instant.now().toString());
                row.put("run", Integer.toString(number));
                row.put("elapsed_seconds", String.format(Locale.ROOT, "%.6f", elapsedSeconds));
                row.put("reported_tokens_per_second", rate);
                row.put("exit_code", Integer.toString(exitCode));
                row.put("model", model);
                row.put("prompt", prompt);
                row.put("max_tokens", Integer.toString(maxTokens));
                row.put("command", shellQuote(command));
                row.put("machine", System.getProperty("os.arch", "unknown"));
                row.put("os", System.getProperty("os.name", "unknown") + " "
                        + System.getProperty("os.version", "unknown"));
                row.put("java_version", System.getProperty("java.version", "unknown"));
                row.put("metal_disabled", Boolean.toString(disableMetal));
                row.put("log_file", logPath.toString());
                rows.add(row);
            }
        }

        List<String> columns = List.copyOf(rows.getFirst().keySet());
        StringBuilder csv = new StringBuilder();
        csv.append(String.join(",", columns)).append('\n');
        for (Map<String, String> row : rows) {
            for (int i = 0; i < columns.size(); i++) {
                if (i > 0) csv.append(',');
                csv.append(csvCell(row.get(columns.get(i))));
            }
            csv.append('\n');
        }
        Files.writeString(csvPath, csv, StandardCharsets.UTF_8);
        List<Double> times = rows.stream().map(r -> Double.parseDouble(r.get("elapsed_seconds")))
                .sorted(Comparator.naturalOrder()).toList();
        double median = times.size() % 2 == 1 ? times.get(times.size() / 2)
                : (times.get(times.size() / 2 - 1) + times.get(times.size() / 2)) / 2.0;
        System.out.printf(Locale.ROOT, "Measured wall time: min=%.3fs median=%.3fs max=%.3fs%n",
                times.getFirst(), median, times.getLast());
        System.out.println("CSV: " + csvPath);
        System.out.println("Note: elapsed time includes process startup/model loading; it is not decode latency.");
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> result = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.equals("--no-java") || arg.equals("--disable-metal") || arg.equals("--help") || arg.equals("-h")) {
                result.put(arg.equals("-h") ? "help" : arg.substring(2), "true");
            } else if (arg.startsWith("--")) {
                if (i + 1 >= args.length || args[i + 1].startsWith("--"))
                    throw new IllegalArgumentException("Missing value for " + arg);
                result.put(arg.substring(2), args[++i]);
            } else {
                throw new IllegalArgumentException("Unexpected argument: " + arg);
            }
        }
        return result;
    }

    private static String required(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("--" + key + " is required");
        return value;
    }
    private static int positiveInt(Map<String, String> o, String k, int d) {
        int n = Integer.parseInt(o.getOrDefault(k, Integer.toString(d)));
        if (n < 1) throw new IllegalArgumentException("--" + k + " must be positive");
        return n;
    }
    private static int nonNegativeInt(Map<String, String> o, String k, int d) {
        int n = Integer.parseInt(o.getOrDefault(k, Integer.toString(d)));
        if (n < 0) throw new IllegalArgumentException("--" + k + " cannot be negative");
        return n;
    }
    private static String csvCell(String value) {
        if (value == null) return "";
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }
    private static String shellQuote(List<String> command) {
        return command.stream().map(s -> s.matches("[A-Za-z0-9_./:=+-]+") ? s
                : "'" + s.replace("'", "'\\''") + "'").reduce((a, b) -> a + " " + b).orElse("");
    }
    private static String tail(String text, int maxChars) {
        return text.length() <= maxChars ? text : text.substring(text.length() - maxChars);
    }
    private static void usage() {
        System.out.println("""
                Usage: java scripts/BenchmarkAqli.java --model MODEL [options]
                  --cli PATH          AQLI executable (default: aqli)
                  --prompt TEXT       Fixed prompt (default: who are you)
                  --max-tokens N      Output token limit (default: 50)
                  --warmup N          Warm-up runs (default: 1)
                  --runs N            Measured runs (default: 5)
                  --out-dir PATH      Output directory (default: build/metal-benchmark)
                  --no-java           Do not pass --java
                  --disable-metal     Set -Daljabr.metal.disable=true in child environment
                """);
    }
}
