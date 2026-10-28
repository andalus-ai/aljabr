// Run with: java scripts/MetalDecodeProfileSummary.java metal-fused.log
// Or:       jbang scripts/MetalDecodeProfileSummary.java --warmup 2 --format csv metal-fused.log
// JDK 25 source-file mode; standard library only.
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class MetalDecodeProfileSummary {
    private static final Pattern PROFILE = Pattern.compile("\\[METAL DECODE PROFILE\\]\\s+(.*)");
    private static final Pattern FIELD = Pattern.compile("([a-z_]+)=([^\\s]+)");
    private static final List<String> METRICS = List.of("wall_ms", "encode_ms", "submit_wait_ms", "gpu_ms");
    private record Sample(int pos, int layers, String path, Map<String, Double> metrics) {}
    private static final class Summary {
        final String path;
        final int samples;
        final int layers;
        final Map<String, Double> metrics = new LinkedHashMap<>();
        Summary(String path, int samples, int layers) {
            this.path = path; this.samples = samples; this.layers = layers;
        }
    }

    public static void main(String[] argv) throws Exception {
        int warmup = 0;
        String format = "table";
        List<String> files = new ArrayList<>();
        for (int i = 0; i < argv.length; i++) {
            switch (argv[i]) {
                case "--warmup" -> {
                    if (++i >= argv.length) throw new IllegalArgumentException("--warmup requires a value");
                    warmup = Integer.parseInt(argv[i]);
                }
                case "--format" -> {
                    if (++i >= argv.length) throw new IllegalArgumentException("--format requires a value");
                    format = argv[i];
                }
                case "--help", "-h" -> { usage(); return; }
                default -> {
                    if (argv[i].startsWith("--")) throw new IllegalArgumentException("Unknown option: " + argv[i]);
                    files.add(argv[i]);
                }
            }
        }
        if (warmup < 0) throw new IllegalArgumentException("--warmup must be >= 0");
        if (!List.of("table", "json", "csv").contains(format))
            throw new IllegalArgumentException("--format must be table, json, or csv");

        List<Sample> samples = new ArrayList<>();
        int malformed = 0;
        if (files.isEmpty()) {
            try (var reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
                int[] bad = {0};
                read(reader, samples, bad);
                malformed += bad[0];
            }
        } else {
            for (String filename : files) {
                try (var reader = Files.newBufferedReader(Path.of(filename), StandardCharsets.UTF_8)) {
                    int[] bad = {0};
                    read(reader, samples, bad);
                    malformed += bad[0];
                } catch (IOException e) {
                    System.err.println("error: cannot read " + filename + ": " + e.getMessage());
                    System.exit(2);
                }
            }
        }
        if (malformed > 0) System.err.println("warning: skipped " + malformed + " malformed profile line(s)");
        if (samples.isEmpty()) {
            System.err.println("error: no valid [METAL DECODE PROFILE] records found");
            System.exit(1);
        }

        Map<String, List<Sample>> grouped = new TreeMap<>();
        for (Sample sample : samples) grouped.computeIfAbsent(sample.path(), k -> new ArrayList<>()).add(sample);
        List<Summary> summaries = new ArrayList<>();
        for (var entry : grouped.entrySet()) {
            List<Sample> pathSamples = entry.getValue();
            if (warmup >= pathSamples.size()) continue;
            List<Sample> kept = pathSamples.subList(warmup, pathSamples.size());
            Summary summary = new Summary(entry.getKey(), kept.size(), kept.getLast().layers());
            for (String metric : METRICS) {
                List<Double> values = kept.stream().map(s -> s.metrics().get(metric)).sorted().toList();
                summary.metrics.put(metric + "_median", median(values));
                summary.metrics.put(metric + "_p95", percentile(values, 95));
                summary.metrics.put(metric + "_min", values.getFirst());
                summary.metrics.put(metric + "_max", values.getLast());
            }
            summaries.add(summary);
        }
        if (summaries.isEmpty()) {
            System.err.println("error: no samples remain after warmup filtering");
            System.exit(1);
        }

        switch (format) {
            case "json" -> printJson(summaries);
            case "csv" -> printCsv(summaries);
            default -> printTable(summaries);
        }
        System.err.printf("Parsed %d samples; summarize by path with warmup=%d.%n", samples.size(), warmup);
    }

    private static void read(BufferedReader reader, List<Sample> samples, int[] malformed) throws IOException {
        String line;
        while ((line = reader.readLine()) != null) {
            Matcher match = PROFILE.matcher(line);
            if (!match.find()) continue;
            Map<String, String> fields = new LinkedHashMap<>();
            Matcher fieldMatcher = FIELD.matcher(match.group(1));
            while (fieldMatcher.find()) fields.put(fieldMatcher.group(1), fieldMatcher.group(2));
            try {
                int pos = Integer.parseInt(fields.get("pos"));
                int layers = Integer.parseInt(fields.get("layers"));
                String path = fields.get("path");
                if (path == null || path.isBlank()) throw new IllegalArgumentException("missing path");
                Map<String, Double> metrics = new LinkedHashMap<>();
                for (String metric : METRICS) {
                    double value = Double.parseDouble(fields.get(metric));
                    if (!Double.isFinite(value) || value < 0) throw new IllegalArgumentException("invalid " + metric);
                    metrics.put(metric, value);
                }
                samples.add(new Sample(pos, layers, path, metrics));
            } catch (RuntimeException e) {
                malformed[0]++;
            }
        }
    }
    private static double median(List<Double> values) {
        int n = values.size();
        return n % 2 == 1 ? values.get(n / 2) : (values.get(n / 2 - 1) + values.get(n / 2)) / 2.0;
    }
    private static double percentile(List<Double> values, int p) {
        int index = Math.max(0, (int) Math.ceil((p / 100.0) * values.size()) - 1);
        return values.get(index);
    }
    private static List<String> columns() {
        List<String> cols = new ArrayList<>(List.of("path", "samples", "layers"));
        for (String metric : METRICS) {
            cols.add(metric + "_median"); cols.add(metric + "_p95");
            cols.add(metric + "_min"); cols.add(metric + "_max");
        }
        return cols;
    }
    private static List<String> values(Summary s) {
        List<String> v = new ArrayList<>(List.of(s.path, Integer.toString(s.samples), Integer.toString(s.layers)));
        for (String col : columns().subList(3, columns().size()))
            v.add(String.format(Locale.ROOT, "%.6f", s.metrics.get(col)));
        return v;
    }
    private static void printCsv(List<Summary> summaries) {
        List<String> cols = columns();
        System.out.println(String.join(",", cols));
        for (Summary s : summaries) System.out.println(values(s).stream().map(MetalDecodeProfileSummary::csv).reduce((a,b)->a+","+b).orElse(""));
    }
    private static void printJson(List<Summary> summaries) {
        System.out.println("[");
        for (int i = 0; i < summaries.size(); i++) {
            Summary s = summaries.get(i);
            System.out.println("  {");
            System.out.println("    \"path\": " + json(s.path) + ",");
            System.out.println("    \"samples\": " + s.samples + ",");
            System.out.println("    \"layers\": " + s.layers + ",");
            int j = 0;
            for (var e : s.metrics.entrySet()) {
                System.out.printf(Locale.ROOT, "    %s: %.6f%s%n", json(e.getKey()), e.getValue(), j++ == s.metrics.size() - 1 ? "" : ",");
            }
            System.out.print("  }" + (i == summaries.size() - 1 ? "\n" : ",\n"));
        }
        System.out.println("]");
    }
    private static void printTable(List<Summary> summaries) {
        List<String> cols = new ArrayList<>(List.of("path", "samples"));
        for (String metric : METRICS) { cols.add(metric + "_median"); cols.add(metric + "_p95"); }
        List<List<String>> rows = new ArrayList<>();
        rows.add(cols);
        for (Summary s : summaries) {
            List<String> row = new ArrayList<>(List.of(s.path, Integer.toString(s.samples)));
            for (String metric : METRICS) {
                row.add(String.format(Locale.ROOT, "%.3f", s.metrics.get(metric + "_median")));
                row.add(String.format(Locale.ROOT, "%.3f", s.metrics.get(metric + "_p95")));
            }
            rows.add(row);
        }
        int[] widths = new int[cols.size()];
        for (List<String> row : rows) for (int i = 0; i < row.size(); i++) widths[i] = Math.max(widths[i], row.get(i).length());
        for (int r = 0; r < rows.size(); r++) {
            List<String> cells = new ArrayList<>();
            for (int i = 0; i < cols.size(); i++) cells.add(r == 0 ? rows.get(r).get(i) + " ".repeat(widths[i] - rows.get(r).get(i).length())
                    : rows.get(r).get(i) + " ".repeat(widths[i] - rows.get(r).get(i).length()));
            System.out.println(String.join("  ", cells));
        }
    }
    private static String csv(String s) { return "\"" + s.replace("\"", "\"\"") + "\""; }
    private static String json(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + "\"";
    }
    private static void usage() {
        System.out.println("""
                Usage: java scripts/MetalDecodeProfileSummary.java [options] [log ...]
                  --warmup N          Discard N initial samples per path (default: 0)
                  --format FORMAT     table, json, or csv (default: table)
                Reads stdin when no log files are supplied.
                """);
    }
}
