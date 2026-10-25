package tech.kayys.aljabr.metal.binding;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;
import org.jboss.logging.Logger;

/**
 * Utility for finding the libaljabr_metal.dylib in standard locations.
 */
public class MetalLibraryDiscovery {
    private static final Logger LOG = Logger.getLogger(MetalLibraryDiscovery.class);
    private static final String LIB_NAME = "libaljabr_metal.dylib";
    private static final Set<String> LEGACY_LIB_NAMES = Set.of(
            "libalkhawarizm_metal.dylib",
            "libaqli_metal.dylib");

    static boolean isLegacyLibraryName(Path path) {
        Path fileName = path.getFileName();
        return fileName != null
                && LEGACY_LIB_NAMES.contains(fileName.toString().toLowerCase(Locale.ROOT));
    }

    public static Path findLibrary() {
        // 1. Explicit override
        String override = System.getProperty("aljabr.metal.dylib");
        if (override != null) {
            Path p = Path.of(override);
            if (isLegacyLibraryName(p)) {
                LOG.warnf("Ignoring legacy Metal dylib override %s; expected the renamed Aljabr library %s",
                        p, LIB_NAME);
            } else if (Files.exists(p)) {
                return p;
            }
        }

        // 2. Prefer freshly built local development outputs when running from the repo.
        Path cwd = Path.of("").toAbsolutePath();
        Path[] localCandidates = new Path[] {
                cwd.resolve("aljabr/backend/metal/aljabr-backend-metal/target/native/darwin-aarch64")
                        .resolve(LIB_NAME),
                cwd.resolve("../aljabr/backend/metal/aljabr-backend-metal/target/native/darwin-aarch64")
                        .resolve(LIB_NAME),
                cwd.resolve("backend/metal/aljabr-backend-metal/target/native/darwin-aarch64").resolve(LIB_NAME),
                cwd.resolve(LIB_NAME)
        };
        for (Path candidate : localCandidates) {
            if (Files.exists(candidate))
                return candidate;
        }

        // 3. Standard Aljabr installation path (~/.aljabr/libs)
        String home = System.getProperty("user.home");
        if (home != null) {
            Path p = Path.of(home, ".aljabr", "libs", LIB_NAME);
            if (Files.exists(p))
                return p;
        }

        // 4. Search in java.library.path
        String libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            for (String dir : libPath.split(File.pathSeparator)) {
                Path p = Path.of(dir, LIB_NAME);
                if (Files.exists(p))
                    return p;
            }
        }

        return cwd.resolve(LIB_NAME);
    }
}
