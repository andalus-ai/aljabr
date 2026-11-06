package tech.kayys.aljabr.spi.provider;

import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A half-open compatibility interval for Aljabr API versions.
 *
 * <p>Grammar: {@code >=MAJOR.MINOR.PATCH <MAJOR.MINOR.PATCH}. Versions are
 * non-negative decimal integers without leading zeroes (except zero). Prerelease
 * and build metadata are intentionally unsupported in this first SPI contract.</p>
 */
public final class ApiCompatibilityRange {
    private static final Pattern RANGE = Pattern.compile(
            "^>=((?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*))"
          + " <((?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*))$");
    private static final Pattern VERSION = Pattern.compile(
            "^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$");

    private final Version minimumInclusive;
    private final Version maximumExclusive;

    private ApiCompatibilityRange(Version minimumInclusive, Version maximumExclusive) {
        if (minimumInclusive.compareTo(maximumExclusive) >= 0) {
            throw new IllegalArgumentException("compatibility range must have a lower bound below its upper bound");
        }
        this.minimumInclusive = minimumInclusive;
        this.maximumExclusive = maximumExclusive;
    }

    public static ApiCompatibilityRange parse(String expression) {
        Objects.requireNonNull(expression, "expression");
        Matcher matcher = RANGE.matcher(expression.trim());
        if (!matcher.matches()) {
            throw new IllegalArgumentException(
                    "apiCompatibility must use '>=MAJOR.MINOR.PATCH <MAJOR.MINOR.PATCH': " + expression);
        }
        return new ApiCompatibilityRange(Version.parse(matcher.group(1)), Version.parse(matcher.group(2)));
    }

    /** Returns whether the supplied stable Aljabr API version is in this range. */
    public boolean includes(String aljabrVersion) {
        Version candidate = Version.parse(aljabrVersion);
        return candidate.compareTo(minimumInclusive) >= 0
                && candidate.compareTo(maximumExclusive) < 0;
    }

    private record Version(long major, long minor, long patch) implements Comparable<Version> {
        private static Version parse(String value) {
            Matcher matcher = VERSION.matcher(value);
            if (!matcher.matches()) {
                throw new IllegalArgumentException("version must be MAJOR.MINOR.PATCH: " + value);
            }
            try {
                return new Version(Long.parseLong(matcher.group(1)),
                        Long.parseLong(matcher.group(2)), Long.parseLong(matcher.group(3)));
            } catch (NumberFormatException ex) {
                throw new IllegalArgumentException("version component is too large: " + value, ex);
            }
        }

        @Override
        public int compareTo(Version other) {
            int result = Long.compare(major, other.major);
            if (result == 0) result = Long.compare(minor, other.minor);
            if (result == 0) result = Long.compare(patch, other.patch);
            return result;
        }
    }
}
