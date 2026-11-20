package tech.kayys.aljabr.spi.model.ir;

import java.math.BigDecimal;
import java.util.List;
import java.util.Objects;

/**
 * Typed, serialization-neutral value used by architecture configuration.
 * Unknown vendor properties should be preserved in the configuration's
 * extension map rather than coerced into one of these standard value types.
 */
public sealed interface ConfigurationValue
        permits ConfigurationValue.StringValue,
                ConfigurationValue.IntegerValue,
                ConfigurationValue.DecimalValue,
                ConfigurationValue.BooleanValue,
                ConfigurationValue.StringListValue,
                ConfigurationValue.IntegerListValue {

    record StringValue(String value) implements ConfigurationValue {
        public StringValue {
            Objects.requireNonNull(value, "value must not be null");
        }
    }

    record IntegerValue(long value) implements ConfigurationValue {
    }

    record DecimalValue(BigDecimal value) implements ConfigurationValue {
        public DecimalValue {
            Objects.requireNonNull(value, "value must not be null");
        }
    }

    record BooleanValue(boolean value) implements ConfigurationValue {
    }

    record StringListValue(List<String> value) implements ConfigurationValue {
        public StringListValue {
            value = List.copyOf(Objects.requireNonNull(value, "value must not be null"));
            if (value.stream().anyMatch(Objects::isNull)) {
                throw new IllegalArgumentException("string list must not contain null values");
            }
        }
    }

    record IntegerListValue(List<Long> value) implements ConfigurationValue {
        public IntegerListValue {
            value = List.copyOf(Objects.requireNonNull(value, "value must not be null"));
            if (value.stream().anyMatch(Objects::isNull)) {
                throw new IllegalArgumentException("integer list must not contain null values");
            }
        }
    }

    static ConfigurationValue of(String value) {
        return new StringValue(value);
    }

    static ConfigurationValue of(long value) {
        return new IntegerValue(value);
    }

    static ConfigurationValue of(boolean value) {
        return new BooleanValue(value);
    }

    static ConfigurationValue of(BigDecimal value) {
        return new DecimalValue(value);
    }
}
