package tech.kayys.aljabr.gguf.loader.gguf;

import tech.kayys.aljabr.gguf.core.GGUFTensorInfo;
import java.util.*;
/**
 * Immutable record representing gguffile data.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record GGUFFile(int version, Map<String, Object> metadata, Map<String, GGUFTensorInfo> tensors) {
    
    public long getLong(String key, long defaultValue) {
        Object v = metadata.get(key);
        if (v instanceof Number n) return n.longValue();
        return defaultValue;
    }
    
    public float getFloat(String key, float defaultValue) {
        Object v = metadata.get(key);
        if (v instanceof Number n) return n.floatValue();
        return defaultValue;
    }
    
    public String getString(String key, String defaultValue) {
        Object v = metadata.get(key);
        if (v instanceof String s) return s;
        return defaultValue;
    }
    
    @SuppressWarnings("unchecked")
    public <T> List<T> getArray(String key) {
        Object v = metadata.get(key);
        if (v instanceof List<?> l) return (List<T>) l;
        return List.of();
    }
    
    public GGUFTensorInfo tensor(String name) {
        GGUFTensorInfo t = tensors.get(name);
        if (t == null) throw new IllegalArgumentException("Tensor not found: " + name);
        return t;
    }
    
    public Optional<GGUFTensorInfo> findTensor(String name) {
        return Optional.ofNullable(tensors.get(name));
    }
    
    public void printInfo() {
        System.out.println("GGUF v" + version + " | " + tensors.size() + " tensors | " + metadata.size() + " metadata");
        System.out.println("Architecture: " + getString("general.architecture", "unknown"));
        System.out.println("Context: " + getLong("llama.context_length", 0));
        System.out.println("Vocab: " + getArray("tokenizer.ggml.tokens").size());
    }
}