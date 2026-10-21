package tech.kayys.aljabr.spi.tokenizer;

import java.util.List;

public interface PreTokenizer {
    List<String> split(String text);
}