import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.converter.gguf.HfConfigParser;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class HfConfigParserTokenizerTest {

    @TempDir
    Path tempDir;

    @Test
    void preservesBpeMergesForGpt2TokenizerMetadata() throws Exception {
        Files.writeString(tempDir.resolve("tokenizer_config.json"), """
                {
                  "bos_token": "<bos>",
                  "eos_token": "<eos>"
                }
                """);
        Files.writeString(tempDir.resolve("tokenizer.json"), """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {
                      "<pad>": 0,
                      "<eos>": 1,
                      "<bos>": 2,
                      "w": 3,
                      "h": 4,
                      "wh": 5,
                      "ere": 6
                    },
                    "merges": [
                      "w h",
                      ["wh", "ere"]
                    ]
                  },
                  "added_tokens": [
                    {"id": 1, "content": "<eos>"},
                    {"id": 2, "content": "<bos>"}
                  ]
                }
                """);

        HfConfigParser.TokenizerData tokenizer = HfConfigParser.parseTokenizer(tempDir);

        assertThat(tokenizer.tokenizerModel()).isEqualTo("gpt2");
        assertThat(tokenizer.bosId()).isEqualTo(2);
        assertThat(tokenizer.eosId()).isEqualTo(1);
        assertThat(tokenizer.merges()).containsExactly("w h", "wh ere");
    }
}
