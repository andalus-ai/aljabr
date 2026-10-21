import tech.kayys.aljabr.gguf.loader.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;

public class GemmaTensors {
    public static void main(String[] args) throws Exception {
        Path p = Path.of(
                "~/.aqli/models/blobs/unsloth/gemma-4-12b-it-GGUF/unsloth__gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf");
        Arena arena = Arena.ofShared();
        GGUFModel model;
        try (GGUFReader r = new GGUFReader(p, arena)) {
            model = new GGUFParser().parse(r.segment(), arena);
        }
        for (var t : model.tensors()) {
            if (t.name().startsWith("blk.0.") || t.name().contains("norm") || t.name().contains("embd")) {
                System.out.println(t.name() + " -> type=" + t.typeId());
            }
        }
    }
}