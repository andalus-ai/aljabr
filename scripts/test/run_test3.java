import java.util.*;
import tech.kayys.aljabr.spi.model.*;

public class run_test3 {
    public static void main(String[] args) {
        ModelFamilyPlugin broken = new ModelFamilyPlugin() {
            @Override
            public String id() {
                return "model-family/broken-descriptor";
            }

            @Override
            public ModelFamilyDescriptor descriptor() {
                throw new IllegalStateException("descriptor discovery failed");
            }
        };

        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.global();
        registry.register(broken);
        System.out.println("Is present: " + registry.plugin("broken-descriptor").isPresent());
    }
}
