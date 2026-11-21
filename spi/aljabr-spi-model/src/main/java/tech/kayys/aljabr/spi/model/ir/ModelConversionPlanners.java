package tech.kayys.aljabr.spi.model.ir;

import java.util.ServiceLoader;

/**
 * Creates a conversion planner from providers visible to the supplied class
 * loader. Applications may still construct a planner with an explicit provider
 * collection when they need a controlled registry.
 */
public final class ModelConversionPlanners {

    private ModelConversionPlanners() {
    }

    public static ModelConversionPlanner discover() {
        return discover(Thread.currentThread().getContextClassLoader());
    }

    public static ModelConversionPlanner discover(ClassLoader classLoader) {
        return new ModelConversionPlanner(
                ServiceLoader.load(TargetFormatCapability.class, classLoader).stream()
                        .map(ServiceLoader.Provider::get)
                        .toList());
    }
}
