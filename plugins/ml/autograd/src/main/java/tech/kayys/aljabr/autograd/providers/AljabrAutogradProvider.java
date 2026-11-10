package tech.kayys.aljabr.autograd.providers;

import tech.kayys.aljabr.autograd.*;
import tech.kayys.aljabr.autograd.spi.GradFnProvider;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class AljabrAutogradProvider implements GradFnProvider {
    @Override
    public void registerGradients(GradRegistry registry) {
        registry.register("gelu", new GeluGrad());
        registry.register("add", new AddGrad());
        registry.register("mul", new MulGrad());
        registry.register("sub", new SubGrad());
        registry.register("div", new DivGrad());
        registry.register("relu", new ReluGrad());
        registry.register("sigmoid", new SigmoidGrad());
    }
}