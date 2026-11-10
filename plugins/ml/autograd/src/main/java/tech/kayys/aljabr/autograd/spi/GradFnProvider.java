
package tech.kayys.aljabr.autograd.spi;

import tech.kayys.aljabr.autograd.GradFn;
import tech.kayys.aljabr.autograd.GradRegistry;
/**
 * 
 * Core interface for kayys module.
 *
 * @author bhangun
 * @since 0.1.0
 */
public interface GradFnProvider {
    void registerGradients(GradRegistry registry);
}
