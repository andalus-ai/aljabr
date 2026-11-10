package tech.kayys.aljabr.autograd;

import tech.kayys.aljabr.ir.*;
import java.util.*;
/**
 * 
 * Core class for tech module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class GradContext {
    // value → gradient value
    private final Map<GValueId, GValueId> gradMap = new HashMap<>();
    // generated backward ops
    private final List<GOp> backwardOps = new ArrayList<>();

    public boolean hasGrad(GValueId id) {
        return gradMap.containsKey(id);
    }

    public GValueId getGrad(GValueId id) {
        return gradMap.get(id);
    }

    public void accumulate(GValueId target, GValueId newGrad) {
        if (!gradMap.containsKey(target)) {
            gradMap.put(target, newGrad);
            return;
        }
        // accumulate: g_old + g_new
        GValueId existing = gradMap.get(target);
        GValueId sum = new GValueId(target.id() + "_grad_sum_" + backwardOps.size());
        backwardOps.add(new GOp(
                "add",
                "grad_accumulate_" + target.id(),
                List.of(
                        new GValueRef(existing),
                        new GValueRef(newGrad)),
                List.of(sum),
                Map.of()));
        gradMap.put(target, sum);
    }

    public void addOp(GOp op) {
        backwardOps.add(op);
    }

    public List<GOp> ops() {
        return backwardOps;
    }

    public Map<GValueId, GValueId> gradMap() {
        return gradMap;
    }
}