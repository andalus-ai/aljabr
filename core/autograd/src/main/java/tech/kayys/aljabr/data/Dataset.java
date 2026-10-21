package tech.kayys.aljabr.data;

import java.util.Iterator;
/**
 * Interface defining contract for dataset implementations.
 *
 * @author bhangun
 * @since 0.1.0
 */


public interface Dataset<T> extends Iterable<T> {
    Iterator<T> iterator();
}