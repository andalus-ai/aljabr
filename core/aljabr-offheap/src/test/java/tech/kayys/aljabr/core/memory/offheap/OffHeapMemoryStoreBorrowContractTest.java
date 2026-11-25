package tech.kayys.aljabr.core.memory.offheap;

import tech.kayys.aljabr.core.memory.BorrowableMemoryStore;
import tech.kayys.aljabr.testing.BorrowableMemoryStoreContract;

/** Applies the shared scoped-borrow contract to the canonical off-heap store. */
class OffHeapMemoryStoreBorrowContractTest extends BorrowableMemoryStoreContract {

    @Override
    protected BorrowableMemoryStore newStore() {
        return new OffHeapMemoryStore();
    }
}
