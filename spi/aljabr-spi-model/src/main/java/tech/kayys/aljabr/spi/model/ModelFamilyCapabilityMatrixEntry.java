package tech.kayys.aljabr.spi.model;

/**
 * A flattened, boolean-field view of a {@link ModelFamilySupportReport}
 * designed
 * for automation, capability matrices, and CI dashboards.
 *
 * <p>
 * Create instances via {@link #from(ModelFamilySupportReport)}.
 * @author bhangun
 */
public final class ModelFamilyCapabilityMatrixEntry {

    private final String familyId;
    private final ModelFamilyBundleProfile bundleProfile;
    private final boolean causalLm;
    private final boolean multimodal;
    private final boolean tokenizer;
    private final boolean moe;
    private final boolean systemPrompt;
    private final boolean architectureAdapterPresent;
    private final boolean directSafetensorReady;
    private final int adapterCount;

    private ModelFamilyCapabilityMatrixEntry(ModelFamilySupportReport report) {
        this.familyId = report.id();
        this.bundleProfile = report.bundleProfile();
        this.causalLm = report.capabilities().contains(ModelFamilyCapability.CAUSAL_LM);
        this.multimodal = report.capabilities().contains(ModelFamilyCapability.VISION);
        this.tokenizer = report.capabilities().contains(ModelFamilyCapability.TOKENIZER);
        this.moe = report.metadata().containsKey("moe");
        this.systemPrompt = !report.capabilities().contains(ModelFamilyCapability.CHAT_TEMPLATE);
        this.architectureAdapterPresent = !report.architectureAdapterIds().isEmpty();
        this.directSafetensorReady = report.directSafetensorReady();
        this.adapterCount = report.architectureAdapterIds().size();
    }

    /**
     * Creates a {@code ModelFamilyCapabilityMatrixEntry} from a support report.
     *
     * @param report the support report; must not be {@code null}
     * @return a new matrix entry
     */
    public static ModelFamilyCapabilityMatrixEntry from(ModelFamilySupportReport report) {
        return new ModelFamilyCapabilityMatrixEntry(report);
    }

    /** The family identifier. */
    public String familyId() {
        return familyId;
    }

    public String id() {
        return familyId;
    }

    /**
     * Whether this family supports causal-language-model (text generation)
     * inference.
     */
    public boolean causalLm() {
        return causalLm;
    }

    public boolean multimodal() {
        return multimodal;
    }

    public boolean systemPrompt() {
        return systemPrompt;
    }

    /** Whether this family has a bundled tokenizer. */
    public boolean tokenizer() {
        return tokenizer;
    }

    /** Whether this family has a Mixture-of-Experts (MoE) runtime requirement. */
    public boolean moe() {
        return moe;
    }

    /** Whether at least one architecture adapter is present. */
    public boolean architectureAdapterPresent() {
        return architectureAdapterPresent;
    }

    /** Whether direct SafeTensor inference is ready ({@code READY}). */
    public boolean directSafetensorReady() {
        return directSafetensorReady;
    }

    /** The bundle profile tier, or {@code null} if unresolved. */
    public ModelFamilyBundleProfile bundleProfile() {
        return bundleProfile;
    }

    /**
     * A compact single-line summary suitable for log output, reports, and diffs.
     * Format: {@code <id>[<profile>] causal_lm=<t|f> tokenizer=<t|f> moe=<t|f>
     * safetensor=<ready|pending> adapters=<n>}
     */
    public String compactSummary() {
        String profileStr = bundleProfile != null ? bundleProfile.key() : "?";
        return familyId + "[" + profileStr + "]"
                + " causal_lm=" + causalLm
                + " tokenizer=" + tokenizer
                + " moe=" + moe
                + " safetensor=" + (directSafetensorReady ? "ready" : "pending")
                + " adapters=" + adapterCount;
    }

    @Override
    public String toString() {
        return compactSummary();
    }
}
