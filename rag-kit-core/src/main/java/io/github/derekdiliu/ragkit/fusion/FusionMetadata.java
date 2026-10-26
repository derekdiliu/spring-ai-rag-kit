package io.github.derekdiliu.ragkit.fusion;

/**
 * Metadata keys written onto fused documents.
 */
public final class FusionMetadata {

	/** The fused score (also available as {@code Document#getScore()}). */
	public static final String FUSION_SCORE = "ragkit_fusion_score";

	/**
	 * A {@code Map<String, Integer>} from source name to the 1-based rank the document had
	 * in that source. Sources that did not return the document are absent.
	 */
	public static final String SOURCE_RANKS = "ragkit_source_ranks";

	private FusionMetadata() {
	}

}
