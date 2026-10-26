package io.github.derekdiliu.ragkit.fusion;

import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.util.Assert;

/**
 * One ranked result list produced by a single retriever (or a single query), best match
 * first, together with the weight its ranks should carry during fusion.
 *
 * @param source a short name for where the list came from, e.g. {@code "vector"} or
 * {@code "fulltext"}; recorded in the fused documents' metadata
 * @param weight how much this list counts relative to the others (must be positive)
 * @param documents the documents, ordered from most to least relevant
 */
public record RankedList(String source, double weight, List<Document> documents) {

	public RankedList {
		Assert.hasText(source, "source must not be empty");
		Assert.isTrue(weight > 0 && Double.isFinite(weight), "weight must be a positive finite number");
		Assert.notNull(documents, "documents must not be null");
		Assert.noNullElements(documents, "documents must not contain null elements");
		documents = List.copyOf(documents);
	}

	public static RankedList of(String source, List<Document> documents) {
		return new RankedList(source, 1.0, documents);
	}

}
