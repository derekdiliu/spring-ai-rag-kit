package io.github.derekdiliu.ragkit.retrieval;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import io.github.derekdiliu.ragkit.fusion.RankFusion;
import io.github.derekdiliu.ragkit.fusion.RankedList;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.util.Assert;

/**
 * A {@link DocumentRetriever} that runs several retrievers for the same query in parallel
 * and merges their results with a {@link RankFusion} strategy (Reciprocal Rank Fusion by
 * default).
 * <p>
 * Because it is a plain {@code DocumentRetriever}, it plugs straight into Spring AI's
 * {@code RetrievalAugmentationAdvisor}, which otherwise accepts only one retriever:
 *
 * <pre>{@code
 * var hybrid = HybridDocumentRetriever.builder()
 *     .retriever("vector", VectorStoreDocumentRetriever.builder().vectorStore(store).topK(20).build())
 *     .retriever("fulltext", PgFullTextDocumentRetriever.builder(jdbcTemplate).topK(20).build())
 *     .topK(5)
 *     .build();
 *
 * var advisor = RetrievalAugmentationAdvisor.builder().documentRetriever(hybrid).build();
 * }</pre>
 *
 * The {@link Query} is passed to every delegate unchanged, so a filter supplied under
 * {@code VectorStoreDocumentRetriever.FILTER_EXPRESSION} in the query context applies to
 * all of them (rag-kit retrievers honour the same key).
 */
public final class HybridDocumentRetriever implements DocumentRetriever {

	private final List<NamedRetriever> retrievers;

	private final RankFusion fusion;

	private final int topK;

	private final Executor executor;

	private HybridDocumentRetriever(Builder builder) {
		this.retrievers = List.copyOf(builder.retrievers);
		this.fusion = builder.fusion;
		this.topK = builder.topK;
		this.executor = builder.executor != null ? builder.executor : Executors.newVirtualThreadPerTaskExecutor();
	}

	public static Builder builder() {
		return new Builder();
	}

	@Override
	public List<Document> retrieve(Query query) {
		Assert.notNull(query, "query must not be null");

		List<CompletableFuture<RankedList>> futures = new ArrayList<>(this.retrievers.size());
		for (NamedRetriever named : this.retrievers) {
			futures.add(CompletableFuture.supplyAsync(
					() -> new RankedList(named.name(), named.weight(), named.retriever().retrieve(query)),
					this.executor));
		}

		List<RankedList> rankedLists = new ArrayList<>(futures.size());
		try {
			for (CompletableFuture<RankedList> future : futures) {
				rankedLists.add(future.join());
			}
		}
		catch (CompletionException ex) {
			futures.forEach(f -> f.cancel(true));
			if (ex.getCause() instanceof RuntimeException runtime) {
				throw runtime;
			}
			throw ex;
		}

		List<Document> fused = this.fusion.fuse(rankedLists);
		return fused.size() > this.topK ? List.copyOf(fused.subList(0, this.topK)) : fused;
	}

	public List<NamedRetriever> getRetrievers() {
		return this.retrievers;
	}

	public RankFusion getFusion() {
		return this.fusion;
	}

	public int getTopK() {
		return this.topK;
	}

	/**
	 * A delegate retriever with the name used in fusion metadata and the weight of its
	 * ranks.
	 */
	public record NamedRetriever(String name, double weight, DocumentRetriever retriever) {

		public NamedRetriever {
			Assert.hasText(name, "name must not be empty");
			Assert.isTrue(weight > 0 && Double.isFinite(weight), "weight must be a positive finite number");
			Assert.notNull(retriever, "retriever must not be null");
		}

	}

	public static final class Builder {

		public static final int DEFAULT_TOP_K = 5;

		private final List<NamedRetriever> retrievers = new ArrayList<>();

		private final Set<String> names = new HashSet<>();

		private RankFusion fusion = RankFusion.rrf();

		private int topK = DEFAULT_TOP_K;

		private Executor executor;

		private Builder() {
		}

		/** Adds a delegate with weight 1. */
		public Builder retriever(String name, DocumentRetriever retriever) {
			return retriever(name, 1.0, retriever);
		}

		/** Adds a delegate whose ranks count {@code weight} times during fusion. */
		public Builder retriever(String name, double weight, DocumentRetriever retriever) {
			NamedRetriever named = new NamedRetriever(name, weight, retriever);
			Assert.isTrue(this.names.add(name), () -> "duplicate retriever name: " + name);
			this.retrievers.add(named);
			return this;
		}

		public Builder fusion(RankFusion fusion) {
			Assert.notNull(fusion, "fusion must not be null");
			this.fusion = fusion;
			return this;
		}

		/** Maximum number of fused documents returned. Defaults to 5. */
		public Builder topK(int topK) {
			Assert.isTrue(topK > 0, "topK must be positive");
			this.topK = topK;
			return this;
		}

		/**
		 * Executor for the delegate calls. Defaults to a virtual thread per task, which
		 * suits the blocking I/O retrievers do.
		 */
		public Builder executor(Executor executor) {
			Assert.notNull(executor, "executor must not be null");
			this.executor = executor;
			return this;
		}

		public HybridDocumentRetriever build() {
			Assert.isTrue(this.retrievers.size() >= 2, "a hybrid retriever needs at least two delegate retrievers");
			return new HybridDocumentRetriever(this);
		}

	}

}
