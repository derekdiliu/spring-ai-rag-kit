package io.github.derekdiliu.ragkit.retrieval;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

import io.github.derekdiliu.ragkit.fusion.FusionMetadata;
import io.github.derekdiliu.ragkit.fusion.RankFusion;
import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;

class HybridDocumentRetrieverTests {

	private static Document doc(String id) {
		return Document.builder().id(id).text("text " + id).build();
	}

	private static DocumentRetriever returning(String... ids) {
		List<Document> docs = java.util.Arrays.stream(ids).map(HybridDocumentRetrieverTests::doc).toList();
		return query -> docs;
	}

	@Test
	void fusesDelegatesAndAppliesTopK() {
		var hybrid = HybridDocumentRetriever.builder()
			.retriever("vector", returning("a", "b", "c"))
			.retriever("fulltext", returning("c", "d"))
			.topK(2)
			.build();

		List<Document> result = hybrid.retrieve(new Query("anything"));

		assertThat(result).extracting(Document::getId).containsExactly("c", "a");
		assertThat(result.get(0).getMetadata()).containsEntry(FusionMetadata.SOURCE_RANKS,
				Map.of("vector", 3, "fulltext", 1));
	}

	@Test
	void passesTheSameQueryIncludingContextToEveryDelegate() {
		var seenByVector = new AtomicReference<Query>();
		var seenByFulltext = new AtomicReference<Query>();
		var hybrid = HybridDocumentRetriever.builder().retriever("vector", q -> {
			seenByVector.set(q);
			return List.of();
		}).retriever("fulltext", q -> {
			seenByFulltext.set(q);
			return List.of();
		}).build();
		Query query = Query.builder()
			.text("q")
			.context(Map.<String, Object>of("vector_store_filter_expression", "lang == 'java'"))
			.build();

		hybrid.retrieve(query);

		assertThat(seenByVector.get()).isSameAs(query);
		assertThat(seenByFulltext.get()).isSameAs(query);
	}

	@Test
	void runsDelegatesConcurrently() {
		// Each delegate waits for the other to start; sequential execution would time out.
		var latch = new CountDownLatch(2);
		Function<String, DocumentRetriever> waiting = id -> q -> {
			latch.countDown();
			try {
				assertThat(latch.await(5, TimeUnit.SECONDS)).as("other delegate started").isTrue();
			}
			catch (InterruptedException ex) {
				Thread.currentThread().interrupt();
				throw new IllegalStateException(ex);
			}
			return List.of(doc(id));
		};
		var hybrid = HybridDocumentRetriever.builder()
			.retriever("one", waiting.apply("a"))
			.retriever("two", waiting.apply("b"))
			.build();

		assertThat(hybrid.retrieve(new Query("q"))).hasSize(2);
	}

	@Test
	void defaultExecutorUsesVirtualThreadsOnJava21AndDaemonThreadsBefore() throws Exception {
		var ranOn = new AtomicReference<Thread>();
		var hybrid = HybridDocumentRetriever.builder().retriever("one", q -> {
			ranOn.set(Thread.currentThread());
			return List.of();
		}).retriever("two", returning()).build();

		hybrid.retrieve(new Query("q"));

		assertThat(ranOn.get().isDaemon()).isTrue();
		if (Runtime.version().feature() >= 21) {
			// Thread.isVirtual() is not in the Java 17 API.
			assertThat(Thread.class.getMethod("isVirtual").invoke(ranOn.get())).isEqualTo(true);
		}
	}

	@Test
	void propagatesDelegateFailure() {
		var hybrid = HybridDocumentRetriever.builder().retriever("ok", returning("a")).retriever("broken", q -> {
			throw new IllegalStateException("database down");
		}).build();

		assertThatIllegalStateException().isThrownBy(() -> hybrid.retrieve(new Query("q")))
			.withMessage("database down");
	}

	@Test
	void usesConfiguredFusion() {
		var hybrid = HybridDocumentRetriever.builder()
			.retriever("vector", returning("a"))
			.retriever("fulltext", returning("b"))
			.fusion(RankFusion.weightedScore())
			.build();

		assertThat(hybrid.getFusion()).isInstanceOf(io.github.derekdiliu.ragkit.fusion.WeightedScoreFusion.class);
		assertThat(hybrid.retrieve(new Query("q"))).hasSize(2);
	}

	@Test
	void requiresAtLeastTwoDistinctlyNamedRetrievers() {
		assertThatIllegalArgumentException()
			.isThrownBy(() -> HybridDocumentRetriever.builder().retriever("vector", returning("a")).build());
		assertThatIllegalArgumentException().isThrownBy(() -> HybridDocumentRetriever.builder()
			.retriever("vector", returning("a"))
			.retriever("vector", returning("b")));
	}

}
