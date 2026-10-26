package io.github.derekdiliu.ragkit.eval;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class RetrievalEvaluatorTests {

	private static Document doc(String id) {
		return Document.builder().id(id).text("text " + id).build();
	}

	/** Answers each query text with fixed document ids. */
	private static DocumentRetriever answering(Map<String, List<String>> idsByQuery) {
		return query -> idsByQuery.get(query.text()).stream().map(RetrievalEvaluatorTests::doc).toList();
	}

	private final List<EvalCase> cases = List.of(EvalCase.of("q1", "first", "a"), EvalCase.of("q2", "second", "b"));

	@Test
	void averagesEachMetricOverAllQueries() {
		var retriever = answering(Map.of("first", List.of("a", "x"), "second", List.of("x", "b")));

		EvalReport report = RetrievalEvaluator.of(retriever).evaluate(this.cases);

		assertThat(report.mean(Metric.MRR, 10)).isCloseTo((1.0 + 0.5) / 2, within(1e-12));
		assertThat(report.mean(Metric.HIT_RATE, 1)).isCloseTo(0.5, within(1e-12));
		assertThat(report.getResults()).extracting(EvalReport.QueryResult::rankedIds)
			.containsExactly(List.of("a", "x"), List.of("x", "b"));
	}

	@Test
	void matchesDocumentsWithACustomIdAndCountsChunksOfOneDocumentOnce() {
		// The store assigns its own ids; the source id lives in metadata and repeats per chunk.
		DocumentRetriever retriever = query -> Arrays.asList(chunk("c1", "x"), chunk("c2", "x"), chunk("c3", "a"));

		EvalReport report = RetrievalEvaluator.of(retriever)
			.documentId(document -> (String) document.getMetadata().get("source"))
			.evaluate(List.of(EvalCase.of("q1", "first", "a")));

		assertThat(report.getResults().get(0).rankedIds()).containsExactly("x", "a");
		assertThat(report.mean(Metric.MRR, 10)).isCloseTo(0.5, within(1e-12));
	}

	private static Document chunk(String id, String source) {
		return Document.builder().id(id).text("chunk " + id).metadata("source", source).build();
	}

	@Test
	void assertionPassesAtOrAboveTheMinimum() {
		var retriever = answering(Map.of("first", List.of("a"), "second", List.of("b")));

		RetrievalEvaluator.of(retriever)
			.evaluate(this.cases)
			.assertThat(Metric.RECALL, 1)
			.isAtLeast(1.0)
			.isAtLeast(0.5);
	}

	@Test
	void assertionFailureNamesTheMetricAndTheWorstQueries() {
		var retriever = answering(Map.of("first", List.of("a"), "second", List.of("x")));
		EvalReport report = RetrievalEvaluator.of(retriever).evaluate(this.cases);

		assertThatThrownBy(() -> report.assertThat(Metric.RECALL, 5).isAtLeast(0.8)).isInstanceOf(AssertionError.class)
			.hasMessage("recall@5 was 0.5000, expected at least 0.8000; lowest-scoring queries: q2 (0.00), q1 (1.00)");
	}

	@Test
	void requiresCases() {
		assertThatIllegalArgumentException().isThrownBy(() -> RetrievalEvaluator.of(q -> List.of()).evaluate(List.of()));
	}

	@Test
	void caseRequiresAPositivelyGradedRelevantDocument() {
		assertThatIllegalArgumentException().isThrownBy(() -> EvalCase.of("q1", "query"));
		assertThatIllegalArgumentException().isThrownBy(() -> new EvalCase("q1", "query", Map.of("a", 0)));
	}

}
