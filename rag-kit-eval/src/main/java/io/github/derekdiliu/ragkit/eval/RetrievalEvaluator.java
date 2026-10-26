package io.github.derekdiliu.ragkit.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.util.Assert;

/**
 * Runs every {@link EvalCase} through a {@link DocumentRetriever} and records the ranked
 * document ids, from which {@link EvalReport} computes any {@link Metric} at any cut-off.
 *
 * <pre>{@code
 * EvalReport report = RetrievalEvaluator.of(retriever).evaluate(cases);
 * report.assertThat(Metric.RECALL, 5).isAtLeast(0.80);
 * }</pre>
 *
 * Retrieve at least as many documents as the largest {@code k} you report.
 */
public final class RetrievalEvaluator {

	private final DocumentRetriever retriever;

	private final Function<Document, String> documentId;

	private RetrievalEvaluator(DocumentRetriever retriever, Function<Document, String> documentId) {
		this.retriever = retriever;
		this.documentId = documentId;
	}

	/** Evaluates {@code retriever}, matching documents to cases by {@link Document#getId()}. */
	public static RetrievalEvaluator of(DocumentRetriever retriever) {
		Assert.notNull(retriever, "retriever must not be null");
		return new RetrievalEvaluator(retriever, Document::getId);
	}

	/**
	 * Matches retrieved documents to {@link EvalCase#relevance()} ids with this function
	 * instead, e.g. a metadata key when the store assigns its own ids or splits documents
	 * into chunks. Chunks mapping to the same id count once, at the best position.
	 */
	public RetrievalEvaluator documentId(Function<Document, String> documentId) {
		Assert.notNull(documentId, "documentId must not be null");
		return new RetrievalEvaluator(this.retriever, documentId);
	}

	public EvalReport evaluate(List<EvalCase> cases) {
		Assert.notEmpty(cases, "cases must not be empty");
		List<EvalReport.QueryResult> results = new ArrayList<>(cases.size());
		for (EvalCase evalCase : cases) {
			List<String> rankedIds = this.retriever.retrieve(new Query(evalCase.query()))
				.stream()
				.map(this.documentId)
				.filter(Objects::nonNull)
				.distinct()
				.toList();
			results.add(new EvalReport.QueryResult(evalCase, rankedIds));
		}
		return new EvalReport(results);
	}

}
