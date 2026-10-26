package io.github.derekdiliu.ragkit.eval.junit;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import io.github.derekdiliu.ragkit.eval.EvalDatasets;

import org.junit.jupiter.api.extension.ExtendWith;

/**
 * Injects the cases of a JSON Lines dataset into a {@code List<EvalCase>} test method
 * parameter. The format is described in {@link EvalDatasets}.
 *
 * <pre>{@code
 * @Test
 * void retrieverMeetsTargets(@EvalDataset("classpath:eval/questions.jsonl") List<EvalCase> cases) {
 *     EvalReport report = RetrievalEvaluator.of(retriever).evaluate(cases);
 *     report.assertThat(Metric.RECALL, 5).isAtLeast(0.80);
 * }
 * }</pre>
 */
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@ExtendWith(RetrievalEvalExtension.class)
public @interface EvalDataset {

	/** Spring resource location, e.g. {@code classpath:eval/questions.jsonl}. */
	String value();

}
