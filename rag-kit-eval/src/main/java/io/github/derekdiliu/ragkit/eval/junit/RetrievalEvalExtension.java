package io.github.derekdiliu.ragkit.eval.junit;

import java.util.List;

import io.github.derekdiliu.ragkit.eval.EvalDatasets;

import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.ParameterContext;
import org.junit.jupiter.api.extension.ParameterResolver;

/**
 * Resolves {@link EvalDataset} parameters. Registered automatically by the annotation, so
 * an explicit {@code @ExtendWith} is not needed.
 */
public class RetrievalEvalExtension implements ParameterResolver {

	@Override
	public boolean supportsParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
		return parameterContext.isAnnotated(EvalDataset.class)
				&& parameterContext.getParameter().getType().isAssignableFrom(List.class);
	}

	@Override
	public Object resolveParameter(ParameterContext parameterContext, ExtensionContext extensionContext) {
		String location = parameterContext.findAnnotation(EvalDataset.class).orElseThrow().value();
		return EvalDatasets.load(location);
	}

}
