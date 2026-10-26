package io.github.derekdiliu.ragkit.pgvector;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * Deterministic bag-of-words embedding for tests: each lower-cased word is hashed into one
 * of {@link #DIMENSIONS} buckets and the vector is L2-normalised. Texts sharing words get
 * high cosine similarity, which is all the integration tests need, with no model download
 * or API key.
 */
final class HashingEmbeddingModel implements EmbeddingModel {

	static final int DIMENSIONS = 64;

	@Override
	public EmbeddingResponse call(EmbeddingRequest request) {
		List<Embedding> embeddings = new ArrayList<>();
		List<String> texts = request.getInstructions();
		for (int i = 0; i < texts.size(); i++) {
			embeddings.add(new Embedding(embed(texts.get(i)), i));
		}
		return new EmbeddingResponse(embeddings);
	}

	@Override
	public float[] embed(Document document) {
		return embed(document.getText());
	}

	@Override
	public float[] embed(String text) {
		float[] vector = new float[DIMENSIONS];
		for (String word : text.toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
			if (!word.isEmpty()) {
				vector[Math.floorMod(word.hashCode(), DIMENSIONS)] += 1f;
			}
		}
		double norm = 0;
		for (float v : vector) {
			norm += v * v;
		}
		if (norm > 0) {
			float scale = (float) (1 / Math.sqrt(norm));
			for (int i = 0; i < vector.length; i++) {
				vector[i] *= scale;
			}
		}
		else {
			vector[0] = 1f;
		}
		return vector;
	}

	@Override
	public int dimensions() {
		return DIMENSIONS;
	}

}
