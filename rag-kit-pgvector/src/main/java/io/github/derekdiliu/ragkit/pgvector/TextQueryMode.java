package io.github.derekdiliu.ragkit.pgvector;

/**
 * How the user's query text is turned into a PostgreSQL {@code tsquery}.
 */
public enum TextQueryMode {

	/**
	 * Match documents containing <em>any</em> of the query's terms, ranked by how many and
	 * how close together they are. Suits natural-language questions, where requiring every
	 * word (the PostgreSQL default) usually returns nothing. Stop words are dropped and
	 * terms are stemmed with the configured text search configuration; when identifier
	 * splitting is on, {@code createRecipe} also yields {@code create} and {@code recipe}.
	 */
	ANY_TERMS,

	/**
	 * {@code websearch_to_tsquery}: all terms required, {@code "quoted phrases"},
	 * {@code or}, and {@code -excluded} terms, like a search engine box. Suits keyword
	 * queries typed by people who expect that syntax. No identifier splitting is applied.
	 */
	WEB_SEARCH

}
