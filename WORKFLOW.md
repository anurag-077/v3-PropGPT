# PropGPT - Workflow Guide

## Overview
PropGPT compares two villages by blending curated real-estate data with an LLM-driven reasoning layer. The Streamlit interface now follows a lean pipeline: collect user intent, load the scoped dataset, assemble mapping-key documents, retrieve only the necessary context, and let the LLM craft the narrative. Deterministic rollups were removed to keep the flow focused on query → evidence → answer.

## Architecture Highlights

1. **Mappings & Metadata**
   - `COLUMN_MAPPING`: business-friendly mapping keys → dataframe columns.
   - `CATEGORY_MAPPING`: high-level categories (`demand`, `supply`, `price`, `demography`) → mapping keys.
   - Helpers (`get_category_keys`, `get_columns_for_keys`, `flatten_columns`) translate user selections into concrete data slices.

2. **Data Preparation**
   - `load_and_clean_data` loads the cached pickle (or Excel), filters by villages and years (2020–2024), and normalises values.
   - `create_documents` packages yearly series for each mapping key with metadata describing the villages and columns involved.

3. **Retrieval Stack**
   - `create_embeddings` instantiates HuggingFace all-mpnet-base-v2 embeddings (CPU by default).
   - `build_vector_store` persists a FAISS index per `(villages, keys, columns)` cache key.
   - `build_bm25_retriever` complements semantic search with keyword recall.
   - `hybrid_retrieve` deduplicates FAISS + BM25 results while respecting mapping-key filters.

4. **LLM Interface**
   - `get_llm` resolves the provider (Gemini → NVIDIA → OpenAI fallback).
   - `planner_identify_mapping_keys` asks the LLM which mapping keys best satisfy the query.
   - `build_prompt` assembles the final instruction set using the user query, planner keys, and retrieved evidence.
   - `beautify_markdown` / `strip_tables` clean the streamed response before rendering.

## End-to-End Flow

1. **User Input**
   - Select two distinct villages, choose one or more categories, and optionally type a natural-language question.
   - If the query is blank, a default comparison prompt is generated.

2. **Data Loading**
   - Load the dataset and filter rows to the selected villages and supported years.
   - Retain only the columns covered by the chosen categories.

3. **Document Creation**
   - Use `COLUMN_MAPPING` to assemble one document per mapping key covering both villages.
   - Each document stores yearly value strings and metadata (`mapping_key`, `village1`, `village2`, `years`, `columns`).

4. **Vector Store Preparation**
   - Build or reuse a FAISS index and BM25 retriever for the generated documents.

5. **Query Analysis**
   - Invoke `planner_identify_mapping_keys` with the user query and candidate keys.
   - Fallback heuristics ensure at least a minimal key set is returned.

6. **Retrieval**
   - Call `hybrid_retrieve` with the planner-selected keys.
   - The retriever fetches evidence for each key (issuing separate FAISS/BM25 lookups when multiple keys are requested).

7. **LLM Generation**
   - Compile the prompt via `build_prompt`, including the query, villages, categories, mapping keys, and retrieved context.
   - Stream the LLM response (with fallback to single-shot invocation if streaming unsupported).
   - Token usage is tracked with `count_tokens`.

8. **Display**
   - Present the analysis (markdown), token metrics, and retrieved source summaries.
   - The footer keeps branding lightweight.

## Key Functions (Quick Reference)

| Function | Role |
| --- | --- |
| `get_category_keys` | Map category → mapping keys. |
| `get_columns_for_keys` | Expand mapping keys to concrete dataframe columns. |
| `create_documents` | Build `Document` objects grouped by mapping key. |
| `build_vector_store` / `build_bm25_retriever` | Prepare FAISS + BM25 hybrid retrievers. |
| `planner_identify_mapping_keys` | LLM planner for mapping key selection. |
| `hybrid_retrieve` | Merge semantic + lexical hits scoped per mapping key. |
| `build_prompt` | Compose the final instruction-rich prompt for the LLM. |
| `beautify_markdown` / `strip_tables` | Clean the streamed response before rendering. |

## Data Flow Diagram

```
User Input (villages, categories, query)
    ↓
Load & Clean Data (years 2020–2024)
    ↓
Create Documents (per mapping key)
    ↓
Build / Load FAISS + BM25
    ↓
Planner → Mapping Keys
    ↓
Hybrid Retrieval (per planner key)
    ↓
Prompt Assembly (query + context)
    ↓
LLM Generation (stream/fallback)
    ↓
Render Results (analysis, tokens, sources)
```

## Example Walk-through

**Input**
- Villages: `baner` vs `hinjewadi`
- Categories: `Demand`, `Price`
- Query: “How did flat absorption and weighted average rates differ YoY?”

**Execution**
1. Candidate mapping keys come from the selected categories.
2. Planner highlights keys such as `property type wise units sold` and `property type wise weighted average rate per sqft`.
3. Documents are generated for these keys and cached in FAISS/BM25.
4. Hybrid retrieval returns evidence snippets tagged with the chosen mapping keys.
5. The prompt combines the user query, village names, planner keys, and retrieved text.
6. Gemini (or fallback provider) streams a comparative analysis grounded in the retrieved sources.
7. The UI shows the markdown answer, token usage, and a concise list of sources.

## Benefits

- **Targeted Evidence**: Planner-driven mapping keys keep retrieval tight and relevant.
- **Repeatable Results**: Cached FAISS indexes and deterministic document construction ensure consistency.
- **Provider Agnostic**: LLM selection is controlled via environment variables, enabling easy swaps.
- **Transparent Output**: Retrieved snippets and token counts make every response auditable.
- **Extensible**: Adding new mapping keys or categories automatically flows through the pipeline.

