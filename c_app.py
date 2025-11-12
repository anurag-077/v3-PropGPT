import os
import json
import logging
import re
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List
import textwrap
import pandas as pd
import streamlit as st
import tiktoken
from dotenv import load_dotenv

os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"


# Must be the first Streamlit command
# --------------- UI CONFIG ----------------
st.set_page_config(
    page_title="PropGPT Sigmavalue",
    page_icon="🧠",
    layout="wide"
)
from fuzzywuzzy import process
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
import joblib

from config import (
    EXCEL_FILE,
    PICKLE_FILE,
    SHEET_CONFIG,
    get_category_mapping,
    get_column_mapping
)

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set Pandas future behavior
pd.set_option('future.no_silent_downcasting', True)

# Load environment variables
load_dotenv()

# Initialize global mappings
CATEGORY_MAPPING = None
COLUMN_MAPPING = None
# Resolved id columns (store for debug/UI)
RESOLVED_ID_COLS = {}


@st.cache_resource
def load_mappings(comparison_type: str):
    """Return (category_mapping, column_mapping) for a comparison type and cache the result."""
    cat_map = get_category_mapping(comparison_type)
    col_map = get_column_mapping(comparison_type)
    # Normalize column names inside the column mapping so they match normalized dataframe columns
    try:
        normalized_col_map = {}
        for key, cols in (col_map or {}).items():
            normalized_cols = [normalize_colname(str(c)) for c in cols]
            normalized_col_map[key] = normalized_cols
    except Exception:
        normalized_col_map = col_map
    return cat_map, normalized_col_map


def set_mappings_for_type(comparison_type: str) -> None:
    """Set the global mappings based on comparison type (uses cached loader)."""
    global CATEGORY_MAPPING, COLUMN_MAPPING
    # use cached loader to avoid repeated imports/processing
    cat_map, col_map = load_mappings(comparison_type)
    CATEGORY_MAPPING = cat_map
    COLUMN_MAPPING = col_map
    
def get_category_keys(category: str) -> List[str]:
    """Return mapping keys associated with a category."""
    if CATEGORY_MAPPING is None:
        raise RuntimeError("CATEGORY_MAPPING not initialized. Call set_mappings_for_type first.")
    return CATEGORY_MAPPING.get(category.lower(), [])

def get_columns_for_keys(mapping_keys: List[str]) -> Dict[str, List[str]]:
    """Return a dict of mapping_key -> column names filtered by keys."""
    if COLUMN_MAPPING is None:
        raise RuntimeError("COLUMN_MAPPING not initialized. Call set_mappings_for_type first.")
    columns_by_key: Dict[str, List[str]] = {}
    for key in mapping_keys:
        cols = COLUMN_MAPPING.get(key)
        if not cols:
            logger.warning("Mapping key '%s' missing in COLUMN_MAPPING", key)
            continue
        columns_by_key[key] = cols
    return columns_by_key

@st.cache_data(ttl=3600)
def get_filtered_dataframe(comparison_type: str):
    """Get cached dataframe filtered by comparison type"""
    if not Path(pickle_path).exists():
        df_all = initialize_dataframe()
    else:
        df_all = joblib.load(pickle_path)
        # normalize column names in case pickle was created before normalization changes
        if df_all is not None and not df_all.empty:
            df_all.columns = [normalize_colname(str(c)) for c in df_all.columns]
        
    if df_all is None or df_all.empty:
        return None
        
    # Filter for comparison type
    return df_all[df_all["__type"] == comparison_type].copy()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def flatten_columns(columns_by_key: Dict[str, List[str]]) -> List[str]:
    """Flatten dict of key->columns into a unique column list preserving order."""
    ordered: List[str] = []
    seen = set()
    for cols in columns_by_key.values():
        for col in cols:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
    return ordered

def normalize_colname(name):
    name = re.sub(r'[-\s]+', ' ', name.strip().lower())
    name = re.sub(r'\(in\s+sqft\)', '(in sqft)', name)
    return name


# File paths - using current directory
BASE_DIR = Path(__file__).parent
excel_path = BASE_DIR / EXCEL_FILE
pickle_path = BASE_DIR / PICKLE_FILE

# Initialize combined dataframe
def initialize_dataframe():
    try:
        # Always try to load fresh data from Excel
        if Path(pickle_path).exists():
            os.remove(pickle_path)
            st.info("Refreshing data from Excel file...")
        
        if not Path(excel_path).exists():
            st.error(f"❌ Excel file not found: {excel_path}")
            st.stop()
        
        st.info(f"Loading data from {excel_path}...")
        with st.spinner("Reading Excel file..."):
            dfs = pd.read_excel(excel_path, sheet_name=None)
            st.success(f"✅ Excel file loaded successfully")
        
        combined = []
        for ctype, cfg in SHEET_CONFIG.items():
            if cfg["sheet"] in dfs:
                df = dfs[cfg["sheet"]].copy()
                # Normalize column names so config id_col matches regardless of spacing/case
                df.columns = [normalize_colname(str(c)) for c in df.columns]
                df["__type"] = ctype
                combined.append(df)
        
        if not combined:
            st.error("❌ No valid sheets found in Excel file!")
            st.stop()
        
        df_all = pd.concat(combined, ignore_index=True)
        joblib.dump(df_all, pickle_path)
        return df_all
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        logger.exception("Data loading error")
        st.stop()

# Load initial data
if not Path(pickle_path).exists():
    initialize_dataframe()
        
    st.info(f"Loading data from {excel_path}...")
    with st.spinner("Reading Excel file..."):
        dfs = pd.read_excel(excel_path, sheet_name=None)
        st.success(f"✅ Excel file loaded successfully")
        st.write("Available sheets:", list(dfs.keys()))
        
        # Display sheet information
        for sheet_name, df in dfs.items():
            st.write(f"\nSheet: {sheet_name}")
            st.write(f"Columns: {df.columns.tolist()}")
            st.write(f"Rows: {len(df)}")
            
        if not any(cfg["sheet"] in dfs for cfg in SHEET_CONFIG.values()):
            st.error("❌ None of the required sheets found in Excel file!")
            st.write("Required sheets:", [cfg["sheet"] for cfg in SHEET_CONFIG.values()])
            st.stop()
    
        if not dfs:
            st.error(f"❌ No sheets found in {excel_path}")
            st.stop()
            
        # Log available sheets
        logger.info(f"Available sheets in Excel: {list(dfs.keys())}")
        logger.info(f"Expected sheets: {[cfg['sheet'] for cfg in SHEET_CONFIG.values()]}")
        
        combined = []
        for ctype, cfg in SHEET_CONFIG.items():
            sheet_name = cfg["sheet"]
            if sheet_name not in dfs:
                logger.warning(f"Sheet '{sheet_name}' not found for type '{ctype}'")
                continue
                
            df = dfs[sheet_name].copy()
            # Normalize columns on this load as well
            df.columns = [normalize_colname(str(c)) for c in df.columns]
            if cfg["id_col"] not in df.columns:
                logger.warning(f"ID column '{cfg['id_col']}' not found in sheet '{sheet_name}'")
                continue
                
            df["__type"] = ctype
            combined.append(df)
            logger.info(f"Loaded {len(df)} rows from sheet '{sheet_name}'")
        
        if not combined:
            st.error("❌ No valid data found in any sheet. Please check sheet names and ID columns.")
            st.stop()
            
        df_all = pd.concat(combined, ignore_index=True)
        logger.info(f"Combined data shape: {df_all.shape}")
        
        # Save to pickle
        joblib.dump(df_all, pickle_path)
        logger.info(f"Data saved to {pickle_path}")
        
        
# Load and clean data
def load_and_clean_data(excel_path, pickle_path, comparison_type, items=None, years=None, category=None):
    try:
        if Path(pickle_path).exists():
            df = joblib.load(pickle_path)
            # Normalize column names in case pickle was created before normalization changes
            df.columns = [normalize_colname(str(c)) for c in df.columns]
            logger.info(f"Pickle file loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        else:
            logger.error(f"Pickle file not found at {pickle_path}")
            return None, None, None
        
        # Filter by comparison type
        df = df[df["__type"] == comparison_type].drop(columns=["__type"])

        # Resolve ID column similar to get_comparison_items (support normalized and fuzzy matches)
        configured_id = SHEET_CONFIG[comparison_type]["id_col"]
        id_col = configured_id
        if id_col not in df.columns:
            cols_norm = {normalize_colname(str(c)): c for c in df.columns}
            if id_col in cols_norm:
                matched_col = cols_norm[id_col]
                logger.warning("ID column '%s' not present; using normalized match '%s'", id_col, matched_col)
                id_col = matched_col
            else:
                try:
                    choices = list(cols_norm.keys())
                    best, score = process.extractOne(id_col, choices)
                    if score >= 70:
                        matched_col = cols_norm[best]
                        logger.warning("ID column '%s' not found; fuzzy-matched to '%s' (score=%s)", id_col, matched_col, score)
                        id_col = matched_col
                    else:
                        logger.error("ID column '%s' not found for type '%s'. Available columns: %s", id_col, comparison_type, df.columns.tolist())
                        logger.debug("Expected id_col: %s | Normalized columns: %s", id_col, list(cols_norm.keys()))
                        return None, None, None
                except Exception as exc:
                    logger.exception("Error during id_col fuzzy matching: %s", exc)
                    return None, None

        # Clean and normalize ID column values
        try:
            df[id_col] = df[id_col].astype(str).str.strip().str.lower()
        except Exception:
            df[id_col] = df[id_col].astype(str)

        available_items = df[id_col].unique()
        logger.info(f"Available {comparison_type}s (sample): {list(available_items)[:20]}")

        if items:
            lowered = [i.lower() for i in items]
            df = df[df[id_col].isin(lowered)]
            if df.empty:
            # Attempt fuzzy fallback: map requested items to closest available items
                try:
                    available = [str(x).strip().lower() for x in available_items]
                    mapped = []
                    mapping_info = {}
                    for orig in lowered:
                        best, score = process.extractOne(orig, available)
                        mapping_info[orig] = (best, score)
                        # Accept matches with a reasonable score
                        if score >= 65:
                            mapped.append(best)
                    if mapped:
                        logger.info("Fuzzy-mapped requested items %s -> %s (scores: %s)", items, mapped, mapping_info)
                        df = df[df[id_col].isin(mapped)]
                except Exception as exc:
                    logger.warning("Fuzzy fallback failed: %s", exc)

            if df.empty:
                logger.error(f"No data for {comparison_type}s {items}")
                return None, None, None
            logger.info(f"Filtered data for {comparison_type}s {items}. Shape: {df.shape}")
        
        # Year filtering: if years is None -> skip year filtering entirely
        if years is None:
            logger.info("No year filtering applied (years=None)")
        else:
            # sanitize incoming years list and restrict to sensible range
            years = [y for y in years if isinstance(y, int) and 1900 <= y <= 9999]
            if years:
                df = df[df["year"].isin(years)]
                logger.info(f"Filtered data for years {years}. Shape: {df.shape}")
            else:
                logger.info("Year filter provided but resulted in empty/invalid set; skipping year filter")
        
        # Sort only by columns that exist in this dataframe
        sort_cols = [c for c in ["final location", "year"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols)
        
        if category and category != "general":
            relevant_columns = ["final location", "year"]
            category_keys = get_category_keys(category)
            category_columns = flatten_columns(get_columns_for_keys(category_keys))
            for col in df.columns:
                if col in category_columns:
                    relevant_columns.append(col)
            relevant_columns = list(dict.fromkeys(relevant_columns))
            df = df[[col for col in relevant_columns if col in df.columns]]
            logger.info(f"Filtered columns for category '{category}'. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        defaults = {
            "year": 2020,
            "total sold - igr": 0,
            "1bhk_sold - igr": 0,
            "flat total": 0,
            "shop total": 0,
            "office total": 0,
            "others total": 0,
            "1bhk total": 0,
            "<1bhk total": 0
        }
        
        df = df.infer_objects(copy=False).fillna({col: defaults.get(col, 0) for col in df.columns})
        logger.info(f"Final data shape: {df.shape}, columns: {df.columns.tolist()}")
        return df, defaults, id_col
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None


def create_documents(df, item_ids: List[str], defaults, columns_by_key: Dict[str, List[str]], years: List[int] = None, comparison_type: str = "Location", id_col: str = "final location"):
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]

    documents: List[Document] = []
    for mapping_key, columns in columns_by_key.items():
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            continue

        content_lines: List[str] = []
        for item_id in [i.lower() for i in item_ids]:
            # filter using resolved id column
            item_df = df[df[id_col] == item_id]

            # If this is project-level data (single-row per project) or there is no 'year' column,
            # extract values directly rather than building year-series
            if comparison_type.strip().lower() == "project" or "year" not in df.columns:
                # take first matching row if any
                for col in valid_cols:
                    value = defaults.get(col, 'N/A')
                    if not item_df.empty and col in item_df.columns:
                        try:
                            value = item_df.iloc[0][col]
                        except Exception:
                            value = item_df[col].iloc[0]
                    content_lines.append(f"{item_id}_{mapping_key}_{col}: {value}")
            else:
                for col in valid_cols:
                    year_values = []
                    for year in years:
                        year_df = item_df[item_df["year"] == year]
                        value = year_df[col].iloc[0] if not year_df.empty and col in year_df.columns else defaults.get(col, 'N/A')
                        year_values.append(f"{year}:{value}")
                    content_lines.append(f"{item_id}_{mapping_key}_{col}: {', '.join(year_values)}")

        if content_lines:
            documents.append(
                Document(
                    page_content="\n".join(content_lines),
                    metadata={
                        'columns': valid_cols,
                        'items': [i.lower() for i in item_ids],
                        'mapping_key': mapping_key,
                        'years': years,
                    }
                )
            )
            logger.info("Created document for mapping key %s with columns: %s", mapping_key, valid_cols)

    logger.info("Created %s documents for items: %s", len(documents), item_ids)
    return documents

def count_tokens(text, model="gpt-4o-mini"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0


@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_openai import ChatOpenAI  # local import for optional dependency

    provider = (os.getenv("USE_LLM") or "gemini").strip().lower()
    logger.info("Using LLM provider: %s", provider)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise RuntimeError(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai google-generativeai"
            ) from exc

        api_key = (os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini.")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemma-3-27b-it"),
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=8192,
            convert_system_message_to_human=True,
        )

    if provider == "nvidia":
        api_key = (os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY", "")).strip()
        if not api_key.startswith("nvapi-"):
            raise RuntimeError("Missing/invalid NVIDIA_API_KEY (expected raw 'nvapi-...' key).")
        return ChatOpenAI(
            model=os.getenv("NVIDIA_MODEL", "gemma-3-27b-it"),
            api_key=api_key,
            base_url=(os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1").strip(),
            temperature=0.3,
            max_completion_tokens=2048,
            timeout=60,
            max_retries=1,
        )

    api_key = (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")).strip()
    if not api_key or not api_key.startswith("sk-"):
        raise RuntimeError("Missing/invalid OPENAI_API_KEY.")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=0.3,
        max_completion_tokens=15000,
        max_retries=3,
    )


def planner_identify_mapping_keys(llm, query: str, candidate_keys: List[str]) -> List[str]:
    if not candidate_keys:
        return []

    sys_instr = (
        "You are a planning assistant that selects the most relevant mapping keys for answering a "
        "real-estate analytics question. Return ONLY a JSON list of mapping keys from CANDIDATE_KEYS."
    )
    prompt = f"""
    User Query: {query}

    CANDIDATE_KEYS:
    {json.dumps(candidate_keys, indent=2)}

    Rules:
    - Choose the smallest set of mapping keys that covers the metrics implied by the question.
    - Prefer specific keys over broad ones.
    - Output ONLY a JSON array (no commentary).
    """
    try:
        raw_resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw_text = getattr(raw_resp, "content", None) or str(raw_resp)
        start, end = raw_text.find("["), raw_text.rfind("]") + 1
        if start == -1 or end <= 0:
            raise ValueError("Planner did not return JSON array")
        parsed = json.loads(raw_text[start:end])
        if not isinstance(parsed, list):
            raise ValueError("Planner output is not a list")
        filtered = [key for key in parsed if key in candidate_keys]
        return filtered or candidate_keys[: min(6, len(candidate_keys))]
    except Exception as exc:
        logger.warning("[planner_identify_mapping_keys] fallback due to: %s", exc)
        query_tokens = set(re.findall(r"[\w>]+", query.lower()))
        heuristic = [
            key for key in candidate_keys
            if any(token in key.lower() for token in query_tokens)
        ]
        return heuristic or candidate_keys[: min(6, len(candidate_keys))]


def agent_pick_relevant_columns(llm, query: str, selected_keys: List[str], candidate_columns: List[str]) -> List[str]:
    """Use LLM to prune candidate columns to only those relevant to the user query."""
    if not candidate_columns:
        return []

    sys_instr = (
        "You select strictly relevant dataframe column names for the user's analytics query. "
        "Return ONLY a JSON list of exact column names from CANDIDATE_COLUMNS—no extra text."
    )
    prompt = f"""
    User Query: {query}

    Selected Mapping Keys (context only):
    {json.dumps(selected_keys, indent=2)}

    CANDIDATE_COLUMNS:
    {json.dumps(candidate_columns, indent=2)}

    Rules:
    - Choose only columns that are directly useful to answer the query (avoid generic/noise columns).
    - Keep the set small but sufficient (usually 5–20).
    - Output ONLY a JSON array of column names from CANDIDATE_COLUMNS. No markdown, no commentary.
    """
    try:
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e <= 0:
            raise ValueError("Agent did not return a JSON list.")
        picked = json.loads(raw[s:e])
        if not isinstance(picked, list):
            raise ValueError("Agent output is not a list.")
        picked = [c for c in picked if c in candidate_columns]
        picked = list(dict.fromkeys(picked))
        return picked or candidate_columns[: min(15, len(candidate_columns))]
    except Exception as exc:
        logger.warning("[agent_pick_relevant_columns] fallback due to: %s", exc)
        query_tokens = re.findall(r"\w+", query.lower())
        heuristic = [
            col for col in candidate_columns
            if any(token in col.lower() for token in query_tokens)
        ]
        return heuristic or candidate_columns[: min(15, len(candidate_columns))]


def compute_metrics(df: pd.DataFrame, mapping_keys: List[str], columns_by_key: Dict[str, List[str]], item_ids: List[str], id_col: str = "final location", comparison_type: str = "Location") -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for key in mapping_keys:
        key_columns = [col for col in columns_by_key.get(key, []) if col in df.columns]
        if not key_columns:
            continue
        metrics[key] = {}
        for col in key_columns:
            metrics[key][col] = {}
            for item_id in [i.lower() for i in item_ids]:
                item_df = df[df[id_col] == item_id]
                # If per-year data available and this is not a project-level sheet, use yearly logic
                if item_df.empty:
                    metrics[key][col][item_id] = {
                        "yearly": {},
                        "latest_year": None,
                        "latest_value": None,
                        "total": None,
                        "average": None,
                        "yoy_change": None,
                    }
                    continue
                if comparison_type.strip().lower() != "project" and "year" in df.columns:
                    item_df = item_df.sort_values("year")
                    # Coerce column to numeric to avoid string aggregation errors
                    numeric_series = pd.to_numeric(item_df[col], errors='coerce')

                    # Build yearly dict using numeric values (None if not numeric)
                    yearly: Dict[int, float] = {}
                    for _, row in item_df[["year", col]].iterrows():
                        year = int(row["year"]) if pd.notna(row["year"]) else None
                        val = pd.to_numeric(row[col], errors='coerce')
                        yearly[year] = (float(val) if pd.notna(val) else None)

                    # Latest year/value (numeric if available)
                    latest_year = int(item_df.iloc[-1]["year"]) if pd.notna(item_df.iloc[-1]["year"]) else None
                    latest_val_raw = numeric_series.iloc[-1]
                    latest_value = float(latest_val_raw) if pd.notna(latest_val_raw) else None

                    # Aggregations on numeric only
                    total_raw = numeric_series.sum(skipna=True)
                    total = float(total_raw) if pd.notna(total_raw) else None

                    avg_raw = numeric_series.mean(skipna=True)
                    average = float(avg_raw) if pd.notna(avg_raw) else None

                    # YoY change (use last two non-NaN numbers)
                    nn = numeric_series.dropna()
                    yoy_change = None
                    if nn.shape[0] >= 2:
                        yoy_change = float(nn.iloc[-1] - nn.iloc[-2])

                    metrics[key][col][item_id] = {
                        "yearly": yearly,
                        "latest_year": latest_year,
                        "latest_value": latest_value,
                        "total": total,
                        "average": average,
                        "yoy_change": yoy_change,
                    }
                else:
                    # Project-level / single-row case: take single value as latest and summary
                    val = None
                    if col in item_df.columns:
                        try:
                            raw = item_df.iloc[0][col]
                        except Exception:
                            raw = item_df[col].iloc[0]
                        val = pd.to_numeric(raw, errors='coerce') if pd.notna(raw) else None

                    latest_value = float(val) if pd.notna(val) else (float(val) if val is not None and not pd.isna(val) else None)
                    metrics[key][col][item_id] = {
                        "yearly": {},
                        "latest_year": None,
                        "latest_value": latest_value,
                        "total": latest_value,
                        "average": latest_value,
                        "yoy_change": None,
                    }
    return metrics


def build_cache_key(items: List[str], mapping_keys: List[str], columns: List[str]) -> str:
    payload = {
        "items": sorted([i.lower() for i in items]),
        "keys": sorted(mapping_keys),
        "columns": sorted(columns),
    }
    return md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_vector_store(documents: List[Document], embeddings: HuggingFaceEmbeddings, cache_key: str):
    index_dir = BASE_DIR / "vector_cache"
    index_dir.mkdir(exist_ok=True)
    vector_store_path = index_dir / cache_key

    if vector_store_path.exists():
        try:
            store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded FAISS index from %s", vector_store_path)
            return store
        except Exception as exc:
            logger.warning("Failed to load FAISS index (%s). Rebuilding.", exc)

    store = FAISS.from_documents(documents, embeddings)
    store.save_local(str(vector_store_path))
    logger.info("FAISS index saved to %s", vector_store_path)
    return store


def build_bm25_retriever(documents: List[Document]):
    try:
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = 8
        return retriever
    except Exception as exc:
        logger.warning("Failed to initialise BM25 retriever: %s", exc)
        return None


def hybrid_retrieve(query: str, mapping_keys: List[str], vector_store: FAISS, bm25_retriever, top_k: int = 6):
    retrieved = []
    seen_contents = set()

    for key in mapping_keys:
        faiss_docs = []
        if vector_store:
            try:
                faiss_docs = vector_store.similarity_search(query, k=top_k, filter={"mapping_key": key})
            except Exception as exc:
                logger.warning("FAISS search failed for key %s: %s", key, exc)

        bm25_docs = []
        if bm25_retriever:
            try:
                bm25_results = bm25_retriever.get_relevant_documents(query)
                bm25_docs = [doc for doc in bm25_results if doc.metadata.get("mapping_key") == key][:top_k]
            except Exception as exc:
                logger.warning("BM25 retrieval failed for key %s: %s", key, exc)

        combined = faiss_docs + bm25_docs
        for doc in combined:
            content_hash = md5(doc.page_content.encode()).hexdigest()
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)
            retrieved.append(doc)

    logger.info("Hybrid retrieval returned %s documents", len(retrieved))
    return retrieved


def build_prompt(question: str, comparison_type: str, items: List[str], mapping_keys: List[str], selected_columns: List[str], context: str, category_summary: str):
    # Create items list for display
    if len(items) == 1:
        items_display = items[0]
    elif len(items) == 2:
        items_display = f"{items[0]} vs {items[1]}"
    else:
        items_display = ", ".join(items[:-1]) + f" and {items[-1]}"
    
    return f"""
You are PropGPT, an elite real-estate analyst. Answer in clean, well-structured MARKDOWN.

MARKDOWN FORMATTING RULES (CRITICAL):
1. Use ### for main section headings (Executive Summary, Trend Highlights, Deep Insights)
2. Use #### for subsection headings (location names like Baner, Hinjewadi, Ravet, or categories like Flats, Offices)
3. Use - for bullet points (NOT * or ◦)
4. NEVER use bold asterisks (**text**) or italic asterisks (*text*) for headings
5. Always put a blank line between major sections only
6. NO blank lines after subheadings - content should start immediately
7. NO TABLES - use bullet points instead
8. NO separate year bullets - integrate year info into content

REQUEST
- Query: "{question}"
- Comparison Type: {comparison_type}
- {"Analyze" if len(items) == 1 else "Compare"}: {items_display} ({comparison_type.lower()}-wise, 2020–2024)
- Number of Items: {len(items)}
- Categories: {category_summary}

RETRIEVED EVIDENCE (context)
{context}

OUTPUT FORMAT (STRICT - follow exactly):

# [Title/Topic Name]

### Executive Summary
[2-3 sentences summarizing key findings]

### {"Key Metrics" if len(items) == 1 else "Trend Highlights"}
[Bullet points with key metrics and trends]

### {"Deep Insights" if len(items) == 1 else "Deep Insights"}
{"" if len(items) == 1 else f"""
For each {comparison_type.lower()}, include:

#### [Location/Item Name]
#### Flats:
- Average Price (2020): ₹X per sq. ft. → (2024): ₹Y per sq. ft. (Growth: Z%)
- [Key insight about trend]
- [Another metric or observation]

#### Offices:
- Average Price (2020): ₹X per sq. ft. → (2024): ₹Y per sq. ft. (Growth: Z%)
- [Key insight about trend]
"""}

CRITICAL DO's AND DON'Ts:
✓ DO: "Average Price (2020): ₹9,915.77 per sq. ft. → (2024): ₹12,396.71 per sq. ft."
✗ DON'T: Separate bullets for "2020" or "2024"
✓ DO: Integrate years directly into the metric statement
✗ DON'T: Add blank lines after #### headings
✓ DO: Start content immediately after heading
✗ DON'T: Use bare year numbers as bullet points
"""

def beautify_markdown(text: str) -> str:
    """
    Simple markdown cleanup - prevents automatic numbered list creation
    """
    if not text:
        return ""

    # Step 1: Prevent markdown from creating numbered lists with years
    # Escape years at the start of lines to prevent auto-numbering
    text = re.sub(r'^(\d{4}\.)', r'\\\1', text, flags=re.MULTILINE)
    
    # Step 2: Basic markdown cleanup
    text = re.sub(r'[◦•◆■●]\s+', '- ', text)
    
    # Step 3: Fix section headers
    # text = re.sub(r'^\\([^]+?)\\\s:\s*$', r'#### \1:', text, flags=re.MULTILINE)
    
    # Step 4: Ensure proper spacing for bullets
    text = re.sub(r'(?<!\n)(-\s)', r'\n\1', text)
    
    # Step 5: Fix concatenated headings (like "### Executive SummaryThe Pune")
    text = re.sub(r'(### [A-Za-z ]+)([A-Z][a-z])', r'\1\n\n\2', text)
    
    return text.strip()

def strip_tables(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    out: List[str] = []
    for ln in lines:
        s = ln.lstrip()
        # drop pipe-table rows and separator rows commonly seen in tables
        if s.startswith('|'):
            continue
        if re.match(r"^\|?\s*:?[-]{3,}:?\s*(\|\s*:?[-]{3,}:?\s*)*$", s):
            continue
        out.append(ln)
    return "\n".join(out)


# Get items for comparison type
@st.cache_data(ttl=3600)  # Cache for 60 seconds only
def get_comparison_items(comparison_type):
    try:
        if not Path(pickle_path).exists():
            df_all = initialize_dataframe()
        else:
            df_all = joblib.load(pickle_path)
        
        if df_all is None or df_all.empty:
            logger.error("No data loaded")
            return []
        
        # Filter for comparison type
        type_df = df_all[df_all["__type"] == comparison_type].copy()
        if type_df.empty:
            logger.error(f"No data for comparison type: {comparison_type}")
            return []
        
        # Get items from ID column
        id_col = SHEET_CONFIG[comparison_type]["id_col"]

        # If configured id_col isn't present, try to find a close match among columns
        if id_col not in type_df.columns:
            # try normalized exact match
            cols_norm = {normalize_colname(str(c)): c for c in type_df.columns}
            if id_col in cols_norm:
                matched_col = cols_norm[id_col]
                logger.warning("ID column '%s' not present; using normalized match '%s'", id_col, matched_col)
                id_col = matched_col
            else:
                try:
                    # fuzzy match against normalized names
                    choices = list(cols_norm.keys())
                    best, score = process.extractOne(id_col, choices)
                    if score >= 70:
                        matched_col = cols_norm[best]
                        logger.warning("ID column '%s' not found; fuzzy-matched to '%s' (score=%s)", id_col, matched_col, score)
                        id_col = matched_col
                    else:
                        logger.error("ID column '%s' not found for type '%s'. Available columns: %s", id_col, comparison_type, type_df.columns.tolist())
                        logger.debug("Expected id_col: %s | Normalized columns: %s", id_col, list(cols_norm.keys()))
                        return []
                except Exception as exc:
                    logger.exception("Error during id_col fuzzy matching: %s", exc)
                    return []

    # Build items list from resolved id_col
        RESOLVED_ID_COLS[comparison_type] = id_col
        items = sorted(type_df[id_col].dropna().astype(str).str.strip().str.lower().unique())
        logger.info(f"Found {len(items)} items for {comparison_type} using id_col '{id_col}'")
        return items
    
    except Exception as e:
        logger.exception(f"Error getting items for {comparison_type}")
        return []
        
        logger.info(f"Loading data for {comparison_type}")
        df = joblib.load(pickle_path)
        
        # Log the shape and content summary
        logger.info(f"Loaded dataframe shape: {df.shape}")
        logger.info(f"Available types in data: {df['__type'].unique().tolist()}")
        
        # Filter by comparison type
        type_df = df[df["__type"] == comparison_type]
        logger.info(f"Filtered {comparison_type} data shape: {type_df.shape}")
        
        if type_df.empty:
            logger.error(f"No data found for type: {comparison_type}")
            return []
        
        # Get the ID column
        id_col = SHEET_CONFIG[comparison_type]["id_col"]
        if id_col not in type_df.columns:
            logger.error(f"ID column '{id_col}' not found in data. Available columns: {type_df.columns.tolist()}")
            return []
        
        # Get unique items
        items = sorted(type_df[id_col].dropna().str.strip().str.lower().unique())
        
        if not items:
            logger.error(f"No items found in column '{id_col}' for {comparison_type}")
            return []
        
        logger.info(f"Found {len(items)} items for {comparison_type}: {items[:5]}...")
        return items
        
    except Exception as e:
        logger.exception(f"Error in get_comparison_items for {comparison_type}")
        return []

# --- Project Recommendations Helper ---
def get_project_recommendations(df):
    """
    Returns a list of dicts with project_name, village (final_location), and city for project search recommendations.
    """
    required_cols = ['project name', 'final_location', 'city']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")
    recs = df[required_cols].drop_duplicates().dropna()
    return recs.to_dict(orient='records')

# Streamlit UI

# Debug info at startup
st.sidebar.markdown("### Debug Information")
with st.sidebar.expander("System Status", expanded=False):
    st.write("Excel File:", excel_path)
    st.write("Pickle File:", pickle_path)
    st.write("Excel File Exists:", Path(excel_path).exists())
    st.write("Pickle File Exists:", Path(pickle_path).exists())
    st.write("Sheet Configuration:", SHEET_CONFIG)

# Professional Dark Mode CSS
st.markdown("""
<style>
    /* Color Scheme */
    :root {
        --bg-primary: #0F1419;
        --bg-secondary: #1A1F2E;
        --bg-tertiary: #252D3D;
        --accent-primary: #4A90E2;
        --accent-secondary: #6BA3F5;
        --text-primary: #E8EAED;
        --text-secondary: #B0B5C2;
        --border-color: #34394F;
        --success: #34C759;
        --error: #FF3B30;
    }
    
    /* Main App Container */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
        line-height: 1.6;
    }
    
    /* Scrollbar Styling */
    .stApp::-webkit-scrollbar {
        width: 8px;
    }
    .stApp::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    .stApp::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 4px;
    }
    .stApp::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.5rem;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    h2 {
        font-size: 1.875rem;
        color: var(--text-primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    h3 {
        font-size: 1.25rem;
        color: var(--accent-secondary);
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    h4 {
        font-size: 1rem;
        color: var(--text-primary);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Paragraph and Text */
    p, body {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    /* Main Container */
    .main {
        background-color: var(--bg-primary);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--bg-secondary);
    }
    
    /* Sidebar Text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--text-primary);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > select,
    textarea {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        padding: 10px 12px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1) !important;
        outline: none !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(74, 144, 226, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary Button Style */
    .stButton > button[kind="secondary"] {
        background-color: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: var(--border-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px 16px;
        color: var(--text-primary);
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--border-color);
    }
    
    .streamlit-expanderHeader p {
        margin: 0;
    }
    
    /* Multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: var(--accent-primary);
        border-radius: 6px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--accent-primary);
        border-bottom: 2px solid var(--accent-primary);
    }
    
    /* Containers and Sections */
    .stContainer {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--border-color);
    }
    
    /* Metrics */
    .stMetric {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent-secondary);
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Messages */
    .stSuccess {
        background-color: rgba(52, 199, 89, 0.1);
        color: var(--success);
        border: 1px solid var(--success);
        border-radius: 6px;
        padding: 12px 16px;
    }
    
    .stError {
        background-color: rgba(255, 59, 48, 0.1);
        color: var(--error);
        border: 1px solid var(--error);
        border-radius: 6px;
        padding: 12px 16px;
    }
    
    .stWarning {
        background-color: rgba(255, 159, 64, 0.1);
        color: #FF9F40;
        border: 1px solid #FF9F40;
        border-radius: 6px;
        padding: 12px 16px;
    }
    
    .stInfo {
        background-color: rgba(74, 144, 226, 0.1);
        color: var(--accent-secondary);
        border: 1px solid var(--accent-primary);
        border-radius: 6px;
        padding: 12px 16px;
    }
    
    /* Code Blocks */
    code {
        background-color: var(--bg-tertiary);
        color: #7EC699;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.85rem;
        font-family: 'Fira Code', 'Monaco', monospace;
    }
    
    pre {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px !important;
        overflow-x: auto;
    }
    
    /* Markdown Content */
    .markdown-text-container {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        margin: 12px 0;
    }
    
    /* Lists */
    ul, ol {
        color: var(--text-secondary);
    }
    
    li {
        margin-bottom: 8px;
        color: var(--text-secondary);
    }
    
    li strong {
        color: var(--text-primary);
    }
    
    /* Links */
    a {
        color: var(--accent-primary);
        text-decoration: none;
        transition: color 0.2s ease;
    }
    
    a:hover {
        color: var(--accent-secondary);
        text-decoration: underline;
    }
    
    /* Horizontal Line */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--border-color), transparent);
        margin: 24px 0;
    }
    
    /* JSON Display */
    .stJson {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px !important;
    }
    
    /* Table Styling */
    [data-testid="stTable"] {
        background-color: var(--bg-tertiary);
    }
    
    table {
        color: var(--text-secondary);
    }
    
    th {
        color: var(--text-primary);
        font-weight: 600;
        background-color: var(--bg-tertiary);
        border-color: var(--border-color);
    }
    
    td {
        border-color: var(--border-color);
    }
    
    tr:hover {
        background-color: rgba(74, 144, 226, 0.05);
    }
    
    /* Spinners and Loaders */
    .stSpinner > div {
        border-color: var(--accent-primary);
    }
    
    /* Dividers */
    .divider {
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        height: 1px;
        margin: 20px 0;
    }
    
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div style='padding: 2rem 0; border-bottom: 1px solid #34394F; margin-bottom: 2rem;'>
    <h1 style='margin: 0 0 0.5rem 0;'>Real Estate Analysis Platform</h1>
    <p style='margin: 0; color: #B0B5C2; font-size: 0.95rem;'>Advanced AI-powered comparative analysis of real estate metrics across multiple locations</p>
</div>
""", unsafe_allow_html=True)

# Check for OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

llm = get_llm()
if not llm:
    st.error("Failed to initialize LLM. Please check your API key.")
    st.stop()

# Get initial items for default comparison type
initial_comparison_type = list(SHEET_CONFIG.keys())[0]  # Get first comparison type as default
initial_items = get_comparison_items(initial_comparison_type)
if not initial_items:
    st.error(f"No items found in {EXCEL_FILE}. Please check the data file.")
    st.stop()

# Sidebar for inputs
with st.sidebar:
    st.markdown("""
    <div style='padding-bottom: 1.5rem; border-bottom: 1px solid #34394F; margin-bottom: 1.5rem;'>
        <h2 style='margin: 0 0 0.25rem 0; font-size: 1.25rem;'>Configuration</h2>
        <p style='margin: 0; color: #B0B5C2; font-size: 0.85rem;'>Set analysis parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Refresh Button
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Data Cache")
    with col2:
        if st.button("Refresh"):
            if Path(pickle_path).exists():
                os.remove(pickle_path)
                st.success("Cache cleared!")
                st.rerun()
    
    st.divider()
    
    # Comparison Type Selection
    st.write("**Comparison Type**")
    comparison_type = st.selectbox(
        "Select Type",
        options=list(SHEET_CONFIG.keys()),
        help="Choose the type of comparison to perform",
        label_visibility="collapsed"
    )
    
    # Initialize mappings based on comparison type
    set_mappings_for_type(comparison_type)
    
    # Show Excel sheet info
    with st.expander("Data Source Details"):
        st.write("Sheet:", SHEET_CONFIG[comparison_type]["sheet"])
        st.write("ID Column:", SHEET_CONFIG[comparison_type]["id_col"])
        if Path(excel_path).exists():
            df = get_filtered_dataframe(comparison_type)
            if df is not None and not df.empty:
                st.write("Columns:", len(df.columns))
                st.write("Rows:", len(df))
                configured = SHEET_CONFIG[comparison_type]["id_col"]
                resolved = configured
                if resolved not in df.columns:
                    cols_norm = {normalize_colname(str(c)): c for c in df.columns}
                    if resolved in cols_norm:
                        resolved = cols_norm[resolved]
                    else:
                        try:
                            best, score = process.extractOne(resolved, list(cols_norm.keys()))
                            if score >= 70:
                                resolved = cols_norm[best]
                            else:
                                resolved = None
                        except Exception:
                            resolved = None

                if resolved and resolved in df.columns:
                    sample_vals = df[resolved].dropna().astype(str).str.strip().str.lower().unique()[:10]
                    st.write(f"Sample values:", list(sample_vals))
            else:
                st.info("Dataset not loaded.")
    
    st.divider()
    
    # Dynamic Item Selection
    st.write(f"**Select {comparison_type}s**")
    if comparison_type.strip().lower() == "project":
        # Use formatted recommendations for project search
        df_projects, _, _ = load_and_clean_data(excel_path, pickle_path, comparison_type="Project")
        project_recs = get_project_recommendations(df_projects)
        items = [f"{rec['project name']}, {rec['final_location']}, {rec['city']}" for rec in project_recs]
        # Map display string to project_name for later use
        project_name_map = {f"{rec['project name']}, {rec['final_location']}, {rec['city']}": rec['project name'] for rec in project_recs}
        selected_display_items = st.multiselect(
            f"Projects",
            options=items,
            max_selections=5,
            help="Choose 1-5 projects",
            label_visibility="collapsed"
        )
        # For downstream logic, use only the project_name
        selected_items = [project_name_map[item] for item in selected_display_items]
        if selected_display_items:
            st.caption(f"{len(selected_display_items)} project(s) selected")
    else:
        items = get_comparison_items(comparison_type)
        selected_items = st.multiselect(
            f"Select {comparison_type}s",
            options=items,
            max_selections=5,
            help=f"Choose 1-5 {comparison_type}s",
            label_visibility="collapsed"
        )
        if selected_items:
            st.caption(f"{len(selected_items)} {comparison_type}(s) selected")
    
    st.divider()

    # Categories Selection - Dynamic based on comparison type mappings
    st.write("**Analysis Categories**")
    category_options = [key.title() for key in CATEGORY_MAPPING.keys()]
    categories = st.multiselect(
        "Categories",
        options=category_options,
        default=category_options[1:2] if len(category_options) > 1 else category_options,
        help="Choose analysis focus areas",
        label_visibility="collapsed"
    )
    
    st.divider()

    # Show candidate mapping keys for selected categories
    if categories:
        candidate_keys = []
        for category in [cat.lower() for cat in categories]:
            candidate_keys.extend(get_category_keys(category))
        candidate_keys = sorted(set(candidate_keys))
        with st.expander("Selected Category Mapping Keys", expanded=False):
            st.write("Mapping keys that will be passed to the selection agent:")
            st.write([key.title() for key in candidate_keys])

    st.divider()

    # Custom Query
    st.write("**Analysis Query**")
    query = st.text_area(
        "Query",
        placeholder="e.g., Compare demand trends for residential properties",
        help="Leave empty for automatic analysis",
        height=80,
        label_visibility="collapsed"
    )

# Main content area
st.divider()

# Analysis Button
generate_btn = st.button(
    "Generate Analysis",
    type="primary",
    use_container_width=True,
    help="Execute analysis with selected parameters"
)

if generate_btn:
    # Input validation
    if not selected_items:
        st.error(f"Please select at least one {comparison_type} to analyze.")
        st.stop()
    
    if len(selected_items) > 5:
        st.error(f"Maximum 5 {comparison_type}s can be analyzed at once.")
        st.stop()
    
    # Check for duplicates (should be prevented by multiselect but verify)
    if len(set(i.lower() for i in selected_items)) != len(selected_items):
        st.markdown(f"""
            <div style='background-color: #fce4e4; color: #cc0033; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                <strong>❌ Error:</strong> Please select different {comparison_type}s for comparison.""")

    if not categories:
        st.error("Please select at least one category to analyze.")
        st.stop()

    if not query or not query.strip():
        # Create default query based on number of items
        if len(selected_items) == 1:
            query = f"Analyze {', '.join(categories).lower()} metrics for {selected_items[0]}"
        elif len(selected_items) == 2:
            items_display = f"{selected_items[0]} and {selected_items[1]}"
            query = f"Compare {', '.join(categories).lower()} metrics for {items_display}"
        else:
            items_display = ", ".join(selected_items[:-1]) + f" and {selected_items[-1]}"
            query = f"Compare {', '.join(categories).lower()} metrics for {items_display}"
        st.info(f"Using default query: {query}")

    logger.info("Query: '%s'", query)

    try:
        llm = get_llm()
        # Extract model information for display
        provider = (os.getenv("USE_LLM") or "gemini").strip().lower()
        if provider == "gemini":
            model_name = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")
            provider_display = "Google Gemini"
        elif provider == "nvidia":
            model_name = os.getenv("NVIDIA_MODEL", "gemma-3-27b-it")
            provider_display = "NVIDIA"
        else:  # openai
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            provider_display = "OpenAI"
    except Exception as exc:
        logger.error("Failed to initialize LLM: %s", exc)
        st.error(f"LLM initialization failed: {exc}")
        st.stop()

    # Get correct mapping for selected comparison type
    cat_map, col_map = load_mappings(comparison_type)
    CATEGORY_MAPPING = cat_map
    COLUMN_MAPPING = col_map

    # For Project comparisons we do not want to restrict to 2020-2024
    req_years = None if str(comparison_type).strip().lower() == "project" else [2020, 2021, 2022, 2023, 2024]
    df, defaults, id_col = load_and_clean_data(
        excel_path,
        pickle_path,
        comparison_type=comparison_type,
        items=selected_items,
        years=req_years
    )

    if df is None or df.empty:
        st.error(f"No data found for selected {comparison_type}s.")
        # Provide helpful debug info
        try:
            if Path(pickle_path).exists():
                dbg_df = joblib.load(pickle_path)
                dbg_df.columns = [normalize_colname(str(c)) for c in dbg_df.columns]
                type_df = dbg_df[dbg_df["__type"] == comparison_type]
                st.markdown("**Debug info (sample):**")
                st.write("Available Columns:", type_df.columns.tolist())
                configured_id = SHEET_CONFIG[comparison_type]["id_col"]
                st.write("Configured ID column (from config):", configured_id)

                # Try to resolve configured id column to an actual column name
                resolved = configured_id
                if resolved not in type_df.columns:
                    cols_norm = {normalize_colname(str(c)): c for c in type_df.columns}
                    if resolved in cols_norm:
                        resolved = cols_norm[resolved]
                    else:
                        try:
                            best, score = process.extractOne(resolved, list(cols_norm.keys()))
                            if score >= 70:
                                resolved = cols_norm[best]
                            else:
                                resolved = None
                        except Exception:
                            resolved = None

                if resolved and resolved in type_df.columns:
                    sample_vals = type_df[resolved].dropna().astype(str).str.strip().str.lower().unique()[:20]
                    st.write(f"Sample values for resolved id column '{resolved}':", list(sample_vals))
                else:
                    st.write("Could not resolve the configured ID column to any available header.")
        except Exception as exc:
            st.write("Failed to load debug info:", str(exc))

        st.stop()

    logger.info("DataFrame columns after filtering: %s", df.columns.tolist())

    embeddings = get_embeddings()

    selected_categories = [cat.lower() for cat in categories]
    candidate_keys = []
    for category in selected_categories:
        candidate_keys.extend(get_category_keys(category))
    if not candidate_keys:
        candidate_keys = list(COLUMN_MAPPING.keys())
    candidate_keys = sorted(set(candidate_keys))

    planner_keys = planner_identify_mapping_keys(llm, query, candidate_keys)
    if not planner_keys:
        planner_keys = candidate_keys
    logger.info("Planner selected mapping keys: %s", planner_keys)

    columns_by_key = get_columns_for_keys(planner_keys)
    candidate_columns = flatten_columns(columns_by_key)

    picked_columns = agent_pick_relevant_columns(llm, query, planner_keys, candidate_columns)
    if not picked_columns:
        picked_columns = candidate_columns
    logger.info("Column agent selected columns: %s", picked_columns)

    filtered_columns_by_key = {}
    for key, cols in columns_by_key.items():
        chosen = [col for col in cols if col in picked_columns]
        if chosen:
            filtered_columns_by_key[key] = chosen
    if not filtered_columns_by_key:
        filtered_columns_by_key = columns_by_key

    final_mapping_keys = list(filtered_columns_by_key.keys())
    final_columns = flatten_columns(filtered_columns_by_key)

    documents = create_documents(df, selected_items, defaults, filtered_columns_by_key, comparison_type=comparison_type, id_col=id_col)
    if not documents:
        # Provide debug info to help identify why documents could not be built
        st.error("❌ Failed to build knowledge documents for the selected filters.")
        with st.expander("Debug: why no documents?", expanded=True):
            st.write("Resolved id_col:", id_col)
            st.write("Dataframe columns:", df.columns.tolist())
            st.write("Final mapping keys:", final_mapping_keys)
            st.write("Final columns candidate list:", final_columns)
            key_valid = {}
            for key, cols in filtered_columns_by_key.items():
                valid = [c for c in cols if c in df.columns]
                key_valid[key] = valid
            st.write("Per-mapping-key valid columns (present in df):", key_valid)
            # Show sample rows for the selected items (if present) to inspect values
            try:
                sel_vals = df[id_col].dropna().astype(str).str.strip().str.lower().unique()[:50]
                st.write(f"Sample unique values in '{id_col}':", list(sel_vals))
            except Exception as exc:
                st.write("Failed to sample id_col values:", str(exc))
        st.stop()

    cache_key = build_cache_key(selected_items, final_mapping_keys, final_columns)
    with st.spinner("🔄 Preparing retrieval index..."):
        vector_store = build_vector_store(documents, embeddings, cache_key)
        bm25_retriever = build_bm25_retriever(documents)

    query_context_docs = hybrid_retrieve(query, final_mapping_keys, vector_store, bm25_retriever, top_k=6)
    if not query_context_docs:
        st.error("❌ Retrieval failed to surface relevant context. Refine your query and try again.")
        st.stop()
    context = "\n\n".join(doc.page_content.strip() for doc in query_context_docs)
  
    category_summary = ", ".join(categories)

    # Show Query Intelligence outputs in UI
    with st.expander("Query Intelligence Outputs", expanded=False):
        st.markdown("#### Selected Mapping Keys")
        st.write(final_mapping_keys)
        st.markdown("#### Selected Columns (Agent)")
        st.write(final_columns)


    # Analysis Configuration Summary
    st.markdown('''
        <div style='border-top: 2px solid #34394F; padding: 1.5rem 0; margin: 2rem 0 1.5rem 0;'>
            <h3 style='color: #E8EAED; margin: 0 0 1rem 0; font-weight: 600; letter-spacing: 0.5px;'>Configuration</h3>
        </div>
    ''', unsafe_allow_html=True)
    
    formatted_prompt = build_prompt(
        question=query.strip(),
        comparison_type=comparison_type,
        items=selected_items,
        mapping_keys=final_mapping_keys,
        selected_columns=final_columns,
        context=context,
        category_summary=category_summary
    )
    
    # Display configuration in a collapsible section
    with st.expander("View Detailed Configuration", expanded=False):
        config = {
            "Query": query.strip(),
            "Items": {f"Item {i+1}": item for i, item in enumerate(selected_items)},
            "Categories": categories,
            "Selected Metrics": final_mapping_keys,
            "Data Points": final_columns
        }
        st.json(config)
 

    input_tokens = count_tokens(formatted_prompt)
    logger.info("Input tokens: %s", input_tokens)

    # Results Section
    st.markdown('''
        <div style='border-top: 2px solid #34394F; padding: 1.5rem 0; margin: 2rem 0 1.5rem 0;'>
            <h3 style='color: #E8EAED; margin: 0 0 1rem 0; font-weight: 600; letter-spacing: 0.5px;'>Analysis Results</h3>
        </div>
    ''', unsafe_allow_html=True)

    # Model indicator
    st.markdown(f"""
    <div style='background-color: #1A1F2E; border-left: 3px solid #4A90E2; padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1rem;'>
        <p style='color: #6BA3F5; margin: 0; font-size: 0.9rem; font-weight: 500;'>
            🤖 Generated by <strong>{provider_display} - {model_name}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Toggle streaming behavior (set True to stream by default)
    stream = True

    with st.spinner("Generating analysis..."):
        if stream:
            response_container = st.empty()
            full_response = ""
            for chunk in llm.stream(formatted_prompt):
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_response += chunk_text
                rendered = strip_tables(beautify_markdown(full_response))
                response_container.markdown(rendered, unsafe_allow_html=True)
            output_tokens = count_tokens(full_response)
        else:
            response = llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            output_tokens = count_tokens(response_text)
            rendered = strip_tables(beautify_markdown(response_text))
            st.markdown(rendered, unsafe_allow_html=True)

    # Metrics Section
    st.markdown('''
        <div style='border-top: 2px solid #34394F; padding: 1.5rem 0; margin: 2rem 0 1rem 0;'>
            <h3 style='color: #E8EAED; margin: 0 0 1rem 0; font-weight: 600; letter-spacing: 0.5px;'>Analysis Metrics</h3>
        </div>
    ''', unsafe_allow_html=True)

    with st.expander("Token Usage & Model Info", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Input Tokens", input_tokens)
        with col2:
            st.metric("Output Tokens", output_tokens)
        with col3:
            st.metric("Total Tokens", input_tokens + output_tokens)
        with col4:
            st.metric("Model", f"{provider_display}\n{model_name}")

    # Retrieved Sources Section
    st.markdown('''
        <div style='border-top: 2px solid #34394F; padding: 1.5rem 0; margin: 2rem 0 1rem 0;'>
            <h3 style='color: #E8EAED; margin: 0 0 1rem 0; font-weight: 600; letter-spacing: 0.5px;'>Data Sources</h3>
        </div>
    ''', unsafe_allow_html=True)

    with st.expander("Retrieved Sources", expanded=False):
        for idx, doc in enumerate(query_context_docs, start=1):
            st.markdown(f"**Source {idx}** — {doc.metadata.get('mapping_key')}")
            st.text(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
            st.caption(str(doc.metadata))
            st.markdown("---")

    st.markdown('''
    <div style="background-color: #1A1F2E; border-left: 3px solid #4A90E2; padding: 1rem; border-radius: 4px; margin: 2rem 0 1.5rem 0;">
        <p style="color: #6BA3F5; margin: 0; font-weight: 500;">Analysis completed successfully</p>
    </div>
''', unsafe_allow_html=True)

    # Show project recommendations for Project search type
    if comparison_type.strip().lower() == "project":
        try:
            project_recs = get_project_recommendations(df)
            with st.expander("Project Recommendations", expanded=False):
                for rec in project_recs:
                    st.write(f"{rec['project name']} | {rec['final_location']} | {rec['city']}")
        except Exception as e:
            st.markdown(f"<div style='background-color: #2a1a1a; border-left: 3px solid #e74c3c; padding: 1rem; border-radius: 4px;'><p style='color: #ec7063; margin: 0;'>Could not generate project recommendations: {e}</p></div>", unsafe_allow_html=True)

footer_html = '''
<div style="border-top: 1px solid #34394F; padding: 2rem 0; margin: 3rem 0 0 0; text-align: center;">
    <p style="color: #B0B5C2; margin: 0; font-size: 0.9rem; letter-spacing: 0.3px;">
        <strong>PropGPT</strong> by SigmaValue - Advanced Real Estate Analysis Platform
    </p>
</div>
'''

st.markdown(footer_html, unsafe_allow_html=True)