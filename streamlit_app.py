
import os, io, uuid, time, re, unicodedata
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import uuid

def _new_uuid():
    return str(uuid.uuid4())

import streamlit as st

from excel_parser import detect_main_sheet, scan_columns, find_rubric_columns
from providers import generate, grade_numeric, GenConfig, list_models, detect_provider, sanity_check_model

st.set_page_config(page_title="LLM Sheet Automation v3.1.2", layout="wide")

# --- prompt-source fix: use Prompt (not Philosophy) -------------------------
import re as _re_prompt_map
_PROMPT_HEADER_PATTERN = _re_prompt_map.compile(r"\b(P[1-3])\b.*\b(Original|Perturbed)\b.*\bPrompt Response\b", _re_prompt_map.I)
def resolve_prompt_col_from_response_header(response_header: str) -> str:
    m = _PROMPT_HEADER_PATTERN.search(str(response_header))
    if not m:
        return ""
    pN = m.group(1).upper()
    kind = m.group(2).title()
    return f"{pN} {kind} Prompt"
# ---------------------------------------------------------------------------


def reapply_env_from_session():
    prov_env = {
        "prov_openai": ("OPENAI_API_KEY",),
        "prov_anthropic": ("ANTHROPIC_API_KEY",),
        "prov_google": ("GOOGLE_API_KEY",),
        "prov_mistral": ("MISTRAL_API_KEY",),
        "prov_cohere": ("COHERE_API_KEY",),
        "prov_groq": ("GROQ_API_KEY",),
        "prov_dashscope": ("DASHSCOPE_API_KEY",),
        "prov_deepseek": ("DEEPSEEK_API_KEY",),
    }
    for pk, envvars in prov_env.items():
        if f"{pk}_key_value" in st.session_state and st.session_state[f"{pk}_key_value"]:
            for ev in envvars:
                os.environ[ev] = st.session_state[f"{pk}_key_value"]
reapply_env_from_session()

for k, v in {"abort": False, "run_id": "", "frozen": False, "template_bytes": None,
            "mapping_bytes": None, "df": None, "main_sheet": None,
            "rubrics_cache": [], "resolver": None, "mapping_df": None}.items():
    st.session_state.setdefault(k, v)

DASHES = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "-"]
def normalize_col(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    # remove zero-width / BOM
    for z in ["\u200b","\u200c","\u200d","\ufeff"]:
        s = s.replace(z, "")
    # unify dash variants
    for d in DASHES:
        s = s.replace(d, "-")
    # unify quotes and whitespace/newlines
    s = s.replace("\r\n","\n").replace("\r","\n").replace("\n", " ")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def _col_variants(base: str):
    b = normalize_col(base)
    variants = {b}
    # allow with/without the word "prompt" before "response"
    variants.add(b.replace(" original prompt response", " original response"))
    variants.add(b.replace(" perturbed prompt response", " perturbed response"))
    variants.add(b.replace(" prompt response", " response"))
    return variants


def build_resolver(columns) -> dict:
    mapping = {}
    for c in columns:
        norm = normalize_col(c)
        mapping[norm] = c
        mapping[norm.replace(" ", "")] = c
    return mapping

def resolve_column(name: str, resolver: dict, columns) -> str|None:
    if name in columns: return name
    key = normalize_col(name)
    if key in resolver: return resolver[key]
    key2 = key.replace(" ", "")
    if key2 in resolver: return resolver[key2]
    for k, original in resolver.items():
        if key in k:
            return original
    return None

def sidebar_provider(name, key_env, prov_key):
    with st.sidebar.expander(name, expanded=False):
        sess_key_name = f"{prov_key}_key_value"
        old = st.session_state.get(sess_key_name, os.environ.get(key_env, ""))
        key_val = st.text_input(f"{name} API Key", type="password", value=old if old else "", key=f"{prov_key}_key", disabled=st.session_state.frozen)
        if key_val and not st.session_state.frozen:
            st.session_state[sess_key_name] = key_val
            os.environ[key_env] = key_val
        if prov_key not in st.session_state:
            st.session_state[prov_key] = {"model": "", "temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 0, "threads": 2}
        if st.session_state.get(sess_key_name):
            models = list_models(name)
            st.write("Model")
            if models:
                default_model = st.session_state[prov_key]["model"] or models[0]
                st.session_state[prov_key]["model"] = st.selectbox(f"Choose model for {name}", options=models, index=(models.index(default_model) if default_model in models else 0), key=f"{prov_key}_model", disabled=st.session_state.frozen)
            else:
                st.session_state[prov_key]["model"] = st.text_input(f"Enter model ID for {name}", value=st.session_state[prov_key]["model"], key=f"{prov_key}_model_txt", disabled=st.session_state.frozen)
            if st.button(f"Sanity check {name}", key=f"ping_{prov_key}", disabled=st.session_state.frozen) and st.session_state[prov_key]["model"]:
                ok, msg = sanity_check_model(name, st.session_state[prov_key]["model"])
                st.info(f"Sanity check: {msg}")
            st.write("Overrides (optional)")
            st.session_state[prov_key]["temperature"] = st.number_input(f"{name} temperature", 0.0, 2.0, value=st.session_state[prov_key]["temperature"] or 0.0, step=0.1, key=f"{prov_key}_temp", disabled=st.session_state.frozen)
            st.session_state[prov_key]["top_p"] = st.number_input(f"{name} top-p", 0.0, 1.0, value=st.session_state[prov_key]["top_p"] or 1.0, step=0.05, key=f"{prov_key}_topp", disabled=st.session_state.frozen)
            st.session_state[prov_key]["seed"] = st.number_input(f"{name} seed (0=none)", 0, 10_000_000, value=st.session_state[prov_key]["seed"] or 0, step=1, key=f"{prov_key}_seed", disabled=st.session_state.frozen)
            st.session_state[prov_key]["max_tokens"] = st.number_input(f"{name} max tokens (0=default)", 0, 32768, value=st.session_state[prov_key]["max_tokens"] or 0, step=64, key=f"{prov_key}_mtok", disabled=st.session_state.frozen)
            st.session_state[prov_key]["threads"] = st.number_input(f"{name} threads", 1, 32, value=st.session_state[prov_key]["threads"] or 2, step=1, key=f"{prov_key}_threads", disabled=st.session_state.frozen)
        else:
            st.caption("Leave blank to disable this provider for this run.")

st.sidebar.title("Controls")
stub_mode = st.sidebar.checkbox("Stub mode (no API calls)", value=False, disabled=st.session_state.frozen)
os.environ["STUB_MODE"] = "1" if stub_mode else "0"
st.markdown(f"<div style='background:{'#fff3cd' if stub_mode else '#d1e7dd'};color:{'#664d03' if stub_mode else '#0f5132'};padding:8px 12px;border-radius:6px;margin:0 0 8px 0;font-weight:600'>{'STUB mode (no API calls)' if stub_mode else 'LIVE mode (real API calls)'}</div>", unsafe_allow_html=True)
if st.button("Force LIVE (disable stub now)"):
    stub_mode = False
    os.environ["STUB_MODE"] = "0"
    st.experimental_rerun()

global_system_prompt = st.sidebar.text_area("Global System Prompt (optional)", height=120, disabled=st.session_state.frozen)
tools_on = st.sidebar.checkbox("Enable Tools (if supported)", value=False, disabled=st.session_state.frozen)

st.sidebar.subheader("Default generation settings")
global_temperature = st.sidebar.number_input("Temperature", 0.0, 2.0, 0.0, 0.1, disabled=st.session_state.frozen)
global_top_p = st.sidebar.number_input("Top-p", 0.0, 1.0, 1.0, 0.05, disabled=st.session_state.frozen)
global_seed = st.sidebar.number_input("Seed (0 = none)", 0, 10_000_000, 0, 1, disabled=st.session_state.frozen)
global_max_tokens = st.sidebar.number_input("Max tokens per response (0 = provider default)", 0, 32768, 0, 128, disabled=st.session_state.frozen)

st.sidebar.subheader("Row range")
row_start = st.sidebar.number_input("Row start (0-index)", min_value=0, value=0, step=1, disabled=st.session_state.frozen)
row_end = st.sidebar.number_input("Row end (exclusive; 0 = to last)", min_value=0, value=0, step=1, disabled=st.session_state.frozen)

st.sidebar.subheader("Empty cell detection")
empty_rules_default = "-, —, –, NA, N/A, na, n.a., nan, ., …"
empty_rules_str = st.sidebar.text_input("Treat these values as empty (comma-separated)", value=empty_rules_default, disabled=st.session_state.frozen)
EMPTY_TOKENS = [t.strip().lower() for t in empty_rules_str.split(",") if t.strip()]
EMPTY_TOKENS.append("")

backfill_meta_only = st.sidebar.checkbox("Backfill Meta Only (no generation/scoring)", value=False, disabled=st.session_state.frozen)

st.sidebar.subheader("Parallelism")
global_threads = st.sidebar.slider("Total threads (upper bound)", 1, 48, 8, disabled=st.session_state.frozen)

st.sidebar.markdown("---")
st.sidebar.header("Providers")
def list_providers():
    return [
        ("OpenAI","OPENAI_API_KEY","prov_openai"),
        ("Anthropic","ANTHROPIC_API_KEY","prov_anthropic"),
        ("Google (Gemini)","GOOGLE_API_KEY","prov_google"),
        ("Mistral","MISTRAL_API_KEY","prov_mistral"),
        ("Cohere","COHERE_API_KEY","prov_cohere"),
        ("Groq","GROQ_API_KEY","prov_groq"),
        ("DashScope (Qwen)","DASHSCOPE_API_KEY","prov_dashscope"),
        ("DeepSeek","DEEPSEEK_API_KEY","prov_deepseek"),
    ]
for display, envvar, key in list_providers():
    with st.sidebar: sidebar_provider(display, envvar, key)

skip_no_key = st.sidebar.checkbox("Skip providers with no API key", value=True, disabled=st.session_state.frozen)

def provider_key_present(prov: str) -> bool:
    if stub_mode:
        return True
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "groq": "GROQ_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    var = env_map.get(prov, "")
    val = os.environ.get(var, "")
    return bool(val and val.strip())

def effective_model_id(mapping_model: str) -> str:
    s = (mapping_model or "").strip()
    sl = s.lower()
    def pick(prov_key: str, default: str=""):
        return (st.session_state.get(prov_key, {}) or {}).get("model") or default or s
    if ("claude" in sl and not sl.startswith("claude-")) or "anthropic" in sl:
        return pick("prov_anthropic", s)
    if ("gpt-5" in sl) or ("gpt-4" in sl) or ("openai" in sl) or (sl in {"gpt","gpt5","gpt-5"}):
        return pick("prov_openai", s)
    if "gemini" in sl or "google" in sl:
        return pick("prov_google", s)
    if "mistral" in sl or "mixtral" in sl:
        return pick("prov_mistral", s)
    if "cohere" in sl or "command" in sl:
        return pick("prov_cohere", s)
    if "llama" in sl or "groq" in sl:
        return pick("prov_groq", s)
    if "qwen" in sl or "dashscope" in sl or "ali" in sl:
        return pick("prov_dashscope", s)
    if "deepseek" in sl or "r1" in sl:
        return pick("prov_deepseek", s)
    return s

st.title("LLM Sheet Automation v3.1.2")
st.caption("Fix: assessor alias → real model mapping (e.g., 'DeepSeek‑R1' → 'deepseek-reasoner') for scoring + clearer preflight.")

st.header("Upload files")
template_file = st.file_uploader("Template Excel (.xlsx) — required", type=["xlsx"], key="template_upl", disabled=st.session_state.frozen)
mapping_file = st.file_uploader("Mapping CSV (optional)", type=["csv"], key="mapping_upl", disabled=st.session_state.frozen)

if template_file is not None and not st.session_state.frozen:
    st.session_state.template_bytes = template_file.getvalue()
if mapping_file is not None and not st.session_state.frozen:
    st.session_state.mapping_bytes = mapping_file.getvalue()

def load_excel_from_session():
    if st.session_state.template_bytes is None:
        return None, None
    xl = pd.ExcelFile(io.BytesIO(st.session_state.template_bytes))
    main_sheet = detect_main_sheet(xl)
    df = xl.parse(main_sheet)
    return xl, (main_sheet, df)

if st.session_state.df is None and st.session_state.template_bytes is not None:
    xl, pair = load_excel_from_session()
    if pair:
        st.session_state.main_sheet, st.session_state.df = pair
        st.success(f"Detected main sheet: {st.session_state.main_sheet} — {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")

if st.session_state.df is not None:
    df = st.session_state.df
    main_sheet = st.session_state.main_sheet
    st.success(f"Active sheet: {main_sheet} — {df.shape[0]} rows × {df.shape[1]} cols")

    if st.session_state.mapping_df is None:
        if st.session_state.mapping_bytes is not None:
            try:
                st.session_state.mapping_df = pd.read_csv(io.BytesIO(st.session_state.mapping_bytes))
                st.info(f"Loaded mapping CSV with {st.session_state.mapping_df.shape[0]} rows.")
            except Exception as e:
                st.warning(f"Failed to read mapping CSV: {e}")
        else:
            st.session_state.mapping_df = scan_columns(df)

    mapping = st.session_state.mapping_df
    resolver = build_resolver(df.columns)
    st.session_state.resolver = resolver
    if "resolved_column" not in mapping.columns:
        mapping["resolved_column"] = mapping["column"].apply(lambda x: resolve_column(str(x), resolver, df.columns))

    with st.expander("Column detection", expanded=False):
        def detect_prompt_col(df: pd.DataFrame):
            for c in df.columns:
                cl = str(c).lower()
                if "prompt" in cl and "rubric" not in cl:
                    return c
            return df.columns[0]
        default_prompt = detect_prompt_col(df)
        prompt_col = st.selectbox("Prompt column", options=list(df.columns), index=list(df.columns).index(default_prompt) if default_prompt in df.columns else 0, disabled=st.session_state.frozen)
        rubrics = find_rubric_columns(df)
        st.session_state.rubrics_cache = rubrics
        st.caption(f"Detected {len(rubrics)} rubric column(s).")
        default_rubric = st.selectbox("Default rubric (fallback)", options=(rubrics if rubrics else ["<none>"]), disabled=st.session_state.frozen)

    st.subheader("Preflight")
    rs = int(row_start)
    re_ = int(row_end) if int(row_end)>0 else len(df)
    rows = list(range(max(rs,0), min(re_, len(df))))

    def is_empty(v: str) -> bool:
        s = str(v).strip().lower()
        return s in set(EMPTY_TOKENS)

    resp_map = mapping[(mapping["role"].astype(str).str.startswith("response")) & mapping["resolved_column"].notna()].copy()
    score_map = mapping[(mapping["role"].astype(str).eq("score")) & mapping["resolved_column"].notna()].copy()

    resp_map["effective_model"] = resp_map["model"].apply(effective_model_id)
    score_map["effective_assessor"] = score_map["assessor"].apply(lambda x: effective_model_id(str(x)) if str(x).strip() else "")

    pf_rows = []
    total_resp = total_score = 0
    skipped_no_key_count = 0
    total_prompt_tokens = 0

    def tokens_estimate(s: str) -> int:
        return max(1, int(len(s)/4)) if isinstance(s, str) else 0

    for r in rows:
        _row_prompt_fallback = str(df.at[r, prompt_col]) if prompt_col in df.columns else ""
        for _, mrow in resp_map.iterrows():
            col = mrow["resolved_column"]; model = mrow.get("effective_model","") or mrow.get("model","")
            if not str(model).strip():
                continue
            prov = detect_provider(str(model))
            if skip_no_key and not provider_key_present(prov):
                skipped_no_key_count += 1
                pf_rows.append({"row": r, "type": "response", "column": col, "model": model, "skipped_no_key": True})
                continue
            if is_empty(df.at[r, col]):
                total_resp += 1
                _res_prom_col = resolve_prompt_col_from_response_header(col)
                _prompt_for_this = str(df.at[r, _res_prom_col]) if (_res_prom_col and _res_prom_col in df.columns) else _row_prompt_fallback
                total_prompt_tokens += tokens_estimate(_prompt_for_this)
                pf_rows.append({"row": r, "type": "response", "column": col, "model": model, "skipped_no_key": False})
        for _, srow in score_map.iterrows():
            scol = srow["resolved_column"]
            if not is_empty(df.at[r, scol]):
                continue
            assessor_eff = str(srow.get("effective_assessor","")).strip()
            prov_assessor = detect_provider(assessor_eff) if assessor_eff else None
            if skip_no_key and assessor_eff and not provider_key_present(prov_assessor):
                skipped_no_key_count += 1
                pf_rows.append({"row": r, "type": "score", "column": scol, "model": assessor_eff, "skipped_no_key": True})
                continue
            src_model = srow.get("model","")
            candidates = [c for c in df.columns if ("generated by" in normalize_col(c) and normalize_col(src_model) in normalize_col(c) and any(b in normalize_col(c) for b in _col_variants(scol)))]
            if not candidates:
                continue
            resp_col = candidates[0]
            ans = str(df.at[r, resp_col]).strip()
            if not ans:
                continue
            rc = default_rubric if default_rubric in df.columns else (rubrics[0] if rubrics else None)
            if not rc:
                continue
            total_score += 1
            pf_rows.append({"row": r, "type": "score", "column": scol, "model": assessor_eff or "(default grader)", "skipped_no_key": False})

    pf_df = pd.DataFrame(pf_rows)
    st.write(f"Planned response cells: {total_resp} | Planned score cells: {total_score} | Skipped (no key): {skipped_no_key_count}")
    if not pf_df.empty:
        st.dataframe(pf_df.head(100), use_container_width=True)
        st.download_button("Download preflight execution plan (CSV)", pf_df.to_csv(index=False).encode("utf-8-sig"), file_name="preflight_plan.csv", mime="text/csv")

    st.subheader("Run controls")
    existing_run_id = st.text_input("Use existing Run ID to resume (leave blank to start a new run)", value=st.session_state.run_id, disabled=st.session_state.frozen)
    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        run_btn = st.button("Run", disabled=st.session_state.frozen)
    with colB:
        abort_btn = st.button("Abort")
    with colC:
        thaw_btn = st.button("Thaw (unlock controls)")
    with colD:
        download_log_btn = st.button("Download run log")
    with colE:
        sanity_all_btn = st.button("Sanity check ALL selected providers")

    if abort_btn:
        st.session_state.abort = True
        st.warning("Abort requested. Current tasks will finish; new tasks will not start.")
    if thaw_btn:
        st.session_state.frozen = False
        st.success("Controls unlocked. You can change inputs again.")
    if download_log_btn and st.session_state.run_id:
        path = f"runlog_{st.session_state.run_id}.csv"
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button("Download current run log CSV", data=f.read(), file_name=path, mime="text/csv")
    if sanity_all_btn:
        msgs = []
        for _, name, key in [("prov_openai","OpenAI","OPENAI_API_KEY"),
                             ("prov_anthropic","Anthropic","ANTHROPIC_API_KEY"),
                             ("prov_google","Google","GOOGLE_API_KEY"),
                             ("prov_mistral","Mistral","MISTRAL_API_KEY"),
                             ("prov_cohere","Cohere","COHERE_API_KEY"),
                             ("prov_groq","Groq","GROQ_API_KEY"),
                             ("prov_dashscope","DashScope","DASHSCOPE_API_KEY"),
                             ("prov_deepseek","DeepSeek","DEEPSEEK_API_KEY")]:
            conf = st.session_state.get(_, {})
            model = conf.get("model")
            present = bool(os.environ.get(key,""))
            if model and present and not (os.environ.get("STUB_MODE","0")=="1"):
                ok, msg = sanity_check_model(name, model)
                msgs.append(f"{name}: {model} → {msg}")
            else:
                msgs.append(f"{name}: skipped (no key/model or in STUB).")
        st.info("\n".join(msgs))

    def log_row(run_id: str, **kw):
        import csv
        path = f"runlog_{run_id}.csv"
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts","run_id","row","column","provider","model","action","status","latency_ms","cost_tokens_est","error","mode"])
            if write_header:
                w.writeheader()
            row = {"ts": datetime.utcnow().isoformat(), "run_id": run_id, "latency_ms": 0, "cost_tokens_est": 0, "error": "", "mode": ("STUB" if os.environ.get("STUB_MODE","0")=="1" else "LIVE")}
            row.update(kw)
            w.writerow(row)

    if run_btn:
        if pf_df.empty:
            st.warning("No tasks in this row range. Adjust Row start/Row end or check preflight reasons.")
        else:
            st.session_state.abort = False
            if global_system_prompt:
                os.environ["GLOBAL_SYSTEM_PROMPT"] = global_system_prompt
            run_id = existing_run_id or str(uuid.uuid4())
            st.session_state.run_id = run_id

            progress = st.progress(0.0, text="Responses: starting")
            tasks = []
            with ThreadPoolExecutor(max_workers=int(global_threads)) as ex:
                total_tasks = 0
                for r in rows:
                    if st.session_state.abort: break
                    prompt = str(df.at[r, prompt_col]) if prompt_col in df.columns else ""
                    if not isinstance(prompt, str) or not prompt.strip():
                        continue
                    for _, mrow in resp_map.iterrows():
                        col = mrow["resolved_column"]; model = mrow.get("effective_model","") or mrow.get("model","")
                        if not str(model).strip(): continue
                        prov = detect_provider(str(model))
                        if skip_no_key and not provider_key_present(prov):
                            log_row(run_id, row=r, column=col, provider=prov, model=str(model), action="response", status="skipped_no_key")
                            continue
                        v = str(df.at[r, col]).strip().lower()
                        if v not in set(EMPTY_TOKENS):
                            continue
                        cfg = GenConfig(temperature=global_temperature, top_p=global_top_p, seed=(None if global_seed==0 else int(global_seed)), tools_on=tools_on, max_tokens=(None if global_max_tokens==0 else int(global_max_tokens)))
                        t0 = time.time()
                        def worker(rr=r, cc=col, mm=str(model), c=cfg, t0=t0):
                            try:
                                _res_prom_col = resolve_prompt_col_from_response_header(cc)
                                _prompt_for_this = (
                                    str(df.at[rr, _res_prom_col])
                                    if (_res_prom_col and _res_prom_col in df.columns)
                                    else _row_prompt_fallback
                                )
                                out = generate(mm, _prompt_for_this, system=os.getenv("GLOBAL_SYSTEM_PROMPT",""), cfg=c)
                                return ("ok", rr, cc, mm, int((time.time()-t0)*1000), out, "")
                            except Exception as e:
                                return ("err", rr, cc, mm, int((time.time()-t0)*1000), "", f"{type(e).__name__}: {e}")
                        tasks.append(ex.submit(worker))
                        total_tasks += 1
                done = 0
                for fut in as_completed(tasks):
                    kind, r, col, model_name, latency, text, err = fut.result()
                    if kind == "ok":
                        if str(df.at[r, col]).strip().lower() in set(EMPTY_TOKENS):
                            df.at[r, col] = text
                            suffixes = ["Track (Tools On/Off)", "Gen Config (Temp/Top-p/Seed)", "Run ID (UUID)", "Timestamp (ISO-8601)"]
                            cols = list(df.columns)
                            for suffix in suffixes:
                                for c in cols:
                                    if any(b in normalize_col(c) for b in _col_variants(col)) and normalize_col(suffix) in normalize_col(c):
                                        if str(df.at[r, c]).strip().lower() in set(EMPTY_TOKENS):
                                            if suffix.startswith("Track"):
                                                df.at[r, c] = ("Tools On" if tools_on else "Tools Off")
                                            elif suffix.startswith("Gen Config"):
                                                seed_str = "" if (global_seed==0) else str(global_seed)
                                                df.at[r, c] = f"temp={global_temperature}; top_p={global_top_p}; seed={seed_str}"
                                            elif suffix.startswith("Run ID"):
                                                df.at[r, c] = run_id
                                            elif suffix.startswith("Timestamp"):
                                                df.at[r, c] = datetime.utcnow().isoformat()
                            log_row(run_id, row=r, column=col, provider=detect_provider(model_name), model=model_name, action="response", status="written", latency_ms=latency)
                        else:
                            log_row(run_id, row=r, column=col, provider=detect_provider(model_name), model=model_name, action="response", status="skipped_nonempty", latency_ms=latency)
                    else:
                        log_row(run_id, row=r, column=col, provider=detect_provider(model_name), model=model_name, action="response", status="error", latency_ms=latency, error=err)
                    done += 1
                    progress.progress(done/max(1,total_tasks), text=f"Responses: {done}/{total_tasks}")

            progress = st.progress(0.0, text="Scores: starting")
            tasks = []
            total_tasks = 0
            with ThreadPoolExecutor(max_workers=int(global_threads)) as ex:
                for r in rows:
                    if st.session_state.abort: break
                    for _, srow in score_map.iterrows():
                        score_col = srow["resolved_column"]
                        v = str(df.at[r, score_col]).strip().lower()
                        if v not in set(EMPTY_TOKENS):
                            continue
                        assessor_raw = str(srow.get("assessor","")).strip()
                        assessor = effective_model_id(assessor_raw) if assessor_raw else ""
                        if assessor:
                            prov_assessor = detect_provider(assessor)
                            if skip_no_key and not provider_key_present(prov_assessor):
                                log_row(run_id, row=r, column=score_col, provider=prov_assessor, model=assessor, action="score", status="skipped_no_key")
                                continue
                        src_model = srow.get("model","")
                        candidates = [c for c in df.columns if ("generated by" in normalize_col(c) and normalize_col(src_model) in normalize_col(c) and any(b in normalize_col(c) for b in _col_variants(score_col)))]
                        if not candidates:
                            candidates = [c for c in df.columns if ("generated by" in normalize_col(c) and normalize_col(src_model) in normalize_col(c))]
                            if not candidates: continue
                        if not candidates: continue
                        resp_col = candidates[0]
                        answer = str(df.at[r, resp_col]).strip()
                        if not answer: continue
                        def choose_rubric_for(df: pd.DataFrame, resp_col: str, rubrics: list, default_rubric: str|None):
                            nresp = normalize_col(resp_col)
                            m = re.search(r"\bP\s*([0-9]+)\b", nresp)
                            if m:
                                ptag = m.group(1)
                                for r in rubrics:
                                    if re.search(rf"\bP\s*{ptag}\b", normalize_col(r)):
                                        return r
                            if "perturbed" in nresp:
                                for r in rubrics:
                                    if "perturb" in normalize_col(r):
                                        return r
                            if "original" in nresp:
                                for r in rubrics:
                                    nr = normalize_col(r)
                                    if "original" in nr:
                                        return r
                                for r in rubrics:
                                    if "perturb" not in normalize_col(r):
                                        return r
                            if default_rubric and default_rubric in df.columns:
                                return default_rubric
                            return rubrics[0] if rubrics else None
                        rubric_col = (choose_rubric_for(df, resp_col, st.session_state.rubrics_cache, default_rubric if default_rubric in df.columns else None))
                        if not rubric_col: continue
                        rubric = str(df.at[r, rubric_col]).strip()
                        if not rubric: continue
                        t0 = time.time()
                        assessor_eff = str(srow.get("effective_assessor","")).strip() or (effective_model_id(str(assessor)) if str(assessor).strip() else "")
                        def worker(rr=r, sc=score_col, rb=rubric, ans=answer, am=assessor_eff, t0=t0):
                            try:
                                g = grade_numeric(rb, ans, am) if am else grade_numeric(rb, ans, None)
                                return ("ok", rr, sc, int((time.time()-t0)*1000), g, "")
                            except Exception as e:
                                return ("err", rr, sc, int((time.time()-t0)*1000), "", f"{type(e).__name__}: {e}")
                        tasks.append(ex.submit(worker))
                        total_tasks += 1
                done = 0
                for fut in as_completed(tasks):
                    kind, r, s_col, latency, grade, err = fut.result()
                    if kind == "ok" and grade and (str(df.at[r, s_col]).strip().lower() in set(EMPTY_TOKENS)):
                        df.at[r, s_col] = grade
                        suffixes = ["Track (Tools On/Off)", "Gen Config (Temp/Top-p/Seed)", "Run ID (UUID)", "Timestamp (ISO-8601)"]
                        cols = list(df.columns)
                        for suffix in suffixes:
                            for c in cols:
                                if any(b in normalize_col(c) for b in _col_variants(s_col)) and normalize_col(suffix) in normalize_col(c):
                                    if str(df.at[r, c]).strip().lower() in set(EMPTY_TOKENS):
                                        if suffix.startswith("Track"):
                                            df.at[r, c] = "Tools Off"
                                        elif suffix.startswith("Gen Config"):
                                            df.at[r, c] = "temp=0.0; top_p=1.0; seed="
                                        elif suffix.startswith("Run ID"):
                                            df.at[r, c] = _new_uuid()
                                        elif suffix.startswith("Timestamp"):
                                            df.at[r, c] = datetime.utcnow().isoformat()
                        log_row(run_id, row=r, column=s_col, provider="assessor", model=assessor or "(default)", action="score", status="written", latency_ms=latency)
                    else:
                        log_row(run_id, row=r, column=s_col, provider="assessor", model=assessor or "(default)", action="score", status=("error" if err else "skipped_nonempty_or_empty_grade"), latency_ms=latency, error=err or "")
                    done += 1
                    progress.progress(done/max(1,total_tasks), text=f"Scores: {done}/{total_tasks}")

            st.session_state.df = df
            st.session_state.frozen = True

            out_buf = io.BytesIO()
            xl = pd.ExcelFile(io.BytesIO(st.session_state.template_bytes))
            with pd.ExcelWriter(out_buf, engine="openpyxl") as wr:
                for sname in xl.sheet_names:
                    d = xl.parse(sname)
                    if sname == main_sheet:
                        d = df
                    d.to_excel(wr, sheet_name=sname, index=False)

            st.download_button("Download updated Excel", data=out_buf.getvalue(), file_name=f"filled_only_empty_{main_sheet}_{int(time.time())}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success(f"Run complete. Run ID: {st.session_state.run_id}. UI frozen to prevent accidental resets. Click 'Thaw' to modify settings.")
else:
    st.info("Upload the Template Excel to configure and run.")
