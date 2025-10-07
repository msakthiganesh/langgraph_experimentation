# -*- coding: utf-8 -*-

import os
import json
import time
import re
import html
import base64
import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import snowflake.connector
from snowflake.connector import errors as sf_errors
from cfgenerator import CFGeneratorSDK

load_dotenv()

# ================================================================
# Environment-based configuration for NL2SQL
# ================================================================

GAIP_MODEL_ID = os.getenv("GAIP_MODEL_ID", "ed8005_openai_gpt4o")
GAIP_EXP_NAME = os.getenv("GAIP_EXP_NAME", "nl2sql_EXP_NAME_prod_ci")
GAIP_EXP_ID = os.getenv("GAIP_EXP_ID", "rag_ci")
GAIP_EXP_DESC = os.getenv("GAIP_EXP_DESC", "ChatBot to assist Business")
DEFAULT_MAX_TOKENS = int(os.getenv("GAIP_MAX_TOKENS", "1500"))
DEFAULT_TEMPERATURE = float(os.getenv("GAIP_TEMPERATURE", "0.1"))

sdk = CFGeneratorSDK(GAIP_EXP_NAME, GAIP_EXP_ID, GAIP_EXP_DESC)

# ================================================================
# Snowflake connection helper
# ================================================================


def get_snowflake_connection():
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER", "YOUR_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD", "YOUR_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT", "YOUR_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "YOUR_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE", "YOUR_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "YOUR_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE", "YOUR_ROLE"),
        session_parameters={
            "QUERY_TIMEOUT": 30,
        },
    )
    return conn


# Load prompt file
try:
    with open('data/SYSTEM_PROMPT_NL_SQL_2.txt', 'r') as sys_prompt:
        SYSTEM_PROMPT_NL_SQL = sys_prompt.read()
except Exception as e:
    print(f"Unable to load prompt file: {e}")

# ================================================================
# SQL normalization, guards, rewrites
# ================================================================

_CODE_BLOCK_RE = re.compile(
    r"```(?:sql)?\s*(.+?)```", flags=re.IGNORECASE | re.DOTALL)


def normalize_sql(text: Any) -> str:
    """
    Normalize LLM output to raw SQL candidate:
    - Extract first fenced code block if present
    - HTML-unescape up to 3 times (e.g., &gt; -> >)
    - Replace non-breaking spaces and trim
    """
    if not isinstance(text, str):
        text = str(text)

    m = _CODE_BLOCK_RE.search(text)
    if m:
        text = m.group(1)

    for _ in range(3):
        new_text = html.unescape(text)
        if new_text == text:
            break
        text = new_text

    return text.replace("\u00A0", " ").strip()


def strip_sql_comments(s: str) -> str:
    """Remove SQL line and block comments."""
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s


def remove_quoted_strings(s: str) -> str:
    """
    Replace string literal contents with spaces to avoid false keyword matches.
    Handles single quotes with '' escaping and basic double quotes.
    """
    out = []
    i = 0
    n = len(s)
    in_single = False
    in_double = False
    while i < n:
        ch = s[i]
        if in_single:
            if ch == "'" and i + 1 < n and s[i + 1] == "'":
                out.append(" ")  # mask escaped single quote
                i += 2
                continue
            elif ch == "'":
                in_single = False
                out.append("'")
                i += 1
            else:
                out.append(" ")
                i += 1
        elif in_double:
            if ch == '"':
                in_double = False
                out.append('"')
                i += 1
            else:
                out.append(" ")
                i += 1
        else:
            if ch == "'":
                in_single = True
                out.append("'")
                i += 1
            elif ch == '"':
                in_double = True
                out.append('"')
                i += 1
            else:
                out.append(ch)
                i += 1
    return "".join(out)


def split_first_statement_safely(sql: str) -> str:
    """
    Return the first top-level SQL statement, splitting on semicolons
    that are NOT inside quotes or parentheses.
    """
    depth = 0
    in_single = False
    in_double = False
    n = len(sql)
    i = 0
    while i < n:
        ch = sql[i]
        if in_single:
            if ch == "'" and not (i + 1 < n and sql[i + 1] == "'"):
                in_single = False
        elif in_double:
            if ch == '"':
                in_double = False
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(depth - 1, 0)
            elif ch == ";" and depth == 0:
                return sql[:i].strip()
        i += 1
    return sql.strip()


def extract_first_sql_statement(text: str) -> str:
    """
    Normalize -> find first WITH/SELECT -> return only the first statement.
    """
    t = normalize_sql(text)
    m = re.search(r"(?is)\b(select|with)\b", t)
    if m:
        t = t[m.start():]
    return split_first_statement_safely(t)


def is_select_only_strict(sql: str) -> bool:
    """
    Strict read-only guard:
    - Must start with WITH or SELECT
    - After stripping comments and quoted strings, must not contain any DML/DDL/etc.
    Note: 'desc' is NOT forbidden to avoid blocking DESCRIBE.
    """
    cleaned = strip_sql_comments(sql).strip()
    if not re.match(r"^\s*(with|select)\b", cleaned, flags=re.IGNORECASE):
        return False

    scan = remove_quoted_strings(cleaned).lower()
    forbidden = (
        r"\b(insert|update|delete|merge|create|alter|drop|truncate|copy|call|grant|revoke|"
        r"undrop|replace|execute|put|get|use|describe)\b"
    )

    return re.search(forbidden, scan) is None


def _rewrite_filter_once(sql: str) -> Tuple[str, bool]:
    """
    Rewrite one occurrence of:
        FUNC(expr) FILTER (WHERE condition)
    to Snowflake conditional aggregation with IFF.
    Supports AVG, SUM, COUNT.
    """
    low = sql.lower()
    idx = low.find(" filter ")
    if idx == -1:
        idx = low.find(" filter(")
        if idx == -1:
            return sql, False

    # Find the ')' that closes the FUNC(expr) before FILTER
    fstart = low.rfind("filter", 0, idx + 1)
    i = fstart - 1
    while i >= 0 and sql[i].isspace():
        i -= 1
    if i < 0 or sql[i] != ")":
        return sql, False

    # Match the '(' for FUNC(expr)
    depth = 0
    j = i
    expr_open = None
    while j >= 0:
        if sql[j] == ')':
            depth += 1
        elif sql[j] == '(':
            depth -= 1
            if depth == 0:
                expr_open = j
                break
        j -= 1
    if expr_open is None:
        return sql, False

    expr_close = i
    expr_str = sql[expr_open + 1:expr_close].strip()

    # Find FUNC name
    k = expr_open - 1
    while k >= 0 and sql[k].isspace():
        k -= 1
    func_end = k + 1
    while k >= 0 and (sql[k].isalpha() or sql[k] == '_'):
        k -= 1
    func_start = k + 1
    func_name = sql[func_start:func_end].strip().upper()
    if func_name not in ("AVG", "SUM", "COUNT"):
        return sql, False

    # Parse FILTER(WHERE cond)
    p = low.find("(", fstart)
    if p == -1:
        return sql, False
    q = p + 1
    while q < len(sql) and sql[q].isspace():
        q += 1
    if low[q:q + 5] != "where":
        return sql, False
    cond_start = q + 5
    while cond_start < len(sql) and sql[cond_start].isspace():
        cond_start += 1

    depth = 0
    r = p
    end_paren = -1
    while r < len(sql):
        if sql[r] == '(':
            depth += 1
        elif sql[r] == ')':
            depth -= 1
            if depth == 0:
                end_paren = r
                break
        r += 1
    if end_paren == -1:
        return sql, False
    cond_str = sql[cond_start:end_paren].strip()

    if func_name == "AVG":
        wrapped_expr = f"IFF({cond_str}, {expr_str}, NULL)"
    elif func_name == "SUM":
        wrapped_expr = f"IFF({cond_str}, {expr_str}, 0)"
    else:  # COUNT
        wrapped_expr = f"IFF({cond_str}, 1, NULL)"

    replacement = f"{func_name}({wrapped_expr})"
    new_sql = sql[:func_start] + replacement + sql[end_paren + 1:]
    return new_sql, True


def rewrite_postgres_filter_to_snowflake(sql: str) -> str:
    """
    Rewrites all occurrences of FUNC(expr) FILTER (WHERE cond) to IFF-based forms.
    """
    changed = True
    safety = 0
    while changed and safety < 20:
        sql, changed = _rewrite_filter_once(sql)
        safety += 1
    return sql


# ================================================================
# JSON coercion helper
# ================================================================

def coerce_jsonable(x, decimal_mode="string"):
    """
    Convert values to JSON-serializable types.
    - decimal_mode: 'string' (preserve precision) or 'float' (numeric ops).
    """
    if x is None:
        return None
    try:
        if isinstance(x, float) and (pd.isna(x) or x != x):  # NaN
            return None
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, Decimal):
        return str(x) if decimal_mode == "string" else float(x)
    if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date, datetime.time)):
        try:
            return x.isoformat()
        except Exception:
            return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(x)).decode("ascii")
    return x


# ================================================================
# Snowflake execution (with parametric attempts)
# ================================================================

def execute_snowflake_query(sql: str, conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [c[0] for c in (cur.description or [])]
        return pd.DataFrame(rows, columns=cols)


def execute_with_retry(sql: str, conn, attempts: int = 3, base_delay: float = 0.75, verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[Exception]]:
    """
    Execute a SQL query up to 'attempts' times with exponential backoff.
    Returns (DataFrame or None, last_exception or None).
    """
    attempt = 0
    last_exc: Optional[Exception] = None
    attempts = max(1, int(attempts or 1))  # ensure at least one try
    while attempt < attempts:
        try:
            df = execute_snowflake_query(sql, conn)
            return df, None
        except Exception as e:
            last_exc = e
            attempt += 1
            if verbose:
                print(f"Execution attempt {attempt} failed: {e}")
            if attempt < attempts:
                sleep_s = base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_s)
    return None, last_exc


# ================================================================
# LLM Invocation
# ================================================================

def invoke_llm_service(
    user_id: str,
    system_prompt: str,
    user_prompt: str,
    session_messages: Optional[List[Dict[str, str]]] = None,
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Calls GAIP LLM via gdk.invoke_llmgateway and returns:
    - raw response dict (gdk response)
    - primary message content (string)
    """
    model_id = model_id or GAIP_MODEL_ID
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    if session_messages:
        messages.extend(session_messages)
    if user_prompt and user_prompt.strip():
        messages.append({"role": "user", "content": user_prompt})

    prompt_template = {"prompt_template": messages}

    response = sdk.invoke_llmgateway(
        prompt_template,
        {"max_tokens": max_tokens, "temperature": temperature},
        model_id
    )

    try:
        content = response["genResponse"]["choices"][0]["message"]["content"]
    except Exception:
        content = str(response)

    return response, content


# ================================================================
# Result types
# ================================================================

@dataclass
class LLMUsage:
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AskResult:
    question: str
    sql_text: Optional[str]
    response_text: Optional[str]
    df: Optional[pd.DataFrame]
    usage: Optional[LLMUsage]
    error: str
    latency_sec: float


# ================================================================
# Error classification and feedback composition
# ================================================================

def is_compilation_or_semantic_error(exc: Exception) -> bool:
    """
    Heuristic: True for Snowflake SQL compilation/syntax/semantic errors that
    should trigger regeneration; False for likely transient issues.
    """
    s = str(exc).lower()
    if isinstance(exc, sf_errors.ProgrammingError):
        if "sql compilation error" in s or "syntax error" in s or "invalid identifier" in s or "unsupported subquery" in s:
            return True
    if "sql compilation error" in s or "syntax error" in s or "invalid identifier" in s:
        return True
    return False


def compose_regeneration_feedback(question: str, prior_sql: str, error_msg: str) -> str:
    """
    Build an instruction that includes the exact Snowflake error and prior SQL.
    """
    feedback = (
        "Original user question:\n"
        f"{question}\n\n"
        "The previous SQL failed in Snowflake with this error:\n"
        f"{error_msg}\n\n"
        "Previous SQL (fix this):\n"
        f"{prior_sql}\n\n"
        "Regenerate a correct Snowflake SQL query that satisfies the question.\n"
        "Important:\n"
        "- Use Snowflake syntax only.\n"
        "- Do not use PostgreSQL FILTER syntax; use conditional aggregation with IFF/CASE and NULLs.\n"
        "- Do not change requested grouping dimension or semantics.\n"
        "- If the selected table lacks required columns, choose a table that has them (e.g., use PAYMENT_METRICS for PAYMENT_TYPE_CD).\n"
        "- Output only a single SELECT/WITH statement, no comments or markdown.\n"
        "- Use fully qualified table names BUSINESS_CONTROL_TOWER.PAYMENT_CONTROL_TOWER.TABLE_NAME.\n"
    )
    return feedback


# ================================================================
# Ask / Execute wrapper with intelligent regeneration
# ================================================================

def ask_snowflake_question(
    question: str,
    conn,
    system_prompt_nl_to_sql: str = SYSTEM_PROMPT_NL_SQL,
    user_id: str = "nl2sql.com",
    session_messages: Optional[List[Dict[str, str]]] = None,
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.1,
    auto_rewrite_filter: bool = True,
    # Execution policy:
    attempts_initial_sql: int = 1,        # execute first SQL only once
    attempts_regenerated_sql: int = 3,    # allow up to 3 regenerated SQL attempts
    regen_max_attempts: int = 2,          # allow up to 2 regeneration cycles
    # regenerate on any error (not just compilation)
    regen_on_any_error: bool = True,
    error_feedback_role: str = "user",    # "user" or "system"
) -> AskResult:
    """
    Generates SQL, executes once, on failure, regenerates new SQL and executes up to N attempts
    for each regenerated SQL. Feedback with exact Snowflake error and prior SQL.
    """
    verbose = True  # Add this line since verbose is referenced but not defined as parameter
    if verbose:
        print(f"Q: {question}")

    t0 = time.time()
    last_usage: Optional[LLMUsage] = None
    last_resp_text: Optional[str] = None
    last_sql_text: Optional[str] = None
    last_exc: Optional[Exception] = None

    for attempt_idx in range(0, 1 + max(0, regen_max_attempts)):
        is_regen = attempt_idx > 0

        if is_regen:
            # Build prompts (with error feedback for regeneration) ...
            if not last_exc:
                effective_system = system_prompt_nl_to_sql
                effective_user = question
            else:
                feedback_block = compose_regeneration_feedback(
                    question,
                    last_sql_text or "",
                    str(last_exc) if last_exc else "Unknown error"
                )

                if error_feedback_role.lower() == "system":
                    effective_system = (
                        system_prompt_nl_to_sql.rstrip() + "\n\nPREVIOUS_ATTEMPT_FEEDBACK:\n" + feedback_block).strip()
                    effective_user = question
                else:
                    effective_system = system_prompt_nl_to_sql
                    effective_user = feedback_block

        if verbose:
            print(
                f"Regeneration attempt {attempt_idx} with error feedback ({error_feedback_role}): {last_exc}")
        else:
            effective_system = system_prompt_nl_to_sql
            effective_user = question

        if verbose:
            print(
                f"Regeneration attempt {attempt_idx} with error feedback ({error_feedback_role}): {last_exc}")

        gdk_response, response_text = invoke_llm_service(
            user_id=user_id,
            system_prompt=effective_system,
            user_prompt=effective_user,
            session_messages=session_messages,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        last_resp_text = response_text

        # Usage
        try:
            u = gdk_response["genResponse"]["usage"]
            last_usage = LLMUsage(
                completion_tokens=int(u.get("completion_tokens", 0)),
                prompt_tokens=int(u.get("prompt_tokens", 0)),
                total_tokens=int(u.get("total_tokens", 0)),
            )
        except Exception:
            pass

        # ===== Normalize / extract / rewrite / guard =====
        sql_text = extract_first_sql_statement(response_text)
        if auto_rewrite_filter:
            rewritten = rewrite_postgres_filter_to_snowflake(sql_text)
            if rewritten != sql_text and verbose:
                print(
                    f"Rewritten: Converted FILTER(WHERE ...) to Snowflake conditional aggregation."
                )
            sql_text = rewritten

        if verbose:
            print(f"Generated SQL (attempt {attempt_idx}): {sql_text}")

        if not is_select_only_strict(sql_text):
            if verbose:
                print(
                    "Guard: Statement is not strictly SELECT/WITH-only (or contains forbidden verbs). Aborting."
                )
            return AskResult(
                question=question,
                sql_text=sql_text,
                response_text=last_resp_text,
                df=None,
                usage=last_usage,
                error="blocked_by_guard",
                latency_sec=round(time.time() - t0, 1),
            )

        # ===== Execute with attempts policy =====
        attempts_for_this_sql = attempts_regenerated_sql if is_regen else attempts_initial_sql

        df, exec_exc = execute_with_retry(
            sql_text, conn, attempts=attempts_for_this_sql, base_delay=0.75, verbose=verbose
        )

        last_sql_text = sql_text
        last_exc = exec_exc

        if df is not None:
            if verbose:
                print(f"Rows returned: {len(df)}")
            t1 = time.time()
            return AskResult(
                question=question,
                sql_text=sql_text,
                response_text=last_resp_text,
                df=df,
                usage=last_usage,
                error="",
                latency_sec=round(t1 - t0, 3),
            )

        # ----- Decide whether to regenerate or stop -----
        if attempt_idx >= regen_max_attempts:
            # No more regenerations allowed
            break

        if regen_on_any_error:
            # Always regenerate on any failure (as per your policy)
            continue
        else:
            # Only regenerate on compilation/syntax/semantic errors
            if is_compilation_or_semantic_error(exec_exc):
                continue
            else:
                # Transient but exhausted attempts for this SQL - stop here
                break

    # All attempts exhausted
    t1 = time.time()
    if verbose and last_exc:
        print(f"Final execution error after regeneration attempts: {last_exc}")
    return AskResult(
        question=question,
        sql_text=last_sql_text,
        response_text=last_resp_text,
        df=None,
        usage=last_usage,
        error="blocked_or_error",
        latency_sec=round(t1 - t0, 3),
    )
# ================================================================
# Example questions
# ================================================================


questions = [
    "What is the total number of payments processed in the last 30 days?",
    "List all payments with fraud hits from yesterday",
    "List all payments with fraud hits from yesterday",
    "What is the total payment volume in payment system?",
    "How many straight-through-processed (STP) payments occurred today?",
    "What is the daily success rate of payments by payment type?",
    "What is the average processing time for payments that hit AML checks?",
    "What is the recent over-time change in payment volume by payment type?",
    "Which payment types have the highest STP rates over the last 30 days compared to their 180-day average?",
    "What is the distribution of payment processing times across different payment systems?",
    "What are the monthly fraud hit rates by payment type?",
    "Which payment types show unusual spikes in fraud hits compared to their historical patterns?",
    "Which payment types have not had any successful AML or fraud checks in the last month despite having payment volume?",
    "What are the peak transaction volumes during business hours and which hour has the highest volume?",
    "What is the trend of total cycle times by source system?",
]

# ================================================================
# Main loop (returns log of with result_json)
# ================================================================


def run_eval(
    questions,
    system_prompt_nl_to_sql: str = SYSTEM_PROMPT_NL_SQL,
    model_id: str = GAIP_MODEL_ID,
    max_result_rows: int = 100,
    save_csv_path: str = "py_file_snowflake_eval_logs_1.csv",
    user_id: str = "nl2sql.com",
    temperature: float = 0.1,
    max_tokens: int = 1500,
    verbose: bool = True,
) -> pd.DataFrame:
    conn = get_snowflake_connection()

    log_question: List[str] = []
    log_response: List[str] = []
    log_result_json: List[str] = []
    log_rowcount: List[int] = []
    log_latency: List[float] = []
    log_completion_tokens: List[int] = []
    log_prompt_tokens: List[int] = []
    log_total_tokens: List[int] = []
    log_sql: List[str] = []
    log_error: List[str] = []

    for user_query in tqdm(questions):
        result = ask_snowflake_question(
            question=user_query,
            conn=conn,
            system_prompt_nl_to_sql=system_prompt_nl_to_sql,
            user_id=user_id,
            model_id=model_id,
            session_messages=None,
            max_tokens=max_tokens,
            temperature=temperature,
            auto_rewrite_filter=True,
            # New execution policy:
            attempts_initial_sql=1,        # execute first SQL once
            attempts_after_regen=3,        # regenerated SQL gets up to 3 attempts
            regen_max_attempts=2,          # try up to 2 regenerations
            regen_on_any_error=True,
            error_feedback_role="user",    # or "system"
            verbose=verbose,
        )

        if result.df is not None:
            df_for_json = result.df.head(max_result_rows).copy()
            records = df_for_json.to_dict(orient="records")
            safe_records = []
            for rec in records:
                safe_rec = {}
                for k, v in rec.items():
                    safe_rec[k] = coerce_jsonable(v, decimal_mode="string")
                safe_records.append(safe_rec)

            result_json = json.dumps(safe_records, ensure_ascii=False)
            rowcount = len(result.df)
        else:
            result_json = "[]"
            rowcount = 0

        completion_tokens = result.usage.completion_tokens if result.usage else 0
        prompt_tokens = result.usage.prompt_tokens if result.usage else 0
        total_tokens = result.usage.total_tokens if result.usage else 0

        log_question.append(result.question)
        log_response.append(result.response_text or "")
        log_result_json.append(result_json)
        log_rowcount.append(rowcount)
        log_latency.append(result.latency_sec)
        log_completion_tokens.append(completion_tokens)
        log_prompt_tokens.append(prompt_tokens)
        log_total_tokens.append(total_tokens)
        log_sql.append(result.sql_text or "")
        log_error.append(result.error)

    df_log = pd.DataFrame({
        "question": log_question,
        "sql_response": log_response,
        "cleaned_sql": log_sql,
        "result_json": log_result_json,
        "row_count": log_rowcount,
        "latency_sec": log_latency,
        "completion_tokens": log_completion_tokens,
        "prompt_tokens": log_prompt_tokens,
        "total_tokens": log_total_tokens,
        "error": log_error,
    })

    if save_csv_path:
        df_log.to_csv(save_csv_path, index=False, encoding="utf-8")
        if verbose:
            print(f"\nCSV saved to {save_csv_path}")

    try:
        conn.close()
    except Exception:
        pass

    return df_log


# ================================================================
# Main execution
# ================================================================

if __name__ == "__main__":
    result_df = run_eval(
        questions,
        system_prompt_nl_to_sql=SYSTEM_PROMPT_NL_SQL,
        max_result_rows=100
    )

    print(f"\nEvaluation completed!")
    print(f"Total questions processed: {len(result_df)}")
    print(f"Successful queries: {len(result_df[result_df['error'] == ''])}")
    print(f"Failed queries: {len(result_df[result_df['error'] != ''])}")

    if len(result_df) > 0:
        avg_latency = result_df['latency_sec'].mean()
        avg_tokens = result_df['total_tokens'].mean()
        print(f"Average latency: {avg_latency:.2f} seconds")
        print(f"Average total tokens: {avg_tokens:.0f}")

        # Show sample results
        print("\nSample successful results:")
        successful = result_df[result_df['error'] == ''].head(3)
        for _, row in successful.iterrows():
            print(f"Q: {row['question'][:80]}...")
            print(f"Rows: {row['row_count']}, Latency: {row['latency_sec']}s")
            print("---")
