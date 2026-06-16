"""Module 3: GigaChat-based semantic judge for candidate pairs."""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat

from candidates import CandidatePair

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
load_dotenv(override=True)


# ── GigaChat initialization (exact pattern from fd.py) ────────────────────────

def get_llm() -> GigaChat:
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise RuntimeError("GIGACHAT_CREDENTIALS не найден. Проверьте .env.")
    return GigaChat(
        credentials=credentials.strip("'\""),
        base_url=os.getenv("GIGACHAT_BASE_URL", "").strip("'\"") or None,
        auth_url=os.getenv("GIGACHAT_AUTH_URL", "").strip("'\"") or None,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip("'\""),
        verify_ssl_certs=False,
        profanity_check=False,
        model="GigaChat-2-Pro",
        timeout=45,
        temperature=0.1,
    )


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    left_file: str
    left_row_id: int
    right_file: str
    right_row_id: int
    label: str          # "match" | "non_match" | "uncertain"
    confidence: float   # 0.0 – 1.0
    reason: str
    intra_supplier: bool = False


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Ты эксперт по сопоставлению товаров. Определи, описывают ли два товара \
из разных прайс-листов один и тот же реальный товар.

Отвечай строго в формате JSON без markdown-обёртки:
{{"label": "match" | "non_match" | "uncertain", "confidence": 0.0-1.0, "reason": "..."}}

Правила:
- "match": один и тот же товар (допустимы различия в языке, сокращениях, маркетинговых словах, порядке слов)
- "non_match": разные товары (разные характеристики, модели, категории, объём памяти, цвет, диагональ)
- "uncertain": недостаточно информации для уверенного решения (неполные названия, только бренд совпадает)

Confidence: 0.9-1.0 = очень уверен, 0.7-0.9 = уверен, 0.5-0.7 = сомневаюсь, <0.5 = очень неуверен.\
"""

HUMAN_TEMPLATE = """\
Товар 1 (поставщик: {left_file}):
  Название: {left_name}
  Цена: {left_price:.0f} руб.

Товар 2 (поставщик: {right_file}):
  Название: {right_name}
  Цена: {right_price:.0f} руб.

Разница в цене: {price_diff_pct:.1f}%
Совпадение токенов: {token_jaccard:.2f}\
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response, handling markdown code blocks."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in response: {text!r}")


def _parse_result(raw: str, pair: CandidatePair) -> JudgeResult:
    """Parse GigaChat JSON response into JudgeResult."""
    data = _extract_json(raw)
    label = str(data.get("label", "uncertain")).strip().lower()
    if label not in ("match", "non_match", "uncertain"):
        label = "uncertain"
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", "")).strip()
    return JudgeResult(
        left_file=pair.left_file,
        left_row_id=pair.left_row_id,
        right_file=pair.right_file,
        right_row_id=pair.right_row_id,
        label=label,
        confidence=confidence,
        reason=reason,
        intra_supplier=pair.intra_supplier,
    )


# ── Judge function ─────────────────────────────────────────────────────────────

def judge_pair(pair: CandidatePair, llm: GigaChat) -> JudgeResult:
    """Call GigaChat to judge a single candidate pair.

    Retries once with a stricter prompt on JSON parse failure.
    Falls back to uncertain/0.0 on second failure.
    """
    messages = _prompt.format_messages(
        left_file=pair.left_file,
        left_name=pair.left_name,
        left_price=pair.left_price,
        right_file=pair.right_file,
        right_name=pair.right_name,
        right_price=pair.right_price,
        price_diff_pct=pair.price_diff_pct,
        token_jaccard=pair.token_jaccard,
    )

    for attempt in range(2):
        try:
            response = llm.invoke(messages)
            return _parse_result(response.content, pair)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if attempt == 0:
                # Retry with explicit reminder
                messages[-1].content += (
                    "\n\nВАЖНО: отвечай ТОЛЬКО JSON объектом, без пояснений."
                )
            else:
                return JudgeResult(
                    left_file=pair.left_file,
                    left_row_id=pair.left_row_id,
                    right_file=pair.right_file,
                    right_row_id=pair.right_row_id,
                    label="uncertain",
                    confidence=0.0,
                    reason=f"Ошибка парсинга ответа: {e}",
                    intra_supplier=pair.intra_supplier,
                )

    # Should never reach here
    return JudgeResult(
        left_file=pair.left_file,
        left_row_id=pair.left_row_id,
        right_file=pair.right_file,
        right_row_id=pair.right_row_id,
        label="uncertain",
        confidence=0.0,
        reason="Неизвестная ошибка",
        intra_supplier=pair.intra_supplier,
    )


def judge_pairs_batch(
    pairs: list[CandidatePair],
    llm: GigaChat | None = None,
    verbose: bool = True,
) -> list[JudgeResult]:
    """Judge a list of candidate pairs, returning JudgeResult for each."""
    if llm is None:
        llm = get_llm()

    results: list[JudgeResult] = []
    total = len(pairs)
    for i, pair in enumerate(pairs):
        result = judge_pair(pair, llm)
        results.append(result)
        if verbose:
            icon = {"match": "✓", "non_match": "✗", "uncertain": "?"}.get(result.label, "?")
            print(
                f"  [{i+1}/{total}] {icon} {result.label} "
                f"(conf={result.confidence:.2f}) "
                f"{pair.left_file.split('_')[1]}:{pair.left_row_id} × "
                f"{pair.right_file.split('_')[1]}:{pair.right_row_id}"
            )
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from load_data import load_all_suppliers
    from candidates import generate_candidates

    print("Loading data...")
    data = load_all_suppliers()
    print("Generating candidates...")
    cands = generate_candidates(data)
    print(f"Judging first 3 candidates...")
    llm = get_llm()
    results = judge_pairs_batch(cands[:3], llm=llm)
    for r in results:
        print(f"\n  {r.label} ({r.confidence:.2f}): {r.reason}")
