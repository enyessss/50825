# =========================
# 1) 표준/외부 임포트
# =========================
from fastapi import FastAPI, Query, Body, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import asyncio, os, json, re, traceback, csv, io
import httpx  # OpenAI/실시간 API 호출용

# =========================
# 2) 안전 임포트(있으면 쓰고, 없어도 서버 구동)
# =========================
def _safe_import(path: str, names: List[str]):
    mod = None
    try:
        mod = __import__(path, fromlist=names)
    except Exception:
        return {n: None for n in names}
    out = {}
    for n in names:
        out[n] = getattr(mod, n, None)
    return out

# settings (OpenAI/서울API/임베딩 등)
_settings = _safe_import("settings", ["settings"])
settings = _settings["settings"]  # 변수(고정): settings

# --- Feature toggles ---
USE_RAG_ELEV = True        # 변수(고정): 엘리베이터는 RAG 사용
USE_RAG_RESTROOM = False   # 변수(고정): 화장실은 CSV만 (RAG 미사용)

# 서울 Open API (실시간 포함 가능)
_seoul = _safe_import(
    "seoul_api",
    ["fetch_total_count", "fetch_page", "fetch_realtime_status", "fetch_realtime_status_by_station"]
)
fetch_total_count = _seoul["fetch_total_count"]
fetch_page = _seoul["fetch_page"]
fetch_realtime_status = _seoul["fetch_realtime_status"]
fetch_realtime_status_by_station = _seoul["fetch_realtime_status_by_station"]

# 정규화
_normalize = _safe_import("normalize", ["normalize_row"])
normalize_row = _normalize["normalize_row"]

# UID
_uid = _safe_import("uid", ["make_uid"])
make_uid = _uid["make_uid"]

# 랭킹
_ranker = _safe_import("ranker", ["rank_items"])
rank_items = _ranker["rank_items"] or (lambda metas, *a, **k: metas)  # 변수(고정): 폴백

# RAG(있을 때만)
_rag = _safe_import(
    "rag_index",
    ["get_collection", "embed_texts", "upsert_docs", "build_doc_text", "collection_count"]
)
get_collection   = _rag["get_collection"]
embed_texts      = _rag["embed_texts"]
upsert_docs      = _rag["upsert_docs"]
build_doc_text   = _rag["build_doc_text"] or (lambda m: json.dumps(m, ensure_ascii=False))
collection_count = _rag["collection_count"] or (lambda: 0)

# =========================
# 3) 앱 메타
# =========================
app = FastAPI(
    title="Seoul Elevator Backend",
    version="1.4.0",
    description="서울 지하철 엘리베이터/휠체어리프트 안내 (LLM 파싱 + 실시간 OpenAPI + RAG + CSV)"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],          # 개발용. 배포 시 도메인 지정 권장
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# =========================
# 4) 전역 상수/메모리 캐시
# =========================
DB_PATH: str = "./data/elevators.json"     # 변수(고정): 엘리베이터 로컬 저장 경로
ELEV_DB: Dict[str, dict] = {}              # 변수(고정): 엘리베이터 메모리 캐시 (uid -> meta)

REST_CSV_PATH: str = "./data1/restrooms.csv"   # 변수(고정): 장애인 화장실 CSV 원본
REST_JSON_PATH: str = "./data1/restrooms.json" # 변수(고정): 장애인 화장실 로컬 캐시(JSON)
REST_DB: List[Dict[str, Any]] = []             # 변수(고정): 장애인 화장실 메모리 캐시 (list of dict)

# =========================
# 5) 공통 유틸
# =========================
def _station_key(s: Optional[str]) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\([^)]*\)", "", s)  # 괄호 제거: '홍대입구(2)' -> '홍대입구'
    s = re.sub(r"역$", "", s)        # 끝의 '역' 제거
    return s

def _station_eq(a: Optional[str], b: Optional[str]) -> bool:
    return _station_key(a) == _station_key(b)

def _station_matches(meta_station: Optional[str], query: Optional[str]) -> bool:
    if not meta_station or not query:
        return False
    return _station_eq(meta_station, query)

def _display_station(st: Optional[str]) -> str:  # 변수(고정)
    s = (st or "").strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"역$", "", s)
    return s + "역" if s else ""

def _infer_station_from_question(q: str) -> Optional[str]:
    qnorm = re.sub(r"\s+", "", q or "")
    names = sorted({m.get("station_name") for m in ELEV_DB.values() if m.get("station_name")}, key=lambda x: len(x or ""), reverse=True)
    for name in names:
        if not name:
            continue
        if re.search(re.escape(name) + r"(역)?", qnorm):
            return name
    return None

# --- [main.py] 테이블 포맷 유틸 (NEW) --------------------------------------

# 변수(고정): 표 컬럼 정의
RESTROOM_COLUMNS: list[str] = ["호선", "층", "내/외부", "출구", "위치", "운영시간", "전화"]  # 변수(고정)
ELEVATOR_COLUMNS: list[str] = ["호선", "층", "연결", "방면/상세", "상태"]                  # 변수(고정)

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows) if rows else "| - |"
    return "\n".join([head, sep, body])

def _restroom_rows(items: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for it in items:
        line  = it.get("line_name") or (it.get("line_no") and f"{it['line_no']}호선") or ""
        floor = ((it.get("ground") or "") + (it.get("floor") or "")).strip() or (it.get("floor") or "")
        gate  = (it.get("gate") or "").replace(" ", "")
        exitc = str(it.get("exit_no") or it.get("exit_code") or "")
        loc   = it.get("detail") or ""
        hrs   = it.get("open_hours") or ""
        phone = it.get("phone") or ""
        rows.append([line, floor, gate, exitc, loc, hrs, phone])
    return rows

def _elevator_rows(items: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for it in items:
        line  = it.get("line_name") or (it.get("line_no") and f"{it['line_no']}호선") or ""
        floor = ""
        lp = it.get("levels_path") or []
        if len(lp) >= 1:
            # (표시는 B2↔B1은 'B2↔B1', 층 칸엔 최상단만 간단히)
            floor = (lp[0] if isinstance(lp[0], str) else str(lp[0]))
        connect = ""
        if len(lp) >= 2:
            a = lp[0] if isinstance(lp[0], str) else str(lp[0])
            b = lp[-1] if isinstance(lp[-1], str) else str(lp[-1])
            connect = f"{a}↔{b}"
        detail = it.get("instl_pstn") or it.get("detail") or it.get("direction_hint") or ""
        status = it.get("use_status_raw") or it.get("status") or ""
        rows.append([line, floor, connect, detail, status])
    return rows

def format_facilities_as_tables(restrooms: list[dict], elevators: list[dict]) -> str:
    parts: list[str] = []
    if restrooms:
        parts.append("### 🚻 장애인 화장실\n" + _md_table(RESTROOM_COLUMNS, _restroom_rows(restrooms)))
    if elevators:
        parts.append("### 🛗 엘리베이터\n" + _md_table(ELEVATOR_COLUMNS, _elevator_rows(elevators)))
    return "\n\n".join(parts) if parts else "표로 표시할 설비를 찾지 못했어요."
# --------------------------------------------------------------------------
# --- [main.py] 요약(LLM/폴백) 유틸 (NEW) ------------------------------------

# 변수(고정): 요약 최대 줄 수 / 최대 항목
SUMMARY_MAX_LINES: int = 3       # 변수(고정)
SUMMARY_SAMPLE_LIMIT: int = 6    # 변수(고정)

def _short_line_for_restroom(m: dict) -> str:
    ln  = m.get("line_name") or (m.get("line_no") and f"{m['line_no']}호선") or ""
    fl  = ((m.get("ground") or "") + (m.get("floor") or "")).strip() or (m.get("floor") or "")
    gate= (m.get("gate") or "").replace(" ", "")
    ex  = m.get("exit_no") or m.get("exit_code") or ""
    det = m.get("detail") or ""
    return " · ".join([x for x in [ln, fl, gate, (f"{ex}번 출구" if ex else ""), det] if x])

def _short_line_for_elev(m: dict) -> str:
    ln  = m.get("line_name") or (m.get("line_no") and f"{m['line_no']}호선") or ""
    lp  = m.get("levels_path") or []
    seg = (f"{lp[0]}↔{lp[-1]}" if len(lp) >= 2 else (lp[0] if lp else "")) or ""
    pos = m.get("instl_pstn") or m.get("detail") or m.get("direction_hint") or ""
    sta = m.get("use_status_raw") or ""
    return " · ".join([x for x in [ln, seg, pos, (f"상태 {sta}" if sta else "")] if x])

def _summarize_rule(restrooms: list[dict], elevators: list[dict],
                    station_hint: str = "", line_no: str | None = None) -> str:
    head = (station_hint or "").strip()
    if line_no:
        head = (head + f" {line_no}호선").strip()

    lines: list[str] = []
    if restrooms:
        # 출구번호, 층 분포 간단 요약
        exits = [str(m.get("exit_no") or m.get("exit_code") or "") for m in restrooms if (m.get("exit_no") or m.get("exit_code"))]
        exits = [e for e in exits if e]
        floors = [((m.get("ground") or "") + (m.get("floor") or "")).strip() or (m.get("floor") or "") for m in restrooms]
        floors = [f for f in floors if f]
        ex_preview = (", ".join(sorted(set(exits), key=lambda x: (len(x), x))[:5]) + (" 외" if len(set(exits))>5 else "")) if exits else "출구 표기 없음"
        fl_preview = (", ".join(sorted(set(floors), key=lambda x: (x.startswith('B'), x))) ) if floors else "층 정보 없음"
        lines.append(f"장애인 화장실 {len(restrooms)}개 · 출구 {ex_preview} · 층 {fl_preview}")

    if elevators:
        stats = [m.get("use_status_raw") or "" for m in elevators]
        buck = _status_bucket(stats)
        # 연결 경로 샘플
        segs = []
        for m in elevators[:SUMMARY_SAMPLE_LIMIT]:
            lp = m.get("levels_path") or []
            seg = (f"{lp[0]}↔{lp[-1]}" if len(lp) >= 2 else (lp[0] if lp else ""))
            if seg: segs.append(seg)
        segs = [*dict.fromkeys(segs)]  # dedup keep order
        seg_preview = ", ".join(segs[:3]) if segs else "연결 정보 없음"
        lines.append(f"엘리베이터 {len(elevators)}개 · 연결 {seg_preview} · 상태 요약(정상 {buck['정상']}, 중지/고장 {buck['중지/고장']}, 정보없음 {buck['정보없음']})")

    if not lines:
        return "요약할 데이터가 없습니다."
    out = (head + " 요약: ").strip() if head else "요약: "
    return out + " / ".join(lines[:SUMMARY_MAX_LINES])

async def _summarize(restrooms: list[dict], elevators: list[dict],
                     station_hint: str = "", line_no: str | None = None,
                     mode: str = "rule") -> str:
    """
    mode: "rule"(기본, 결정론) | "llm"
    - "llm"은 ALLOW_LLM_SUMMARY=True 인 경우에만 시도, 실패 시 rule로 폴백
    """
    # 항상 먼저 rule 생성
    rule_text = _summarize_rule(restrooms, elevators, station_hint, line_no)

    if mode != "llm" or not ALLOW_LLM_SUMMARY or not OPENAI_API_KEY:
        return rule_text

    # 여기부터는 명시적 허용 + 키 있을 때만 LLM 시도 (엄격 프롬프트 + 온도 0)
    bullets = []
    for m in restrooms[:SUMMARY_SAMPLE_LIMIT]:
        bullets.append("- REST " + _short_line_for_restroom(m))
    for m in elevators[:SUMMARY_SAMPLE_LIMIT]:
        bullets.append("- ELEV " + _short_line_for_elev(m))

    sys_p = (
        "주어진 항목만 근거로 2~3문장 한국어 요약을 작성하라. "
        "새 정보 추정/추가는 금지. 숫자/출구/층/연결/상태만 사용. "
        "데이터에 없는 값 언급 금지. 간결한 명사형 문장 선호."
    )
    usr_p = (f"역: {(station_hint + (' ' + str(line_no) + '호선' if line_no else '')).strip() or '미지정'}\n"
             "항목:\n" + "\n".join(bullets if bullets else ["(없음)"]))

    try:
        txt = await _openai_chat(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            temperature=0.0
        )
        llm_text = (txt or "").strip()
        # 가드: LLM 결과가 비었거나 금칙(데이터 외 단어 길게) 감지되면 rule로 대체
        return llm_text if llm_text else rule_text
    except Exception:
        return rule_text
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# =========================
# 6) CSV: 장애인 화장실 로더(정리본) + 포맷터
# =========================
def _parse_exit_code(v):
    s = str(v).strip() if v is not None else ""
    if not s:
        return None, None, None
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"번$", "", s)
    m = re.match(r"^(\d+)(?:-(\d+))?$", s)
    if m:
        main = int(m.group(1))
        sub = int(m.group(2)) if m.group(2) else None
        code = f"{main}-{sub}" if sub is not None else str(main)
        return main, sub, code
    digits = re.sub(r"[^0-9]", "", s)
    if digits:
        return int(digits), None, digits
    return None, None, None

def _normalize_floor(s: Optional[str]) -> str:
    t = (s or "").strip()
    if not t: return ""
    if re.match(r"^B\d+$", t, re.I): return t.upper()
    if re.match(r"^\d+F$", t, re.I): return t.upper()
    m = re.search(r"지하\s*(\d+)\s*층", t)
    if m: return f"B{m.group(1)}"
    m = re.search(r"지상\s*(\d+)\s*층", t)
    if m: return f"{m.group(1)}F"
    m = re.search(r"지하\s*(\d+)", t)
    if m: return f"B{m.group(1)}"
    m = re.search(r"(\d+)\s*층", t)
    if m: return f"{m.group(1)}F"
    return t

def _normalize_gate(s: Optional[str]) -> str:
    t = (s or "").strip()
    if not t: return ""
    if "내" in t: return "내부"
    if "외" in t: return "외부"
    return t

def _extract_line_no(s: Optional[str]) -> str:
    t = (s or "").strip()
    m = re.search(r"(\d+)", t)
    return m.group(1) if m else t

def _format_restroom_line(m: Dict[str, Any]) -> str:
    line_txt = (
        f"{m.get('line_no')}호선" if re.match(r"^\d+$", str(m.get('line_no') or ""))
        else str(m.get('line_name') or "")
    )
    st       = _display_station(m.get("station_name"))
    ground   = (m.get("ground") or "").strip()
    floor    = (m.get("floor") or "").strip()
    gate     = (m.get("gate") or "").strip()
    detail   = (m.get("detail") or "").strip()
    exit_code= (m.get("exit_code") or "").strip() if m.get("exit_code") else ""
    phone    = (m.get("phone") or "").strip()
    hours    = (m.get("open_hours") or "").strip()

    where = []
    if ground: where.append(f"\"{ground}\"")
    if floor:  where.append(f"\"{floor}\"")
    if gate:   where.append(f"게이트 \"{gate}\"")
    if detail: where.append(f"\"{detail}\"")
    if exit_code: where.append(f"\"{exit_code}번 출입구\" 주변")

    head = f"{line_txt} {st} 장애인 화장실은 " if line_txt else f"{st} 장애인 화장실은 "
    extra = []
    if hours: extra.append(f"개방시간: {hours}")
    if phone: extra.append(f"전화번호: {phone}")

    return head + ((" ".join(where)) if where else "상세 위치 정보가 없습니다.") + "." + \
           (("\n" + " ".join(extra)) if extra else "")

def _load_restrooms_from_csv(path: str) -> int:
    """CSV를 읽어 REST_DB(list[dict])에 로드 (BOM 제거 + 구분자 자동감지 + 헤더 정규화)"""
    REST_DB.clear()
    if not os.path.exists(path):
        print(f"[restrooms] CSV not found: {path}")
        return 0

    with open(path, "rb") as fb:
        raw = fb.read()
    text = raw.decode("utf-8-sig", errors="replace")

    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=[",", "\t", ";"])
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        print("[restrooms] 헤더 없음")
        return 0

    def norm_key(h: str) -> str:
        h = (h or "").replace("\ufeff", "")
        h = re.sub(r"\s+", " ", h.strip())
        h = h.replace("(근접)", "근접")
        h = h.replace("출입구  번호", "출입구 번호")
        if h in ("구분", "구분(장애인화장실)", "구분 (장애인 화장실)"):
            return "구분(장애인 화장실)"
        if ("근접" in h) and ("출입구" in h) and ("번호" in h):
            return "근접 출입구 번호"
        if h in ("상세위치", "상세 위치"):
            return "상세위치"
        return h

    headers_norm = [norm_key(h) for h in reader.fieldnames]
    raw_by_norm = {norm_key(h): h for h in reader.fieldnames}

    required = ["운영노선명","역명","지상 또는 지하 구분","역층","게이트 내외 구분","근접 출입구 번호","상세위치","전화번호","개방시간"]
    missing = [h for h in required if h not in headers_norm]
    if missing:
        print("[restrooms] CSV 헤더가 맞지 않습니다. 누락:", ", ".join(missing))
        return 0

    cnt = 0
    for row in reader:
        st = (row.get(raw_by_norm["역명"]) or "").strip()
        if not st:
            continue

        line_name = (row.get(raw_by_norm["운영노선명"]) or "").strip()
        line_no = _extract_line_no(line_name)

        ground = (row.get(raw_by_norm["지상 또는 지하 구분"]) or "").strip()
        floor  = _normalize_floor(row.get(raw_by_norm["역층"]))
        gate   = _normalize_gate(row.get(raw_by_norm["게이트 내외 구분"]))
        exit_raw = (row.get(raw_by_norm["근접 출입구 번호"]) or "").strip()
        exit_main, exit_sub, exit_code = _parse_exit_code(exit_raw)
        detail = (row.get(raw_by_norm["상세위치"]) or "").strip()
        phone  = (row.get(raw_by_norm["전화번호"]) or "").strip()
        hours  = (row.get(raw_by_norm["개방시간"]) or "").strip()

        REST_DB.append({
            "station_name": st,
            "line_name": line_name,   # ex) '2호선'
            "line_no": line_no,       # ex) '2'
            "ground": ground,         # '지상'/'지하'
            "floor": floor,           # 'B1','1F' 등
            "gate": gate,             # '내부'/'외부'
            "exit_no": exit_main,     # int|None
            "exit_sub": exit_sub,     # int|None
            "exit_code": exit_code or (str(exit_main) if exit_main is not None else None),
            "detail": detail,
            "phone": phone,
            "open_hours": hours,
        })
        cnt += 1

    print(f"[restrooms] loaded {cnt} rows from {path}")
    return cnt

def _save_rest_db():
    os.makedirs(os.path.dirname(REST_JSON_PATH), exist_ok=True)
    with open(REST_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(REST_DB, f, ensure_ascii=False)

def _load_rest_db():
    if os.path.exists(REST_JSON_PATH):
        try:
            with open(REST_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    REST_DB[:] = data
        except Exception:
            pass

# =========================
# 7) 파일 I/O (엘리베이터 DB)
# =========================
def _ensure_dirs():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _load_db():
    _ensure_dirs()
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                ELEV_DB.update(json.load(f))
        except Exception:
            pass  # 손상/빈 파일이어도 계속 실행

def _save_db():
    _ensure_dirs()
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(ELEV_DB, f, ensure_ascii=False)

# =========================
# 8) 메타 직렬화/도우미
# =========================
def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}  # 변수
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif isinstance(v, list):
            safe[k] = ",".join(map(str, v))
        elif isinstance(v, dict):
            safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = str(v)
    return safe

# =========================
# 9) 규칙 기반 파싱(엘리베이터 보조)
# =========================
def _parse_constraints_from_question(q: str) -> Tuple[List[str], Optional[int], Optional[str], Optional[str], Optional[str], bool]:
    q = (q or "").strip()
    targets = []
    if re.search(r"(1\s*층|지상)", q): targets.append("1F")
    if re.search(r"대합실", q): targets.append("B1")

    exit_no = None
    m = re.search(r"(?<!\d)(\d{1,2})\s*번\s*출\s*구", q)
    if m: exit_no = int(m.group(1))

    direction = None
    m = re.search(r"([가-힣A-Za-z0-9]+)\s*방면", q)
    if m: direction = m.group(1)

    car_hint = None
    m = re.search(r"\b(\d-\d)\b", q)
    if m: car_hint = m.group(1)

    want_type = None
    if re.search(r"(휠체어\s*리프트|리프트)", q):
        want_type = "휠체어 리프트"
    elif re.search(r"(엘리베이터|엘베)", q):
        want_type = "엘리베이터"

    prefer_internal = False  # 변수
    if (
        re.search(r"(환승|갈아타|환승통로)", q)
        or re.search(r"(방면|행)", q)
        or re.search(r"(플랫폼|승강장|내부)", q)
        or re.search(r"[→>\-]\s*", q)
    ) and ("1F" not in [t.upper() for t in targets]):
        prefer_internal = True

    return targets, exit_no, direction, car_hint, want_type, prefer_internal

# =========================
# 10) OpenAI Chat 래퍼 + LLM 파서
# =========================
OPENAI_API_KEY: str = getattr(settings, "OPENAI_API_KEY", "")                         # 변수(고정)
OPENAI_BASE_URL: str = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")  # 변수(고정)
GENERATION_MODEL: str = getattr(settings, "GENERATION_MODEL", "gpt-3.5-turbo")            # 변수(고정)
HTTP_TIMEOUT_SEC: int = getattr(settings, "HTTP_TIMEOUT_SEC", 8)
# 변수(고정): LLM 요약 허용 여부 (기본 False → 헛소리 방지)
ALLOW_LLM_SUMMARY: bool = False
# 변수(고정)

async def _openai_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not OPENAI_API_KEY:
        return ""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GENERATION_MODEL, "messages": messages, "temperature": temperature}
    url = f"{OPENAI_BASE_URL}/chat/completions"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

async def _llm_parse_question(question: str) -> Dict[str, Any]:
    sys_p = (
        "너는 한국 지하철 안내용 파서야. 사용자의 문장을 읽고 JSON으로만 답해.\n"
        "키: station_hint(문자|null), line_no(문자|null), exit_no(정수|null), "
        "targets(예:['1F','B1'] 없으면 []), want_type('엘리베이터'|'휠체어 리프트'|null), "
        "direction(문자|null), car_hint(문자|null), prefer_internal(불리언)\n"
        "층 표기는 ['B3','B2','B1','1F','2F',...]로 표준화. 설명 금지."
    )
    usr_p = f"문장: {question}"
    try:
        resp = await _openai_chat(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            temperature=0.0
        )
        data = json.loads(resp)
    except Exception:
        data = {}

    data.setdefault("station_hint", None)
    data.setdefault("line_no", None)
    data.setdefault("exit_no", None)
    data.setdefault("targets", [])
    data.setdefault("want_type", None)
    data.setdefault("direction", None)
    data.setdefault("car_hint", None)
    data.setdefault("prefer_internal", False)

    if data["exit_no"] is not None:
        try: data["exit_no"] = int(data["exit_no"])
        except Exception: data["exit_no"] = None

    data["targets"] = [str(t).upper() for t in (data["targets"] or [])]
    if data["want_type"] not in ("엘리베이터", "휠체어 리프트", None):
        data["want_type"] = None

    q = question or ""
    if re.search(r"(대합실|환승통로|환승)", q) and "1F" not in data["targets"]:
        data["prefer_internal"] = True

    return data

# =========================
# 11) 포맷/중복제거/상태문구
# =========================
def _format_levels(meta: Dict[str, Any]) -> str:  # 변수(고정)
    lp = meta.get("levels_path") or []
    return f"({lp[0]}↔{lp[-1]})" if len(lp) >= 2 else ""

def _clean_facility_name(name: str) -> str:
    if not name: return ""
    s = re.sub(r"(내부|외부)", "", name)
    s = re.sub(r"#\d+", "", s)
    return s.strip()

def _format_item(meta: Dict[str, Any]) -> str:  # 레거시 간단 포맷
    name = _clean_facility_name(str(meta.get("elvtr_nm") or ""))
    lv   = _format_levels(meta)
    pos  = meta.get("instl_pstn") or ""
    st   = meta.get("use_status_raw") or "정보없음"
    return f"{name} {lv}, 위치:{pos}, {st}"

def _meta_key(m: Dict[str, Any]) -> str:
    uid = m.get("uid")
    if uid: return str(uid)
    lp = m.get("levels_path") or []
    seg = f"{lp[0]}↔{lp[-1]}" if len(lp) >= 2 else (m.get("opr_sec") or "")
    return "|".join([
        str(m.get("station_name") or ""),
        str(m.get("line_no") or ""),
        str(m.get("equipment_type") or ""),
        str(m.get("elvtr_nm") or ""),
        str(m.get("instl_pstn") or ""),
        seg,
        str("" if m.get("exit_no") is None else m.get("exit_no")),
    ])

def _dedup_metas(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """기본 dedup: 완전 동일 키만 제거 (#1/#2 같은 유닛은 보존 X)"""
    seen, out = set(), []
    for m in metas:
        k = _meta_key(m)
        if k in seen: continue
        seen.add(k); out.append(m)
    return out

def _dedup_metas_preserve_units(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """유닛(#1/#2)까지 포함해서 구분 (외부/출구 안내용)"""
    seen, out = set(), []
    for m in metas:
        lp = m.get("levels_path") or []
        seg = f"{lp[0]}↔{lp[-1]}" if len(lp) >= 2 else (m.get("opr_sec") or "")
        key = "|".join([
            str(m.get("station_name") or ""),
            str(m.get("line_no") or ""),
            str(m.get("equipment_type") or ""),
            str(m.get("elvtr_nm") or ""),   # 유닛 보존
            str(m.get("instl_pstn") or ""),
            seg,
            str("" if m.get("exit_no") is None else m.get("exit_no")),
        ])
        if key in seen: continue
        seen.add(key); out.append(m)
    return out

def _status_sentence_short(meta: Dict[str, Any]) -> str:
    std = meta.get("use_status_std")
    raw = meta.get("use_status_raw") or ""
    if std == "정상운행" or any(x in raw for x in ["정상", "가능", "운행"]): return "사용 가능합니다."
    if std == "운행중지" or any(x in raw for x in ["중지", "고장", "점검"]): return "운행 중지입니다."
    return "정보없음입니다."

def _make_anchor_label(m: Dict[str, Any]) -> str:
    st = (m.get("station_name") or "").strip()
    tlabel = (m.get("equipment_type") or "엘리베이터").strip()
    if m.get("exit_no") is not None: return f"{st} {m['exit_no']}번 출구 {tlabel}"
    if m.get("car_hint"): return f"{st} {m['car_hint']}칸 {tlabel}"
    if m.get("direction_hint"): return f"{st} {m['direction_hint']} {tlabel}"
    return f"{st} {tlabel}"

def _format_item_line(meta: Dict[str, Any]) -> str:
    lp = meta.get("levels_path") or []
    seg = f"({lp[0]}↔{lp[-1]})" if len(lp) >= 2 else ""
    anchor = _make_anchor_label(meta)
    pos = (meta.get("instl_pstn") or "").strip()
    pos_txt = f" — 위치: {pos}" if pos else ""
    return f"{anchor} {seg}{pos_txt} — 현재 {_status_sentence_short(meta)}".strip()

def _split_internal_external(metas: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    internal = [m for m in metas if not m.get("is_surface_link")]
    external = [m for m in metas if m.get("is_surface_link")]
    return _dedup_metas(internal), _dedup_metas(external)

def _is_down(m: Dict[str, Any]) -> bool:
    std = m.get("use_status_std")
    raw = (m.get("use_status_raw") or "")
    if std == "운행중지": return True
    return any(k in raw for k in ["중지", "고장", "점검", "불가", "미운행"])

# =========================
# 12) 실시간 가동현황 갱신
# =========================
async def _refresh_realtime_status(metas: List[Dict[str, Any]], station_hint: Optional[str]) -> List[Dict[str, Any]]:
    if not metas: return metas
    station_name = station_hint or metas[0].get("station_name")
    if callable(fetch_realtime_status_by_station) and station_name:
        try:
            by_station = await fetch_realtime_status_by_station(station_name)  # {key: status}
            def key_of(m: Dict[str, Any]) -> str:
                base = (str(m.get("station_name") or "") + "|" +
                        str(m.get("elvtr_nm") or "") + "|" +
                        str(m.get("instl_pstn") or ""))
                return re.sub(r"\s+", "", base)
            cache = {re.sub(r"\s+", "", k): v for k, v in (by_station or {}).items()}
            for m in metas:
                k = key_of(m)
                if k in cache and cache[k]:
                    m["use_status_raw"] = str(cache[k])
            return metas
        except Exception:
            pass
    if callable(fetch_realtime_status):
        for m in metas:
            uid = m.get("uid")
            try:
                if uid:
                    r = await fetch_realtime_status(uid)
                    if r:
                        m["use_status_raw"] = str(r)
            except Exception:
                continue
    return metas

# =========================
# 13) 폴백(정확일치) + 환승(2-hop)
# =========================
def _fallback_metas_by_db(station_hint: Optional[str], line_no: Optional[str], question: str, limit: int = 6) -> List[Dict[str, Any]]:
    if not station_hint:
        station_hint = _infer_station_from_question(question)
    all_items: List[Dict[str, Any]] = list(ELEV_DB.values())
    if station_hint:
        metas = [m for m in all_items if _station_eq(m.get("station_name"), station_hint)]
    else:
        metas = all_items[:]
    if line_no is None:
        m = re.search(r"(\d+)\s*호선", question or "")
        line_no = m.group(1) if m else None
    if line_no:
        metas = [m for m in metas if (m.get("line_no") or "") == line_no]

    targets, exit_no, direction, car_hint, want_type, prefer_internal = _parse_constraints_from_question(question)
    if want_type:
        typed = [m for m in metas if (m.get("equipment_type") or "") == want_type]
        metas = typed or metas
    needs_surface = "1F" in [t.upper() for t in targets]
    ranked = rank_items(metas, targets, exit_no, direction, car_hint, needs_surface, prefer_internal)
    return ranked[:limit]

def _two_hop_suggestion(station_hint: Optional[str], line_no: Optional[str], question: str, limit: int = 2) -> Optional[Dict[str, Any]]:
    st = station_hint or _infer_station_from_question(question) or ""
    all_items: List[Dict[str, Any]] = list(ELEV_DB.values())
    pool = [m for m in all_items if _station_eq(m.get("station_name"), st)]
    pool = [m for m in pool if m.get("equipment_type") == "엘리베이터"]
    internal, _ = _split_internal_external(pool)
    if not internal: return None

    m = re.search(r"(\d+)\s*호선에서\s*(\d+)\s*호선(?:으로)?\s*환승", question or "")
    from_ln = m.group(1) if m else (line_no or None)
    to_ln   = m.group(2) if m else None

    station_lines = sorted({m.get("line_no") for m in internal if m.get("line_no")})
    if from_ln and not to_ln:
        others = [x for x in station_lines if x != from_ln]
        if len(others) == 1:
            to_ln = others[0]
    if to_ln and not from_ln:
        others = [x for x in station_lines if x != to_ln]
        if len(others) == 1:
            from_ln = others[0]
    if not from_ln and not to_ln and len(station_lines) >= 2:
        from_ln, to_ln = station_lines[0], station_lines[1]
    if from_ln and to_ln and from_ln == to_ln:
        return None

    def _seg(mm: Dict[str, Any]) -> str:
        lp = mm.get("levels_path") or []
        return f"{lp[0]}↔{lp[-1]}" if lp else (mm.get("opr_sec") or "")

    def pick(ln: Optional[str]) -> List[Dict[str, Any]]:
        cands = internal
        if ln:
            cands = [x for x in cands if (x.get("line_no") or "") == ln]
        cands = sorted(cands, key=lambda x: (0 if (x.get("car_hint") or x.get("direction_hint")) else 1, _seg(x)))
        return _dedup_metas(cands)[:limit]

    step1 = pick(from_ln)
    step2 = pick(to_ln)
    if not step1 or not step2:
        return None

    st_name = step1[0].get("station_name") if step1 else (step2[0].get("station_name") if step2 else "")
    head = _display_station(st_name)
    body1 = "\n".join(f"- {_format_item_line(x)}" for x in step1)
    body2 = "\n".join(f"- {_format_item_line(x)}" for x in step2)
    answer = f"{head} 환승 동선 안내:\n① {from_ln or ''}호선 승강장→대합실\n{body1}\n② 대합실→{to_ln or ''}호선 승강장\n{body2}".strip()
    return {"answer": answer, "retrieved": (step1 + step2)}

# =========================
# 14) 의도(intent) 감지
# =========================
def _detect_intent(question: str) -> Dict[str, Any]:
    q = (question or "")

    exit_m = re.search(r"(?<!\d)(\d{1,2})\s*번\s*출\s*구\s*(?:엘리베이터|리프트)?", q)
    want_all_elev = bool(re.search(r"(역|전체|전부|모두).*(엘리베이터).*(있|보여|알려|\?)", q)) \
        or bool(re.search(r"(엘리베이터).*(전부|모두|전체|다)", q))
    transfer_m = re.search(r"(\d+)\s*호선에서\s*(\d+)\s*호선(?:으로)?\s*환승", q)
    want_transfer = bool(transfer_m or re.search(r"환\s*승", q))
    want_concourse = bool(
        re.search(r"대합실\s*로\s*가(는|려면|요)", q)
        or re.search(r"대합실.*(엘리베이터|리프트)", q)
        or re.search(r"환승통로", q)
    ) and not re.search(r"출\s*구", q)
    want_external = bool(re.search(r"(밖|지상|나가|바깥|출\s*구\s*로|밖으로|지상으로)", q))
    want_down_only = bool(re.search(r"(운행\s*중지|중지|고장|점검|사용\s*불가|불가|미운행)", q))
    want_restroom = bool(re.search(r"(장애인\s*화장실|장애인용\s*화장실|무장애\s*화장실)", q))
    car_m = re.search(r"\b(\d-\d)\b", q)
    dir_m = re.search(r"([가-힣A-Za-z0-9]+)\s*방면", q)

    return {
        "exit_query": int(exit_m.group(1)) if exit_m else None,
        "want_all_elev": want_all_elev,
        "transfer": want_transfer,
        "want_concourse": want_concourse,
        "want_external": want_external,
        "want_down_only": want_down_only,
        "want_restroom": want_restroom,
        "car_hint": car_m.group(1) if car_m else None,
        "direction_hint": (dir_m.group(1) + " 방면") if dir_m else None,
    }

# =========================
# 15) Pydantic 모델
# =========================
class AskRequest(BaseModel):
    question: str = Field(..., description="질문")
    station_hint: Optional[str] = Field(None)
    line_no: Optional[str] = Field(None)

# =========================
# 16) 스타트업/헬스
# =========================
@app.on_event("startup")
def _startup():
    _load_db()
    _load_restrooms_from_csv(REST_CSV_PATH)  # CSV 로드 (REST_DB 갱신)

@app.get("/health")
def health():
    try:
        seoul_base  = getattr(settings, "SEOUL_API_BASE_URL", None) if settings else None
        dataset_id  = getattr(settings, "SEOUL_DATASET_ID", None) if settings else None
        rag_top_k   = getattr(settings, "RAG_TOP_K", 6) if settings else 6
        info = {
            "status": "ok",
            "gen_model": GENERATION_MODEL,
            "openai_base": OPENAI_BASE_URL,
            "seoul_base": seoul_base,
            "dataset_id": dataset_id,
            "rag_top_k": rag_top_k,
            "db_items": len(ELEV_DB),
            "rest_rows": len(REST_DB),
        }
        return info
    except Exception as e:
        return {"status": "degraded", "error": str(e), "db_items": len(ELEV_DB), "rest_rows": len(REST_DB)}

# =========================
# 17) 디버그/도우미(엘리베이터/화장실)
# =========================
@app.get("/debug/seoul/total")
async def debug_total():
    if not fetch_total_count:
        return {"error": "seoul_api.fetch_total_count 미구현", "list_total_count": 0}
    total = await fetch_total_count()
    return {"list_total_count": total}

@app.get("/debug/seoul/sample")
async def debug_sample(start: int = Query(1, ge=1), end: int = Query(5, ge=1)):
    if not fetch_page:
        return {"error": "seoul_api.fetch_page 미구현", "count": 0, "preview": []}
    rows = await fetch_page(start, end)
    preview: List[Dict[str, Any]] = []
    for r in rows[:10]:
        preview.append({
            "STN_CD": r.get("STN_CD"),
            "STN_NM": r.get("STN_NM"),
            "ELVTR_NM": r.get("ELVTR_NM"),
            "OPR_SEC": r.get("OPR_SEC"),
            "INSTL_PSTN": r.get("INSTL_PSTN"),
            "USE_YN": r.get("USE_YN"),
            "ELVTR_SE": r.get("ELVTR_SE"),
        })
    return {"count": len(rows), "preview": preview}

@app.post("/restrooms/upload")
async def restrooms_upload(file: UploadFile = File(...)):
    try:
        os.makedirs(os.path.dirname(REST_CSV_PATH), exist_ok=True)
        content = await file.read()
        with open(REST_CSV_PATH, "wb") as out:
            out.write(content)
        rows = _load_restrooms_from_csv(REST_CSV_PATH)
        _save_rest_db()
        return {"saved_as": REST_CSV_PATH, "rows_loaded": int(rows)}
    except Exception as e:
        return {"saved_as": REST_CSV_PATH, "rows_loaded": None, "error": str(e)}

@app.get("/restrooms/count")
def restrooms_count():
    return {"rows": len(REST_DB)}

@app.get("/restrooms/search")
def restrooms_search(station: str = Query(...), line_no: Optional[str] = Query(None)):
    key = _station_key(station)
    def _line_match(line_name: str) -> bool:
        if not line_no:
            return True
        ln = (line_name or "").replace(" ", "")
        return ln == f"{line_no}호선" or ln.startswith(f"{line_no}호선")
    hits = [
        r for r in REST_DB
        if _station_key(r["station_name"]) == key and _line_match(r["line_name"] or "")
    ]
    return {"count": len(hits), "items": hits[:50]}

@app.post("/restrooms/reload")
def restrooms_reload():
    rows = _load_restrooms_from_csv(REST_CSV_PATH)
    _save_rest_db()
    return {"rows_loaded": int(rows)}

@app.get("/debug/db/count")
def db_count():
    return {"total_in_db": len(ELEV_DB)}

@app.get("/debug/index/count")
def index_count():
    try:
        return {"index_count": int(collection_count())}
    except Exception:
        return {"index_count": 0, "note": "collection_count 미구현"}

# =========================
# 18) 수집/인덱싱 (엘리베이터/RAG)
# =========================
@app.post("/ingest")
async def ingest():
    if not (fetch_total_count and fetch_page and normalize_row and make_uid):
        return {"ingested": 0, "created": 0, "updated": 0, "skipped": 0, "message": "ingest 의존성 미구현"}

    total = await fetch_total_count()
    if not total or total <= 0:
        return {"indb": len(ELEV_DB), "ingested": 0, "created": 0, "updated": 0, "skipped": 0, "message": "총 데이터 건수 확인 실패"}

    page = 1000
    tasks = [fetch_page(start, min(start + page - 1, total)) for start in range(1, total + 1, page)]
    pages = await asyncio.gather(*tasks)

    created = updated = skipped = 0
    ids: List[str] = []
    docs: List[str] = []
    metas_for_index: List[Dict[str, Any]] = []

    for rows in pages:
        for r in rows:
            n = normalize_row(r)
            if not n:
                skipped += 1
                continue
            uid = make_uid(n)
            n["uid"] = uid
            if uid in ELEV_DB:
                updated += 1
            else:
                created += 1
            ELEV_DB[uid] = n
            ids.append(uid)
            docs.append(build_doc_text(n))
            metas_for_index.append(n)

    _save_db()

    indexed = 0
    index_error = None
    if ids and callable(embed_texts) and callable(upsert_docs):
        try:
            safe_metas = [_sanitize_meta(m) for m in metas_for_index]
            embs = await embed_texts(docs)
            await upsert_docs(ids, docs, safe_metas, embs)
            indexed = len(ids)
        except Exception as e:
            index_error = str(e)

    return {
        "ingested": created + updated, "created": created, "updated": updated, "skipped": skipped,
        "total_in_db": len(ELEV_DB), "indexed": indexed, "index_error": index_error
    }

# (선택) CSV → JSON 캐시식 인덱싱이 필요하면 별도 구현 가능
# 현재는 REST_DB(list) 그대로 사용

# =========================
# 19) 단순 검색 API
# =========================
@app.get("/elevators/search")
def elevators_search(
    station: str = Query(..., description="역명 (정확 일치)"),
    line_no: Optional[str] = Query(None),
    target_levels: Optional[str] = Query(None, description="CSV 예: 1F,B1"),
    exit_no: Optional[int] = Query(None),
    direction: Optional[str] = Query(None),
    car_hint: Optional[str] = Query(None),
    kind: Optional[str] = Query(None, description="internal|external"),
    limit: int = Query(5, ge=1, le=20),
    debug: bool = Query(False)
):
    all_items: List[Dict] = list(ELEV_DB.values())
    metas = [m for m in all_items if _station_eq(m.get("station_name"), station)]
    if line_no:
        metas = [m for m in metas if (m.get("line_no") or "") == line_no]
    if kind in ("internal", "external"):
        internal, external = _split_internal_external(metas)
        metas = internal if kind == "internal" else external

    targets = [t.strip().upper() for t in (target_levels.split(",") if target_levels else []) if t.strip()]
    needs_surface = "1F" in targets
    prefer_internal = (not needs_surface) and (bool(direction) or bool(car_hint))

    ranked: List[Dict[str, Any]] = rank_items(
        metas, targets, exit_no, direction, car_hint, needs_surface, prefer_internal
    ) if metas else []

    ranked = _dedup_metas(ranked)

    items = [{
        "line_name": m.get("line_name"),
        "type": m.get("equipment_type"),
        "levels_path": m.get("levels_path"),
        "install": m.get("instl_pstn"),
        "status": m.get("use_status_raw"),
        "exit_no": m.get("exit_no"),
        "direction_hint": m.get("direction_hint"),
        "car_hint": m.get("car_hint"),
    } for m in ranked[:limit]]

    resp = {"station_name": station, "line_no": line_no, "items": items}
    if debug:
        resp["debug"] = {"result_count": len(ranked)}
    return resp

@app.get("/toilets/search")
def toilets_search(
    station: str = Query(..., description="역명 (정확 일치)"),
    line_no: Optional[str] = Query(None),
    gate: Optional[str] = Query(None, description="내부|외부"),
    limit: int = Query(10, ge=1, le=50),
):
    metas = [m for m in REST_DB if _station_eq(m.get("station_name"), station)]
    if line_no:
        metas = [m for m in metas if str(m.get("line_no") or "") == str(line_no)]
    if gate in ("내부", "외부"):
        metas = [m for m in metas if (m.get("gate") or "") == gate]
    items = [{
        "line_no": m.get("line_no"),
        "line_name": m.get("line_name"),
        "station_name": m.get("station_name"),
        "ground": m.get("ground"),
        "floor": m.get("floor"),
        "gate": m.get("gate"),
        "exit_no": m.get("exit_no"),
        "detail": m.get("detail"),
        "phone": m.get("phone"),
        "open_hours": m.get("open_hours"),
    } for m in metas[:limit]]
    return {"station_name": station, "line_no": line_no, "count": len(items), "items": items}

# =========================
# 20) LLM + RAG 메인 엔드포인트
# =========================
@app.post("/ask")
async def ask_llm(body: Dict[str, Any] = Body(...)):
    if body.get("_debug_echo"):
        return {"answer": "ECHO", "retrieved": [], "fallback": False, "llm_error": None}

    try:
        question: Optional[str] = body.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="question 필수")

        # 변수(요청옵션): 응답을 표로 받을지
        response_format: str = (body.get("format") or body.get("response_format") or "").lower()  # "table" 기대
        want_table: bool = (response_format == "table")
        intent = _detect_intent(question)
        want_summary: bool = bool(body.get("summary") or body.get("with_summary"))
        summary_mode: str = (body.get("summary_mode") or "rule").lower()

        # === [FAST PATH] 장애인 화장실: CSV만 ===
        if intent.get("want_restroom"):
            st_name = body.get("station_hint") or _infer_station_from_question(question) or ""
            if not st_name:
                qnorm = re.sub(r"\s+", "", question or "")
                cand_names = set()
                cand_names.update({m.get("station_name") for m in ELEV_DB.values() if m.get("station_name")})
                cand_names.update({m.get("station_name") for m in REST_DB if m.get("station_name")})
                for name in sorted(cand_names, key=lambda x: len(x or ""), reverse=True):
                    if name and re.search(re.escape(name) + r"(역)?", qnorm):
                        st_name = name; break

            m_ln = re.search(r"(\d+)\s*호선", question or "")
            line_no = body.get("line_no") or (m_ln.group(1) if m_ln else None)

            pool = [r for r in REST_DB if _station_eq(r.get("station_name"), st_name)]
            if line_no:
                pool = [r for r in pool if str(r.get("line_no") or "") == str(line_no)]

            if not pool:
                return {"answer": f"{_display_station(st_name)} 기준 장애인 화장실 정보를 찾지 못했어요.", "retrieved": [],
                        "fallback": False, "llm_error": None}

            if want_table:
                md = format_facilities_as_tables(pool, [])  # 화장실만 표
                summary_txt = ""
                if want_summary:
                    summary_txt = await _summarize(pool, [], station_hint=st_name, line_no=line_no, mode=summary_mode)
                return {
                    "answer": (md + ("\n\n---\n" + summary_txt if summary_txt else "")),
                    "answer_format": "markdown_table+summary" if summary_txt else "markdown_table",
                    "retrieved": pool[:10],
                    "fallback": False,
                    "llm_error": None
                }


            lines = [_format_restroom_line(r) for r in pool]
            if len(pool) == 1:
                return {"answer": lines[0], "retrieved": pool, "fallback": False, "llm_error": None}
            header = f"{_display_station(st_name)} 장애인 화장실은 총 {len(pool)}개입니다."
            return {"answer": header + "\n" + "\n".join(lines), "retrieved": pool[:10], "fallback": False, "llm_error": None}

        # ─ 엘리베이터 의도
        exit_query_no: Optional[int] = intent["exit_query"]
        want_all_elev: bool = intent["want_all_elev"]
        want_transfer: bool = intent["transfer"]
        want_concourse: bool = intent["want_concourse"]
        want_external: bool = intent["want_external"]
        want_down_only: bool = intent["want_down_only"]

        # ─ LLM 파싱(선행)
        parsed = await _llm_parse_question(question)
        station_hint: Optional[str] = body.get("station_hint") or parsed.get("station_hint")
        line_no: Optional[str]   = body.get("line_no")    or parsed.get("line_no")

        targets, exit_no, direction, car_hint, want_type, prefer_internal = _parse_constraints_from_question(question)
        targets = list({*(t.upper() for t in (parsed.get("targets") or [])), *(t.upper() for t in targets)})
        exit_no = exit_no if exit_no is not None else parsed.get("exit_no")
        direction = (parsed.get("direction") or direction)
        car_hint = (parsed.get("car_hint") or car_hint)
        want_type = (parsed.get("want_type") or want_type)
        prefer_internal = prefer_internal or bool(parsed.get("prefer_internal"))
        needs_surface: bool = "1F" in [t.upper() for t in targets]

        if not station_hint:
            station_hint = _infer_station_from_question(question)
        if not line_no:
            m = re.search(r"(\d+)\s*호선", question or "")
            line_no = m.group(1) if m else None

        # ─ RAG 조회(가능 시)
        metas: List[Dict[str, Any]] = []
        try:
            if USE_RAG_ELEV and get_collection and settings and callable(embed_texts):
                col = get_collection()
                q_emb = await embed_texts([question])
                res = col.query(query_texts=q_emb, n_results=getattr(settings, "RAG_TOP_K", 6)) or {}
                metas = (res.get("metadatas") or [[]])[0]
        except Exception:
            metas = []

        # 1차 필터
        def _apply_filters(ms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = ms[:]
            if station_hint:
                f = [m for m in out if _station_eq(m.get("station_name"), station_hint)]
                out = f or out
            if line_no:
                f = [m for m in out if (m.get("line_no") or "") == line_no]
                out = f or out
            if want_type:
                f = [m for m in out if (m.get("equipment_type") or "") == want_type]
                out = f or out
            return out

        metas = _apply_filters(metas)
        if metas:
            metas = rank_items(metas, targets, exit_no, direction, car_hint, needs_surface, prefer_internal)[:6]

        # 폴백: DB
        if not metas:
            metas = _fallback_metas_by_db(station_hint, line_no, question, limit=6)
        if not metas:
            return {"answer": "관련 설비를 찾지 못했어요.", "retrieved": [], "fallback": True, "llm_error": None}

        # 실시간 갱신
        metas = await _refresh_realtime_status(metas, station_hint)
        st_name = station_hint or (metas[0].get("station_name") if metas else "")

        # === 운행중지/고장 전용 ===
        if want_down_only:
            if not st_name:
                return {"answer": "역명을 인식하지 못했어요.", "retrieved": [], "fallback": True, "llm_error": None}
            pool = [m for m in ELEV_DB.values() if _station_eq(m.get("station_name"), st_name)]
            if line_no:
                pool = [m for m in pool if (m.get("line_no") or "") == line_no]
            pool = [m for m in pool if m.get("equipment_type") == "엘리베이터"]
            pool = await _refresh_realtime_status(pool, st_name)
            downs = _dedup_metas([m for m in pool if _is_down(m)])
            head = _display_station(st_name)
            if not downs:
                return {"answer": f"{head} 기준 운행중지 엘리베이터는 확인되지 않았어요.", "retrieved": [], "fallback": False, "llm_error": None}
            header = f"{head} 운행중지/고장 엘리베이터는 총 {len(downs)}개입니다."
            body = "\n".join(_format_item_line(x) for x in downs)
            return {"answer": header + "\n" + body, "retrieved": downs[:6], "fallback": False, "llm_error": None}

        # A) 환승
        if want_transfer:
            m = re.search(r"(\d+)\s*호선에서\s*(\d+)\s*호선", question or "")
            if m and m.group(1) == m.group(2):
                want_concourse = True
            else:
                twohop = _two_hop_suggestion(station_hint, line_no, question, limit=2)
                if twohop and twohop.get("retrieved"):
                    return {"answer": twohop["answer"], "retrieved": twohop["retrieved"], "fallback": False, "llm_error": None}
                # 폴백
                mm = re.search(r"(\d+)\s*호선에서\s*(\d+)\s*호선", question or "")
                from_ln = mm.group(1) if mm else (line_no or None)
                to_ln = mm.group(2) if mm else None
                pool = [m for m in ELEV_DB.values() if _station_matches(m.get("station_name"), st_name)]
                internal, _ = _split_internal_external([x for x in pool if x.get("equipment_type") == "엘리베이터"])
                def pick(ln: Optional[str]) -> List[Dict[str, Any]]:
                    c = internal
                    if ln: c = [x for x in c if (x.get("line_no") or "") == ln]
                    return _dedup_metas(sorted(c, key=lambda x: (0 if (x.get("car_hint") or x.get("direction_hint")) else 1)))[:2]
                s1, s2 = pick(from_ln), pick(to_ln)
                if s1 and s2:
                    head = _display_station(st_name)
                    part1 = "\n".join(f"- {_format_item_line(x)}" for x in s1)
                    part2 = "\n".join(f"- {_format_item_line(x)}" for x in s2)
                    return {
                        "answer": f"{head} 환승 동선 안내:\n① {from_ln or ''}호선 승강장→대합실\n{part1}\n② 대합실→{to_ln or ''}호선 승강장\n{part2}",
                        "retrieved": (s1 + s2),
                        "fallback": True,
                        "llm_error": None
                    }

        # B) 대합실(내부)
        if want_concourse:
            pool = [m for m in ELEV_DB.values() if _station_matches(m.get("station_name"), st_name)]
            if line_no:
                pool = [m for m in pool if (m.get("line_no") or "") == line_no]
            pool_types = (want_type,) if want_type else ("엘리베이터", "휠체어 리프트")
            typed = [m for m in pool if (m.get("equipment_type") in pool_types)]
            internal, external = _split_internal_external(typed)
            kinds = sorted({m.get("equipment_type") for m in typed})
            label = "엘리베이터/리프트" if len(kinds) > 1 else (kinds[0] if kinds else "엘리베이터/리프트")
            if internal:
                header = f"{_display_station(st_name)} 대합실로 가는 {label}는 총 {len(internal)}개입니다."
                body = "\n".join(_format_item_line(x) for x in internal)
                return {"answer": header + "\n" + body, "retrieved": internal, "fallback": False, "llm_error": None}
            if external:
                header = f"{_display_station(st_name)} 대합실 직결 {label} 정보를 찾지 못했어요. 대신 지상과 연결된 설비입니다."
                body = "\n".join(_format_item_line(x) for x in external[:3])
                return {"answer": header + "\n" + body, "retrieved": external[:3], "fallback": True, "llm_error": None}

        # C) 외부(출구 연결)
        if want_external:
            pool = [m for m in ELEV_DB.values() if _station_matches(m.get("station_name"), st_name)]
            pool = [m for m in pool if m.get("equipment_type") in ("엘리베이터", "휠체어 리프트")]
            _, external = _split_internal_external(pool)
            if line_no:
                external = [x for x in external if (x.get("line_no") or "") == line_no]
            external = _dedup_metas_preserve_units(external)
            if external:
                external = sorted(
                    external,
                    key=lambda x: (
                        999 if x.get("exit_no") is None else int(x.get("exit_no")),
                        str(x.get("instl_pstn") or ""),
                        str(x.get("equipment_type") or "")
                    )
                )
                kinds = sorted({m.get("equipment_type") for m in external})
                label = "엘리베이터/리프트" if len(kinds) > 1 else (kinds[0] if kinds else "엘리베이터")
                header = f"{_display_station(st_name)} 밖으로 나가는 {label}는 총 {len(external)}개입니다."
                body = "\n".join(_format_item_line(x) for x in external)
                return {"answer": header + "\n" + body, "retrieved": external, "fallback": False, "llm_error": None}

        # D) 칸/방면 필터
        car_filter = intent.get("car_hint") or car_hint or parsed.get("car_hint")
        dir_filter = intent.get("direction_hint") or parsed.get("direction") or direction
        if car_filter or dir_filter:
            pool = [m for m in ELEV_DB.values() if _station_matches(m.get("station_name"), st_name)]
            if line_no:
                pool = [m for m in pool if (m.get("line_no") or "") == line_no]
            internal, _ = _split_internal_external([m for m in pool if m.get("equipment_type") == "엘리베이터"])
            cand = internal
            if car_filter:
                cand = [m for m in cand if (m.get("car_hint") == car_filter)]
            if dir_filter:
                key = (dir_filter or "").replace(" 방면", "")
                cand = [m for m in cand if (m.get("direction_hint") or "").find(key) >= 0]
            cand = _dedup_metas(cand)
            if cand:
                header = f"{_display_station(st_name)} {car_filter or dir_filter} 엘리베이터는 총 {len(cand)}개입니다."
                body = "\n".join(_format_item_line(x) for x in cand)
                return {"answer": header + "\n" + body, "retrieved": cand, "fallback": False, "llm_error": None}
            if internal:
                rec = _dedup_metas(internal)[:3]
                header = f"{_display_station(st_name)}에서 요청하신 조건의 엘리베이터를 찾지 못했어요. 대신 내부 엘리베이터를 안내합니다."
                body = "\n".join(_format_item_line(x) for x in rec)
                return {"answer": header + "\n" + body, "retrieved": rec, "fallback": True, "llm_error": None}

        # E) 특정 출구 단건
        if exit_query_no is not None:
            pool = [m for m in ELEV_DB.values() if _station_matches(m.get("station_name"), st_name)]
            if line_no:
                pool = [m for m in pool if (m.get("line_no") or "") == line_no]
            exact = [m for m in pool if m.get("exit_no") == exit_query_no and m.get("equipment_type") == "엘리베이터"]
            exact = _dedup_metas(exact)
            exact_surface = [m for m in exact if m.get("is_surface_link")]
            hit = exact_surface[0] if exact_surface else (exact[0] if exact else None)
            if hit:
                line_txt = _format_item_line(hit)
                return {"answer": f"네. {st_name} {exit_query_no}번 출구에 엘리베이터가 있습니다.\n{line_txt}", "retrieved": [hit], "fallback": False, "llm_error": None}
            alts = [m for m in pool if (m.get("exit_no") is not None and m.get("equipment_type") == "엘리베이터")]
            seen_exit, alts_unique = set(), []
            for m in _dedup_metas(alts):
                no = m.get("exit_no")
                if no in seen_exit: continue
                seen_exit.add(no); alts_unique.append(m)
            if alts_unique:
                lines = "\n".join(_format_item_line(m) for m in alts_unique[:3])
                return {"answer": f"아니요. {st_name} {exit_query_no}번 출구 엘리베이터는 확인되지 않습니다.\n대신 다음을 이용하세요:\n{lines}", "retrieved": alts_unique[:3], "fallback": False, "llm_error": None}
            return {"answer": f"아니요. {st_name} {exit_query_no}번 출구 엘리베이터는 확인되지 않습니다.", "retrieved": [], "fallback": False, "llm_error": None}

        # F) 역 전체 엘리베이터 요약
        if want_all_elev or re.search(r"역.*엘리베이터.*(있니|있어|\?)", question or ""):
            pool = [m for m in ELEV_DB.values() if _station_matches(m.get("station_name"), st_name)]
            if line_no:
                pool = [m for m in pool if (m.get("line_no") or "") == line_no]
            els = [m for m in pool if m.get("equipment_type") == "엘리베이터"]
            els = _dedup_metas(els)

            if els:
                if want_table:
                    elev_rows_ready = []
                    for m in els:
                        elev_rows_ready.append({
                            "line_name": m.get("line_name") or (m.get("line_no") and f"{m['line_no']}호선"),
                            "levels_path": m.get("levels_path"),
                            "instl_pstn": m.get("instl_pstn"),
                            "use_status_raw": m.get("use_status_raw"),
                        })
                    md = "### 🛗 엘리베이터\n" + _md_table(ELEVATOR_COLUMNS, _elevator_rows(elev_rows_ready))
                    return {
                        "answer": md,
                        "answer_format": "markdown_table",
                        "retrieved": els,
                        "fallback": False,
                        "llm_error": None
                    }

                # (기존 문장형)
                count = len(els)
                head_txt = " ".join(
                    [s for s in [_display_station(st_name), (f"{line_no}호선" if line_no else "")] if s]).strip()
                header = f"{head_txt} 엘리베이터는 총 {count}개입니다."
                body = "\n".join(_format_item_line(x) for x in els)
                return {"answer": header + "\n" + body, "retrieved": els, "fallback": False, "llm_error": None}

        # G) 일반 질의 → LLM 한 문장
        sys_p = (
            "당신은 교통약자를 돕는 서울 지하철 설비 안내 도우미입니다.\n"
            "- 한국어로 짧고 명료하게 한 문장으로 답하세요.\n"
            "- 특정 출구를 물었다면 그 출구만, 역 전체를 물었다면 총 개수/출구번호 요약.\n"
            "- 층 표기는 'B1↔1F' 등 간단히, 상태(use_status_raw) 반영.\n"
            "- 주어진 '관련 설비 정보'만 근거로 답하고 모르면 모른다고 답하세요."
        )
        bullets: List[str] = []
        for m in metas:
            lp = m.get("levels_path") or []
            seg = f"{lp[0]}↔{lp[-1]}" if len(lp) >= 2 else (m.get("opr_sec") or "정보없음")
            ex = (f"{m.get('exit_no')}번 출구" if m.get("exit_no") else None)
            inst = m.get("instl_pstn") or "정보없음"
            stat = m.get("use_status_raw") or "정보없음"
            bullets.append(
                f"- {m.get('station_name')} {m.get('line_no')}호선 {m.get('equipment_type')}: "
                + " · ".join([x for x in [seg, ex, inst] if x]) + f" · 상태 {stat}"
            )
        context = "\n".join(bullets)
        usr_p = f"질문: {question}\n\n관련 설비 정보:\n{context}\n\n한 문장으로 답하세요."

        try:
            llm_answer = await _openai_chat(
                [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                temperature=0.1
            )
        except Exception:
            llm_answer = ""

        if want_table:
            rest = []  # 이번 분기서는 화장실 없음
            elev = []
            for m in metas[:6]:
                elev.append({
                    "line_name": m.get("line_name") or (m.get("line_no") and f"{m['line_no']}호선"),
                    "levels_path": m.get("levels_path"),
                    "instl_pstn": m.get("instl_pstn"),
                    "use_status_raw": m.get("use_status_raw"),
                })
            md = format_facilities_as_tables(rest, elev)
            summary_txt = ""
            if want_summary:
                summary_txt = await _summarize([], metas[:6], station_hint=st_name, line_no=line_no)
            return {
                "answer": (md + ("\n\n---\n" + summary_txt if summary_txt else "")),
                "answer_format": "markdown_table+summary" if summary_txt else "markdown_table",
                "retrieved": metas[:6],
                "fallback": False,
                "llm_error": None
            }


        shaped = _format_item_line(metas[0])
        final_answer = (llm_answer or "").strip() or shaped
        return {"answer": final_answer, "retrieved": metas[:6], "fallback": False, "llm_error": None}

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

