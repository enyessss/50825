# OPEN API 호출 전담
# seoul_api.py

import httpx
import xmltodict
from typing import Dict, Any, List, Optional, Tuple
from settings import settings

# =========================
# 내부 헬퍼: URL/검증/파싱
# =========================
def _build_url(start: int, end: int, fmt: str = "json") -> str:
    """
    서울 열린데이터 요청 URL 조립
    - 변수(고정): SEOUL_API_BASE_URL, SEOUL_API_KEY, SEOUL_DATASET_ID
    """
    base = settings.SEOUL_API_BASE_URL.rstrip("/")  # 변수
    key = settings.SEOUL_API_KEY                    # 변수
    dset = settings.SEOUL_DATASET_ID                # 변수
    return f"{base}/{key}/{fmt}/{dset}/{start}/{end}"

def _check_result_code(raw: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    RESULT.CODE 검사 (INFO-000이면 정상)
    JSON/XML 모두 대응
    """
    dset = settings.SEOUL_DATASET_ID  # 변수
    # JSON 형태
    if dset in raw and isinstance(raw[dset], dict):
        result = raw[dset].get("RESULT")
        if isinstance(result, dict):
            code = result.get("CODE")
            msg = result.get("MESSAGE")
            return (code == "INFO-000", code, msg)
    # XML 형태
    for _, v in raw.items():
        if isinstance(v, dict) and "RESULT" in v and isinstance(v["RESULT"], dict):
            code = v["RESULT"].get("CODE")
            msg = v["RESULT"].get("MESSAGE")
            return (code == "INFO-000", code, msg)
    # RESULT 없음 → 일단 통과(일부 응답은 RESULT가 없고 row만 줄 때가 있음)
    return (True, None, None)

def _extract_rows(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    공통 row 리스트만 꺼내기 (JSON/XML 모두 대응)
    """
    dset = settings.SEOUL_DATASET_ID  # 변수
    # JSON 형태
    if dset in raw:
        inner = raw[dset]
        rows = inner.get("row") or []
        return rows if isinstance(rows, list) else [rows]
    # XML 형태 (루트 아래에 데이터셋 이름이 있고 그 안에 row)
    for _, v in raw.items():
        if isinstance(v, dict) and "row" in v:
            rows = v["row"]
            return rows if isinstance(rows, list) else [rows]
    return []

async def _fetch_raw(start: int, end: int) -> Dict[str, Any]:
    """
    원본 응답 가져오기: JSON → 실패 시 XML 폴백
    """
    url_json = _build_url(start, end, "json")  # 변수
    async with httpx.AsyncClient(timeout=30.0) as client:  # 변수
        r = await client.get(url_json)
        if r.status_code == 200:
            try:
                data = r.json()
                ok, code, msg = _check_result_code(data)
                if not ok:
                    # RESULT 에러 → XML로도 시도해 보지만, 대부분 동일 코드일 가능성 큼
                    pass
                return data
            except Exception:
                # JSON 파싱 실패 시 XML 폴백
                pass
        # XML 폴백
        url_xml = _build_url(start, end, "xml")  # 변수
        r2 = await client.get(url_xml)
        r2.raise_for_status()
        data = xmltodict.parse(r2.text)
        # XML도 RESULT 체크
        _ok, _code, _msg = _check_result_code(data)
        return data

async def _fetch_total_count() -> int:
    """
    총 데이터 건수(list_total_count)
    """
    raw = await _fetch_raw(1, 1)  # 변수
    dset = settings.SEOUL_DATASET_ID  # 변수
    # JSON
    if dset in raw and isinstance(raw[dset], dict):
        c = raw[dset].get("list_total_count")
        try:
            return int(c)
        except Exception:
            pass
    # XML
    for _, v in raw.items():
        if isinstance(v, dict) and "list_total_count" in v:
            try:
                return int(v["list_total_count"])
            except Exception:
                pass
    return 0

async def _fetch_page(start: int, end: int) -> List[Dict[str, Any]]:
    """
    구간(start~end) 데이터 행(row) 리스트
    """
    raw = await _fetch_raw(start, end)  # 변수
    return _extract_rows(raw)           # 변수

# =========================
# 내부 헬퍼: 클라이언트 측 필터/정규화
# =========================
def _norm(s: Optional[str]) -> str:
    """역명 비교용 정규화: 공백/괄호/'역' 제거 + 소문자"""
    s = (s or "").strip()
    s = s.lower()
    s = re_sub(r"\s+", "", s)
    s = s.replace("역", "")
    s = re_sub(r"[()]", "", s)
    return s

def _derive_line_no(stn_cd: Optional[str]) -> Optional[str]:
    """
    호선 추정: STN_CD의 백의 자리(예: 0150 → '1', 0249 → '2')
    주의: 100% 규칙은 아님. 추후 테이블 보정 가능.
    """
    try:
        n = int(stn_cd)  # 변수
        return str((n // 100) % 10)
    except Exception:
        return None

# 정규식 모듈이 필요하므로 import
import re as _re
def re_sub(pat: str, repl: str, s: str) -> str:
    return _re.sub(pat, repl, s)

def _match_station(row: Dict[str, Any], station_hint: Optional[str]) -> bool:
    if not station_hint:
        return True
    return _norm(row.get("STN_NM")) == _norm(station_hint) or _norm(station_hint) in _norm(row.get("STN_NM"))

def _match_line(row: Dict[str, Any], line_no: Optional[str]) -> bool:
    if not line_no:
        return True
    derived = _derive_line_no(row.get("STN_CD"))
    return (derived == line_no)

# =========================
# 공개(권장) API: fetch_rows
# =========================
async def fetch_rows(
    station_hint: Optional[str] = None,   # 변수
    line_no: Optional[str] = None,        # 변수
    limit: Optional[int] = None           # 변수: 제한 개수(없으면 전부)
) -> List[Dict[str, Any]]:
    """
    ✅ 권장: 필요한 행만 한 번에 가져오기(페이지네이션 내부 처리)
    - station_hint / line_no가 주어지면 클라이언트 측 필터 적용
    - limit로 결과 개수 제한(성능/지연 줄이기)
    - RESULT.CODE 검사, JSON 실패 시 XML 폴백
    """
    total = await _fetch_total_count()  # 변수
    if total <= 0:
        return []

    page_size = 1000  # 변수(고정): 서울 열린데이터는 1000건 페이징이 무난
    rows_out: List[Dict[str, Any]] = []  # 변수

    # 페이지 단위로 가져오면서, 조건을 만족하는 행만 모음
    for start in range(1, total + 1, page_size):
        end = min(start + page_size - 1, total)
        page_rows = await _fetch_page(start, end)  # 변수
        if not page_rows:
            continue

        # 필터링(역명/호선)
        for r in page_rows:
            if _match_station(r, station_hint) and _match_line(r, line_no):
                rows_out.append(r)
                if limit and len(rows_out) >= limit:
                    return rows_out

    return rows_out

# =========================
# 🟡 호환용 공개 함수(나중에 main.py 정리 후 제거 가능)
# =========================
async def fetch_total_count() -> int:
    """
    (호환용) 총 데이터 건수. 나중에 main.py가 fetch_rows로 전환되면 제거 가능.
    """
    return await _fetch_total_count()

async def fetch_page(start: int, end: int) -> List[Dict[str, Any]]:
    """
    (호환용) 구간 데이터. 나중에 main.py가 fetch_rows로 전환되면 제거 가능.
    """
    return await _fetch_page(start, end)
