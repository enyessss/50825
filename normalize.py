# normalize.py
# 원본 Open API row -> 표준 메타 스키마로 변환
# - ES(에스컬레이터) 제외, EL(엘리베이터)/WL(휠체어 리프트)만 포함
# - 공개 함수는 normalize_rows 하나(리스트 변환). 기존 normalize_row는 호환 래퍼.

import re
from typing import Dict, Any, List, Optional
from datetime import datetime

# =========================
# 내부 헬퍼: 표준화/파생 필드
# =========================
def _normalize_level_token(tok: str) -> Optional[str]:
    """다양한 층 표기를 표준화: B3,B2,B1,1F,2F ..."""
    t = (tok or "").strip().upper()
    # 한글 → 표준
    t = t.replace("지하", "B").replace("지상", "").replace("층", "")
    t = t.replace("B0", "B1")  # 드문 표기 보정
    # 1층/1F 보정
    if t in {"1", "1F", "F1"}:
        return "1F"
    # Bn 표기
    m = re.fullmatch(r"B\d+", t)
    if m:
        return t
    # nF 표기
    m = re.fullmatch(r"(\d+)F", t)
    if m:
        return f"{m.group(1)}F"
    return None

def _split_levels(opr_sec: Optional[str]) -> List[str]:
    """OPR_SEC을 'B2-B1', 'B1↔1F', '지하2층~지하1층', 'B2→B1' 등 분해 후 표준화"""
    if not opr_sec:
        return []
    raw = str(opr_sec)
    parts = re.split(r"\s*(?:-|↔|~|∼|→|=>|->|—)\s*", raw)
    norm = []
    for p in parts:
        t = _normalize_level_token(p)
        if t:
            norm.append(t)
    return norm

def _extract_exit_no(text: Optional[str]) -> Optional[int]:
    """설치위치/명칭에서 출구번호 추출 (여러 표기 변형 대응)"""
    if not text:
        return None
    m = re.search(r"(?<!\d)(\d{1,2})\s*번\s*(?:출|출입)?구", str(text))
    return int(m.group(1)) if m else None

def _extract_car_hint(text: Optional[str]) -> Optional[str]:
    """차량칸 힌트(예: 5-1)"""
    if not text:
        return None
    m = re.search(r"\b(\d-\d)\b", str(text))
    return m.group(1) if m else None

def _extract_direction(text: Optional[str]) -> Optional[str]:
    """방면/상행/하행 힌트 추출"""
    if not text:
        return None
    t = str(text)
    m = re.search(r"([가-힣A-Za-z0-9]+)\s*방면", t)
    if m:
        return m.group(1) + " 방면"
    if "상행" in t:
        return "상행"
    if "하행" in t:
        return "하행"
    return None

def _status_std(use_yn: Optional[str]) -> str:
    """운행상태 표준화: 정상운행 / 운행중지 / 정보없음"""
    if not use_yn:
        return "정보없음"
    t = str(use_yn).strip()
    ok = ["사용가능", "정상", "운행", "가능", "Y", "정상운행", "정상 운행", "운행중", "운행 중"]
    bad = ["점검", "고장", "중지", "불가", "N", "미운행", "운행중지", "운행 중지"]
    if any(x in t for x in ok):
        return "정상운행"
    if any(x in t for x in bad):
        return "운행중지"
    return "정보없음"

def _map_equipment(elvtr_se: Optional[str]) -> Optional[str]:
    """EL/EV -> 엘리베이터, WL -> 휠체어 리프트, ES -> 에스컬레이터"""
    if not elvtr_se:
        return None
    code = str(elvtr_se).strip().upper()
    if code in ["EL", "EV"]:
        return "엘리베이터"
    if code == "WL":
        return "휠체어 리프트"
    if code == "ES":
        return "에스컬레이터"
    return None

def _compute_route_role(levels_path: List[str], direction_hint: Optional[str], car_hint: Optional[str]) -> Optional[str]:
    """
    동선 역할 추정:
    - B1↔1F 포함: concourse↔surface (대합실<->지상)
    - 방향/칸 힌트 있거나 B2↔B1 등: platform↔concourse (승강장<->대합실)
    """
    lvset = set(levels_path)
    has_1f = "1F" in lvset
    has_b1 = "B1" in lvset
    has_dir = bool(direction_hint or car_hint)
    if has_1f and has_b1:
        return "concourse↔surface"
    if has_dir:
        return "platform↔concourse"
    if has_b1 and any(l.startswith("B") for l in lvset if l != "B1"):
        return "platform↔concourse"
    return None

def _strip_station_suffix(stn_nm: str) -> str:
    """역명에서 괄호 보조표기를 제거"""
    return re.sub(r"\s*\([^)]*\)\s*$", "", stn_nm).strip()

def _name_loc_hint(elvtr_nm: Optional[str]) -> Optional[str]:
    """
    엘리베이터 명칭에 포함된 위치 단서를 추출(출구/방면/층 키워드 제거 후 핵심만)
    예: '3번출구 외부 엘리베이터', '대합실↔승강장 엘리베이터(상행)' → '3번출구', '대합실↔승강장'
    """
    if not elvtr_nm:
        return None
    s = str(elvtr_nm)
    # 우선 출구 패턴이 있으면 그 자체가 가장 좋은 단서
    m = re.search(r"(\d{1,2}\s*번\s*출(?:입)?구)", s)
    if m:
        return re.sub(r"\s+", "", m.group(1)).replace("출입", "출")
    # 층/동선 기호 유지, 군더더기 제거
    s = re.sub(r"(엘리베이터|휠체어\s*리프트|내부|외부|상행|하행)", "", s, flags=re.I)
    s = re.sub(r"[#]\d+", "", s)        # '#1' 등 제거
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def _parse_line_no_from_stn_cd(stn_cd: str) -> Optional[str]:
    """STN_CD에서 호선 추정: '백의 자리' (예: 0150 -> '1')"""
    s = str(stn_cd).strip()
    if len(s) >= 3 and s[-3].isdigit():
        return s[-3]
    return None

# =========================
# 핵심 변환기 (단일 row → meta)
# =========================
def _normalize_single_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    내부 전용: 원본 row -> 표준 메타
    ES(에스컬레이터)는 제외, EL/WL만 반환
    """
    stn_cd = str(row.get("STN_CD") or "").strip()                     # 변수
    stn_nm = str(row.get("STN_NM") or "").strip()                     # 변수
    elvtr_nm = row.get("ELVTR_NM")                                    # 변수
    opr_sec = row.get("OPR_SEC")                                      # 변수
    instl_pstn = row.get("INSTL_PSTN")                                # 변수
    use_yn = row.get("USE_YN")                                        # 변수
    elvtr_se = row.get("ELVTR_SE")                                    # 변수

    eq_type = _map_equipment(elvtr_se)
    if eq_type not in ["엘리베이터", "휠체어 리프트"]:
        return None  # ES 제외

    line_no = _parse_line_no_from_stn_cd(stn_cd)
    levels_path = _split_levels(opr_sec)

    # 👉 출구번호 추출은 '명칭' 우선, 그다음 '설치위치'
    exit_no = _extract_exit_no(elvtr_nm or "") or _extract_exit_no(instl_pstn or "")
    direction_hint = _extract_direction(elvtr_nm or "") or _extract_direction(instl_pstn or "")
    car_hint = _extract_car_hint(elvtr_nm or "") or _extract_car_hint(instl_pstn or "")

    use_std = _status_std(use_yn)
    route_role = _compute_route_role(levels_path, direction_hint, car_hint)

    station_name = _strip_station_suffix(stn_nm)
    now_iso = datetime.utcnow().isoformat() + "Z"                      # 변수

    # 내부/외부 판별 보강: 명칭/설치위치 모두 검사
    is_internal = None
    text_pool = " ".join([str(elvtr_nm or ""), str(instl_pstn or "")])
    if "외부" in text_pool:
        is_internal = False
    elif "내부" in text_pool or route_role in {"platform↔concourse", "concourse↔surface"}:
        is_internal = True

    # 이름 기반 위치 단서
    name_loc_hint = _name_loc_hint(elvtr_nm)

    return {
        # 원본 보존 + 핵심 파생 필드
        "stn_cd": stn_cd,
        "station_name": station_name,
        "line_no": line_no,
        "line_name": f"{line_no}호선" if line_no else None,
        "elvtr_se": (str(elvtr_se).strip().upper() if elvtr_se else None),
        "equipment_type": eq_type,
        "elvtr_nm": elvtr_nm,
        "opr_sec": opr_sec,
        "levels_path": levels_path,
        "from_level": levels_path[0] if levels_path else None,
        "to_level": levels_path[-1] if levels_path else None,
        "covers_levels": list(dict.fromkeys(levels_path)),  # 중복 제거(순서 유지)
        "is_surface_link": "1F" in (set(levels_path) if levels_path else set()),
        "instl_pstn": instl_pstn,
        "exit_no": exit_no,
        "surface_exit_label": (f"{exit_no}번 출구" if exit_no is not None else None),
        "direction_hint": direction_hint,
        "car_hint": car_hint,
        "use_status_raw": use_yn,
        "use_status_std": use_std,
        "platform_level_inferred": None,   # 역 단위 보정은 나중 단계에서
        "inference_confidence": None,
        "route_role": route_role,
        "is_internal": is_internal,
        "name_loc_hint": name_loc_hint,    # ← 엘리베이터명에 담긴 위치 단서
        "location_compact": " / ".join([  # ← 응답 포맷에 바로 쓰기 좋은 요약 문자열
            p for p in [
                (elvtr_nm or "").strip() or None,
                instl_pstn and f"위치:{instl_pstn}",
                (levels_path and f"({levels_path[0]}↔{levels_path[-1]})") or None
            ] if p
        ]) or None,
        "updated_at": now_iso
    }

# =========================
# 공개 함수(권장): 리스트 변환
# =========================
def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ✅ 공개: 여러 row를 한꺼번에 표준 메타 리스트로 변환
    - ES 제외, EL/WL만 포함
    - None 결과는 자동 제거
    """
    out: List[Dict[str, Any]] = []  # 변수
    for r in rows:
        n = _normalize_single_row(r)
        if n:
            out.append(n)
    return out

# =========================
# 호환 래퍼: 단일 row 변환 (기존 코드가 호출 중일 수 있어 유지)
# =========================
def normalize_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    (호환용) 단일 row -> 메타 (내부 변환기를 호출)
    - 신규 코드는 normalize_rows 사용 권장
    """
    return _normalize_single_row(row)
