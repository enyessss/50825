# ranker.py
from typing import List, Dict, Any, Optional

# =========================
# 가중치(규칙과 일치하도록 스케일 정렬)
# =========================
TYPE_SCORE_EL: float = 0.1   # 변수(고정): EV 소프트 우선(동점 타이브레이커)
TYPE_SCORE_WL: float = 0.0   # 변수(고정): WL 추가 가중치 없음

W_TGT: float   = 2.0         # 변수(고정): 목표층 포함
W_EXIT: float  = 5.0         # 변수(고정): 출구번호 일치
W_DIR: float   = 1.0         # 변수(고정): 방면/칸 일치
W_SURF: float  = 3.0         # 변수(고정): 지상(1F) 연결
W_STAT_OK: float   = 1.0     # 변수(고정): 상태 정상
W_STAT_STOP: float = -2.0    # 변수(고정): 상태 중지

# 내부 선호(환승/승강장→대합실)
W_INTERNAL_BONUS: float   = 2.0   # 변수(고정): 내부 설비 가산
W_INTERNAL_SURF_PEN: float = 2.0  # 변수(고정): 내부 선호 시 지상연결 패널티

# 요약 길이
SNIPPET_TARGET_MIN: int = 600      # 변수(고정)
SNIPPET_TARGET_MAX: int = 900      # 변수(고정)
SNIPPET_TOPK: int       = 4        # 변수(고정): 요약에 포함할 문서 수


# =========================
# 내부 헬퍼: 항목별 점수
# =========================
def _type_score(eq_type: Optional[str]) -> float:
    if eq_type == "엘리베이터":
        return TYPE_SCORE_EL
    if eq_type == "휠체어 리프트":
        return TYPE_SCORE_WL
    return 0.0

def _covers_target(meta: Dict[str, Any], targets: List[str]) -> bool:
    if not targets:
        return True
    cov = set(meta.get("covers_levels") or [])
    return any(t in cov for t in targets)

def _exit_bonus(meta: Dict[str, Any], want_exit: Optional[int]) -> float:
    if not want_exit:
        return 0.0
    return W_EXIT if meta.get("exit_no") == want_exit else 0.0

def _dir_bonus(direction_hint: Optional[str], car_hint: Optional[str],
               want_dir: Optional[str], want_car: Optional[str]) -> float:
    s = 0.0
    if want_dir and direction_hint:
        # 부분 포함도 1점(소프트)
        if want_dir in direction_hint:
            s += W_DIR
    if want_car and car_hint:
        if want_car == car_hint:
            s += W_DIR
    return s

def _surface_bonus(meta: Dict[str, Any], needs_surface: bool) -> float:
    if not needs_surface:
        return 0.0
    return W_SURF if meta.get("is_surface_link") else 0.0

def _status_score(use_status_std: Optional[str]) -> float:
    if use_status_std == "정상운행":
        return W_STAT_OK
    if use_status_std == "운행중지":
        return W_STAT_STOP
    return 0.0

def _internal_bias(meta: Dict[str, Any], prefer_internal: bool) -> float:
    if not prefer_internal:
        return 0.0
    # 내부 설비면 +, 지상 연결이면 - (환승 의도에서 출구는 덜 적합)
    if meta.get("is_surface_link"):
        return -W_INTERNAL_SURF_PEN
    if meta.get("is_internal"):
        return W_INTERNAL_BONUS
    # route_role 힌트가 있으면 소폭 가산
    role = (meta.get("route_role") or "").lower()
    if "platform" in role:
        return W_INTERNAL_BONUS * 0.5
    return 0.0

def _target_bonus(meta: Dict[str, Any], targets: List[str]) -> float:
    return W_TGT if _covers_target(meta, targets) else 0.0


# =========================
# 내부 헬퍼: 최종 점수, 타이브레이커, 스니펫
# =========================
def _score_item(
    meta: Dict[str, Any],
    targets: List[str],
    want_exit: Optional[int],
    want_dir: Optional[str],
    want_car: Optional[str],
    needs_surface: bool,
    prefer_internal: bool
) -> float:
    s = 0.0
    s += _type_score(meta.get("equipment_type"))
    s += _target_bonus(meta, targets)
    s += _exit_bonus(meta, want_exit)
    s += _dir_bonus(meta.get("direction_hint"), meta.get("car_hint"), want_dir, want_car)
    s += _surface_bonus(meta, needs_surface)
    s += _status_score(meta.get("use_status_std"))
    s += _internal_bias(meta, prefer_internal)
    return s

def _tie_break(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """
    동점 시 결정 규칙(반환: -1= a<b, 0=동등, 1= a>b)
    1) EV 우선
    2) 상태 정상 우선
    3) 지상연결(요청이 있는 경우) 우선
    4) 절대 레벨 얕은 쪽(지상에 더 가까운) 우선
    """
    # 1) EV 우선
    ta, tb = a.get("equipment_type"), b.get("equipment_type")
    if ta != tb:
        if ta == "엘리베이터":
            return 1
        if tb == "엘리베이터":
            return -1
    # 2) 상태
    sa, sb = a.get("use_status_std"), b.get("use_status_std")
    if sa != sb:
        if sa == "정상운행":
            return 1
        if sb == "정상운행":
            return -1
    # 3) 지상연결
    if a.get("is_surface_link") != b.get("is_surface_link"):
        return 1 if a.get("is_surface_link") else -1
    # 4) 레벨 깊이(절대값)
    def depth(x: Dict[str, Any]) -> int:
        lv = (x.get("from_level") or "") + (x.get("to_level") or "")
        # 대충 B가 많을수록 더 깊다고 판단(간단 휴리스틱)
        return (lv.count("B"))
    da, db = depth(a), depth(b)
    if da != db:
        return 1 if da < db else -1
    return 0

def _fmt_seg(meta: Dict[str, Any]) -> str:
    lp = meta.get("levels_path") or []
    return f"{lp[0]}↔{lp[-1]}" if lp else (meta.get("opr_sec") or "정보없음")

def _fmt_one_line(meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"{meta.get('station_name')} {meta.get('line_no')}호선")
    parts.append(meta.get("equipment_type") or "설비")
    parts.append(_fmt_seg(meta))
    if meta.get("exit_no") is not None:
        parts.append(f"{meta.get('exit_no')}번 출구")
    if meta.get("instl_pstn"):
        parts.append(str(meta.get("instl_pstn")))
    if meta.get("use_status_raw"):
        parts.append(f"상태 {meta.get('use_status_raw')}")
    return " · ".join([p for p in parts if p and p != "None호선"])

def _make_snippets(ranked: List[Dict[str, Any]], topk: int = SNIPPET_TOPK) -> str:
    """
    상위 K개를 1줄형 스니펫으로 합쳐 600~900자 안에 맞춤
    """
    lines: List[str] = []
    for m in ranked[:max(topk, 1)]:
        lines.append("- " + _fmt_one_line(m))
    text = "\n".join(lines)
    # 길이 보정(너무 짧으면 topk를 늘리는 대신, 여기선 길이만 확인)
    if len(text) < SNIPPET_TARGET_MIN and len(ranked) > topk:
        extra = []
        for m in ranked[topk:topk+2]:
            extra.append("- " + _fmt_one_line(m))
        text = "\n".join([text] + extra)
    return text[:SNIPPET_TARGET_MAX]

def _build_sources(ranked: List[Dict[str, Any]], topk: int = SNIPPET_TOPK) -> List[Dict[str, Any]]:
    """
    프런트/LLM에 전달할 근거 정보(간단 메타) 목록
    """
    out: List[Dict[str, Any]] = []
    for m in ranked[:topk]:
        out.append({
            "title": f"{m.get('station_name')} {m.get('line_no')}호선 {m.get('equipment_type')}",
            "url": "",  # 서울 열린데이터 상세 URL을 쓰려면 settings에 템플릿 추가
            "exit_no": m.get("exit_no"),
            "levels": m.get("levels_path"),
            "status": m.get("use_status_raw"),
        })
    return out


# =========================
# 공개: 랭킹 + 압축 요약
# =========================
def rank_and_compress(
    question: str,                          # 변수
    station_hint: Optional[str],            # 변수
    line_no: Optional[str],                 # 변수
    docs: List[Dict[str, Any]]              # 변수
) -> Dict[str, Any]:
    """
    ✅ 공개 엔트리: 문서 랭킹 → 상위 K 스니펫(600~900자) 생성
    - 규칙: EV 우선(소프트), WL 포함(추가 가중치 없음), ES는 normalize 단계에서 제외
    - 가중치: 출구 +5, 지상연결 +3, 상태 +1/-2, 방면/칸 +1, 목표층 +2
    """
    # 전처리: 질문에서 힌트(목표층/출구/방면/칸/내부선호) 파싱은 main/normalize에서 수행한다고 가정
    # 여기서는 docs 메타만 가지고 순수 랭킹을 진행하려면, 질문 파싱 결과를 넣어줘야 함.
    # 호환을 위해 question만 받지만, 실제 점수는 메타 내 필드로 계산.

    # 기본 힌트(없으면 널) – main에서 실제 값으로 교체해 호출하는 것을 권장
    target_levels: List[str] = []           # 변수
    want_exit: Optional[int] = None         # 변수
    want_dir: Optional[str] = None          # 변수
    want_car: Optional[str] = None          # 변수
    needs_surface: bool = False             # 변수
    prefer_internal: bool = False           # 변수

    # 점수 계산
    scored = []
    for m in docs:
        if m.get("equipment_type") not in ("엘리베이터", "휠체어 리프트"):
            continue
        sc = _score_item(m, target_levels, want_exit, want_dir, want_car, needs_surface, prefer_internal)
        scored.append((sc, m))

    # 정렬 + 타이브레이커
    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [m for (_, m) in scored]

    # 스니펫/소스/메타 구성
    compressed_text = _make_snippets(ranked, topk=SNIPPET_TOPK)  # 변수
    sources = _build_sources(ranked, topk=SNIPPET_TOPK)          # 변수
    meta = {
        "hits": len(ranked),                 # 변수
        "model": "qwen2.5:0.5b-instruct",    # 변수
        "scales": {
            "exit": W_EXIT, "surface": W_SURF, "status_ok": W_STAT_OK,
            "status_stop": W_STAT_STOP, "dir": W_DIR, "target": W_TGT
        }
    }  # 변수
    return {"compressed_text": compressed_text, "sources": sources, "meta": meta}

# =========================
# 호환: 기존 rank_items (정렬 리스트만 반환)
# =========================
def score_item(
    meta: Dict[str, Any],
    target_levels: List[str],
    want_exit: Optional[int],
    want_dir: Optional[str],
    want_car: Optional[str],
    needs_surface: bool,
    prefer_internal: bool
) -> float:
    """(호환) 외부에서 직접 점수 계산이 필요할 때 사용"""
    return _score_item(meta, target_levels, want_exit, want_dir, want_car, needs_surface, prefer_internal)

def rank_items(
    metas: List[Dict[str, Any]],
    target_levels: List[str],
    want_exit: Optional[int],
    want_dir: Optional[str],
    want_car: Optional[str],
    needs_surface: bool,
    prefer_internal: bool
) -> List[Dict[str, Any]]:
    """(호환) 메타만 정렬해 리스트로 반환"""
    candidates = []
    for m in metas:
        if m.get("equipment_type") not in ["엘리베이터", "휠체어 리프트"]:
            continue
        if target_levels and not _covers_target(m, target_levels):
            continue
        sc = _score_item(m, target_levels, want_exit, want_dir, want_car, needs_surface, prefer_internal)
        candidates.append((sc, m))
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 동점 처리(안정성): 같은 점수 묶음 내에서 EV 우선
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(candidates):
        j = i
        same: List[Dict[str, Any]] = []
        while j < len(candidates) and candidates[j][0] == candidates[i][0]:
            same.append(candidates[j][1])
            j += 1
        # 타이브레이커 적용
        same.sort(key=lambda m: (m.get("equipment_type") != "엘리베이터",
                                 m.get("use_status_std") != "정상운행",
                                 not bool(m.get("is_surface_link"))))
        out.extend(same)
        i = j
    return out
