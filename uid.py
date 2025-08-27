# uid.py
# 공개 함수: make_uid(meta)
# - 주어진 설비 메타에서 안정적인 고유 ID를 생성
# - 같은 설비(row)는 항상 같은 ID로 매핑되도록 설계

import hashlib
from typing import Dict

def make_uid(meta: Dict) -> str:
    """
    ✅ 공개 함수: 설비 메타 → 고유 ID
    - 역코드, 설비구분, 설비명, 운행구간, 설치위치 기반
    - 필요시 station_name까지 포함 가능
    """
    key = "|".join([
        str(meta.get("stn_cd") or ""),        # 변수
        str(meta.get("station_name") or ""), # 변수: 안정성 위해 추가
        str(meta.get("elvtr_se") or ""),     # 변수
        str(meta.get("elvtr_nm") or ""),     # 변수
        str(meta.get("opr_sec") or ""),      # 변수
        str(meta.get("instl_pstn") or "")    # 변수
    ])
    # SHA1으로 해시 → 40자 고정 ID
    return hashlib.sha1(key.encode("utf-8")).hexdigest()  # 변수
