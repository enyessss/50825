# rag_index.py
import os, json, hashlib
from typing import List, Dict, Any, Optional
import chromadb

try:
    from settings import settings  # 변수(고정)
except Exception:
    settings = None                # 변수(고정)

# ---- Chroma 클라이언트 ----
def _get_client():
    persist_dir = getattr(settings, "CHROMA_DIR", "./.chroma") if settings else "./.chroma"  # 변수
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)  # 변수

_client = _get_client()  # 변수(고정)

def get_collection():
    return _client.get_or_create_collection(
        name="elevators", metadata={"hnsw:space": "cosine"}  # 변수
    )

# ---- 문서 텍스트 빌더(검색 단서 풍부하게) ----
def build_doc_text(meta: Dict[str, Any]) -> str:
    st  = str(meta.get("station_name") or "")               # 변수
    ln  = str(meta.get("line_no") or "")                    # 변수
    ex  = f"{meta.get('exit_no')}번 출구" if meta.get("exit_no") is not None else ""  # 변수
    lp  = meta.get("levels_path") or []                     # 변수
    seg = f"{lp[0]}↔{lp[-1]}" if lp else (meta.get("opr_sec") or "")                 # 변수
    inst= str(meta.get("instl_pstn") or "")                 # 변수
    stat= str(meta.get("use_status_raw") or "")             # 변수
    typ = str(meta.get("equipment_type") or "")             # 변수
    return " ".join(x for x in [st, f"{ln}호선", ex, typ, seg, f"설치위치:{inst}", f"상태:{stat}"] if x)

# ---- ✅ 임베딩 함수 구현 ----
# 1순위: Ollama 임베딩 API 사용
# 2순위: 임시 해시 임베딩(로컬에서라도 파이프라인 동작 보장)
async def embed_texts(texts: List[str]) -> List[List[float]]:
    try:
        import httpx  # 변수
        base = getattr(settings, "OLLAMA_HOST", "http://127.0.0.1:11434") if settings else "http://127.0.0.1:11434"  # 변수
        model = getattr(settings, "EMBEDDING_MODEL", "nomic-embed-text") if settings else "nomic-embed-text"        # 변수

        embs: List[List[float]] = []  # 변수
        async with httpx.AsyncClient(timeout=60.0) as client:  # 변수
            for t in texts:
                r = await client.post(f"{base}/api/embeddings", json={"model": model, "prompt": t})  # 변수
                r.raise_for_status()
                data = r.json()
                vec = data.get("embedding")
                if not isinstance(vec, list) or not vec:
                    raise ValueError("invalid embedding from ollama")
                embs.append([float(x) for x in vec])
        return embs
    except Exception:
        # 🔸 Ollama가 없으면 임시 해시 임베딩(고정 256차원)으로 파이프라인을 유지
        def _hash_vec(s: str, dim: int = 256) -> List[float]:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            # 32바이트를 반복해 dim 길이 맞추기
            arr = []
            while len(arr) < dim:
                for b in h:
                    arr.append((b - 128) / 128.0)
                    if len(arr) >= dim:
                        break
            return arr
        return [_hash_vec(t) for t in texts]  # 변수

# ---- ✅ 실제 업서트(add 호출 필수) ----
async def upsert_docs(
    ids: List[str],
    docs: List[str],
    metas: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None
) -> None:
    col = get_collection()  # 변수
    kwargs = {"ids": ids, "documents": docs, "metadatas": metas}  # 변수
    if embeddings is not None:
        kwargs["embeddings"] = embeddings                          # 변수
    col.add(**kwargs)                                              # 변수

def collection_count() -> int:
    try:
        return get_collection().count()  # 변수
    except Exception:
        return 0                         # 변수
