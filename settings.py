# settings.py
# 프로젝트 전체에서 쓰는 전역 설정 모음
# - pydantic_settings.BaseSettings를 사용해 .env 값 주입 가능
# - "변수(고정)"들은 프로젝트 실행 시 고정적으로 쓰임

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ===== OpenAI LLM =====
    OPENAI_API_KEY: str = ""                                # 변수(고정): .env에서 OPENAI_API_KEY 주입
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"      # 변수(고정)
    GENERATION_MODEL: str = "gpt-3.5-turbo"                 # 변수(고정): 생성 모델명
    HTTP_TIMEOUT_SEC: int = 8                               # 변수(고정): 외부 호출 타임아웃(초)

    # ===== 벡터 DB / 임베딩 =====
    EMBEDDING_MODEL: str = "nomic-embed-text"               # 변수(고정): 임베딩 모델명
    CHROMA_DIR: str = "./.chroma_elevators"                 # 변수(고정): 로컬 벡터DB 경로

    # ===== 서울 열린데이터 =====
    SEOUL_API_BASE_URL: str = "http://openapi.seoul.go.kr:8088"  # 변수(고정): 기본 URL
    SEOUL_API_KEY: str = "REPLACE_WITH_YOUR_KEY"                 # 변수(고정): .env에서 덮어쓰기 권장
    SEOUL_DATASET_ID: str = "SeoulMetroFaciInfo"                 # 변수(고정): 데이터셋 ID

    # ===== 파서 / RAG =====
    USE_LLM_PARSER: bool = True     # 변수(고정): LLM 기반 질문 파서 사용 여부
    RAG_TOP_K: int = 6              # 변수(고정): RAG 검색 상위 문서 개수

    class Config:
        env_file = ".env"           # .env 파일에서 환경변수 읽기

# 전역 설정 인스턴스
settings = Settings()   # 변수(고정)
