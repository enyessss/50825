# ictchallenge
### 파일구조 
fastapi_project/
```bash
├── app/
│   ├── init.py
│   └── main.py # FastAPI 실행
├── api/
│   ├── init.py
│   └── routes_elevator.py # 승강기 검색 API
├── services/
│   ├── init.py
│   ├── elevator_service.py # 공공데이터 API 호출 + 결과 반환
│   └── rag_service.py # ChromaDB + LLM 기반 검색 보조
├── models/
│   ├── init.py
│   └── elevator.py # 요청/응답 데이터 모델
└── requirements.txt
└── .env
``` 
