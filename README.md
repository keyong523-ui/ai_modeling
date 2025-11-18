abstractive_summarization.py   # KoBART 추상 요약 모델  /n
extractive_summarization.py    # KoBERT 추출 요약 모델
train_summarization_full.py    # 요약 모델 학습 스크립트

classfication.py               # 문장 유형 분류 모델(KoELECTRA)

inference.py                   # 단일 기사 요약 + 분류 통합 추론
data_inference.py              # 파일 기반 배치 추론
test.py                        # 테스트용 샘플 실행



1. Summarization Models (요약)
  추출적 요약 (Extractive)
    - 모델: KoBERT
    - 역할: 기사에서 가장 중요한 문장 3개 선택
    - 사용 데이터: AI Hub 문서요약 텍스트(뉴스) → 추출요약 라벨 사용

  추상적 요약 (Abstractive)
    - 모델: KoBART
    - 역할: 전체 기사(또는 핵심문장 기반)를 자연스럽게 요약
    - 사용 데이터: AI Hub 문서요약 텍스트 → 추상요약 라벨 사용

학습 실행
python train_summarization_full.py --task extractive
python train_summarization_full.py --task abstractive
