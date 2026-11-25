# LiDAR 주행 실험실 (`lidar-avoidance-ml`)

2D 라이다(LiDAR) 센서 기반 장애물 회피를 교사-학생 모방학습(Behavioral Cloning)으로 학습하고 비교하는 Pygame 시뮬레이터입니다. 로보틱스와 머신러닝 학습자가 센서 기반 회피 모델의 학습 과정과 실제 주행 성능 차이를 한 화면에서 실험할 수 있도록 구성했습니다.

전체 지도를 아는 A\* 경로 계획기로 주행 데이터를 수집한 뒤, 지도 없이 라이다 25빔과 목적지 정보 3개(`sin`, `cos`, 정규화 거리)만 입력받는 학습 모델 5종(`Linear`, `KNN`, `RandomForest`, `MLP`, `1D CNN`)과 규칙 기반 기준선(`APF`)을 비교합니다. 저장된 회귀 20맵 평가([results/benchmark_20maps.csv](results/benchmark_20maps.csv))에서는 기준선 도달률 85% 대비 학습 모델이 최대 100%에 도달했습니다.

## 주요 구성

- **1 수집**: A\* 자동 주행 실시간 기록(15 Hz), 3,000샘플 일괄 생성(20 Hz), 키보드 수동 수집, 수집 샘플 비우기, 데이터셋 저장 및 불러오기, 맵 선택(새 랜덤 맵 · A · B · 출발점 복귀), `처음이라면 · 사전 이론` 바로가기를 제공합니다.
- **2 학습**: 회귀(연속 선속도·각속도)와 분류(좌·직·우 3클래스) 전환, 5종 모델 일괄·개별 학습을 지원합니다. `MLP`와 `1D CNN`은 에포크별 손실 곡선을 표시하며 `Linear`·`KNN`·`RandomForest`는 단일 적합 결과로 학습합니다.
- **3 레이스**: 고정 맵 A(`A · 기본 회피`), 고정 맵 B(`B · 복합 난코스`), 랜덤 맵에서 최대 6대(기준선과 현재 과제의 학습 완료 모델) 동시 레이스(18초 제한)와 단일 주행 예측값 확인을 실행합니다. 고정 비교 맵 20개 평가(15초 제한) 결과는 CSV로 저장합니다.
- **4 시각화**: 총 6종 그래프(`3D PCA · 주행 샘플 분포`, `2D 분포와 결정 경계`, `회귀 · 실제값과 예측값`, `분류 · 혼동 행렬과 F1`, `28차원 특징 중요도`, `모델 종합 비교`)를 표시하고 160 DPI PNG로 저장합니다(`현재 그래프 PNG 저장`, `전체 6종 PNG 저장`).
- **5 해석**: [docs/guide.md](docs/guide.md) 본문, 설명용 애니메이션, 최근 20맵 평가 결과를 앱 안에서 바로 읽을 수 있습니다.

## 빠른 실행

Python 3.10 이상과 [`uv`](https://docs.astral.sh/uv/)가 필요합니다.

```bash
git clone https://github.com/Leehyunbin0131/lidar-avoidance-ml.git
cd lidar-avoidance-ml
uv run python main.py
```

- 저장소에는 `data/`와 `models/`가 포함되어 있지 않습니다. 기준선 단독 레이스는 바로 실행할 수 있으며 `1 수집`에서 데이터를 생성하면 3D PCA 같은 데이터 분포 그래프를 확인할 수 있습니다. 학습 모델 5종 비교 레이스와 모델별 분석 그래프를 보려면 `2 학습`에서 모델을 먼저 학습해야 합니다.
- `pyproject.toml`은 기본으로 `pytorch-cpu` 인덱스를 사용하므로 기본 설치된 PyTorch 빌드는 CPU로만 동작합니다(OpenVINO 추론의 Intel GPU·NPU 지원과는 별개입니다). CUDA GPU 학습과 추론을 사용하려면 CUDA 지원 PyTorch 빌드를 별도로 설치해야 합니다.

## 모델 및 입력 구조

- **관측 벡터 (28차원)**: 전방 180° 라이다 거리 25개(정규화), 목적지 상대 방향 `sin`과 `cos` 2개, 정규화 거리 1개로 구성됩니다.
- **학습 모델 (5종)**:
  - `Linear`: `StandardScaler` + `Ridge(alpha=1.0)`(회귀) / `LogisticRegression(max_iter=500)`(분류)
  - `KNN`: `StandardScaler` + `KNeighborsRegressor` / `KNeighborsClassifier`(`n_neighbors=7`, `weights="distance"`, `algorithm="kd_tree"`)
  - `RandomForest`: 결정 트리 60개(`max_depth=16`)
  - `MLP (PyTorch)`: `28 → 96 → 64 → 출력` 완전연결 신경망(`Dropout 0.08`, `Adam lr=1.5e-3`, `batch_size=64`)
  - `1D CNN (PyTorch)`: 라이다 25빔 `Conv1d(1→16, k=5)` → `Conv1d(16→32, k=3)` → `FC 64`와 목적지 3차원 결합
- **기준선 (`BASE`, 라이다 회피 기준선)**: 지도 없이 동일한 28차원 입력으로 목적지 인력과 장애물 척력을 합산하는 무상태 인공 퍼텐셜 필드(APF) 규칙 주행기입니다.
- **하드웨어 가속**:
  - **학습 장치**: `2 학습 → 학습 설정`의 장치 선택은 `MLP`와 `1D CNN`에만 적용됩니다. CUDA 지원 PyTorch 환경에서는 CUDA GPU를 자동 감지하며 기본 `uv` 환경과 `scikit-learn` 3종(`Linear`, `KNN`, `RandomForest`)은 CPU에서 실행됩니다.
  - **추론 백엔드**: `3 레이스` 탭의 추론 장치 선택은 `MLP`와 `1D CNN`에만 적용됩니다. 하드웨어 감지 결과에 따라 `PyTorch CPU`, `CUDA GPU`, `OV CPU`, `OV GPU`(Intel GPU 감지 시), `OV NPU`(Intel AI Boost 감지 시)를 선택할 수 있으며 OpenVINO 선택 시 `models/ov_*.xml`·`.bin` 파일로 자동 변환합니다.

## 기본 사용 흐름

1. **데이터 수집**: `1 수집` 탭에서 **데이터셋 자동 생성 시작** 또는 **샘플 3,000개 일괄 생성**으로 주행 데이터를 생성합니다. 직접 조종 기록을 추가하려면 **수동 주행 데이터 수집 시작**을 사용합니다.
2. **모델 학습**: `2 학습` 탭에서 회귀 또는 분류를 선택하고 **5종 모델 일괄 학습**을 실행합니다. **학습 설정**에서는 `MLP`·`1D CNN`의 에포크(40·60·100)와 학습 장치, 좌우 반전 증강, 분류 전용 **클래스 균형** 옵션을 변경할 수 있습니다.
3. **고정 맵 비교 및 20맵 평가**: `3 레이스` 탭에서 **A · 기본 회피**를 선택해 **동시 레이스 시작**으로 기본 회피 동작을 확인합니다. 맵을 바꾸면 레이스가 정지하므로 **B · 복합 난코스**로 전환한 뒤 **동시 레이스 시작**을 다시 눌러 모델 간 격차를 비교합니다. **20맵 평가 실행**을 수행하면 시드 1000–1019(난이도 12)의 고정 비교 맵 20개를 15초 제한으로 평가하고 참여 주행기별 집계 1행씩을 `results/benchmark_20maps.csv`에 덮어씁니다. 이 파일은 `5 해석` 탭에서도 불러옵니다.
4. **시각화 및 해석**: `4 시각화` 탭에서 데이터 분포와 모델 결과를 확인하고 `results/plots/`에 PNG로 저장합니다. `5 해석` 탭에서는 원리와 평가 결과를 확인합니다.

## 단축키 및 조작

- **공통**: `1`–`5` 탭 전환
- **1 수집 · 2 학습 · 3 레이스**: `N` 새 랜덤 맵 생성, `R` 출발 위치 복귀, 지도 좌클릭/우클릭 출발 위치·목적지 이동(장애물 위 클릭 제외)
- **1 수집**: `W A S D` / 방향키 수동 조종, `Space` 수동 수집 시작·중지
- **3 레이스**: `Space` 주행 시작·중지
- **4 시각화**: 3D PCA 그래프 좌클릭 드래그 회전
- **5 해석**: `↑`/`↓`·`PageUp`/`PageDown`·`Home`/`End`·마우스 휠·스크롤바 드래그 스크롤, `←`/`→` 이전·다음 장 이동 (`N`·`R` 및 지도 클릭 비활성)

## 관련 문서

- [docs/guide.md](docs/guide.md): 시뮬레이터 원리, 모델별 특징, 평가 결과 해석
- [docs/prompt_log.md](docs/prompt_log.md): 프로젝트 개발 및 검증 과정의 생성 AI 프롬프트 기록
- 참고 문헌: Ross et al. (2011), [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)
