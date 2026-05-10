# Week 10: vLLM Deployment Practice

## 과제 개요

이 과제의 목표는 DGX H100 서버에서 `DeepSeek-Coder-V2-Lite-Instruct` 계열 모델을 vLLM으로 OpenAI 호환 API 서버 형태로 배포하고, API 테스트, 벤치마크, 로컬 에이전트 연동 예제를 준비하는 것이다.

## DGX 접속 전제

- DGX H100 서버에 SSH로 접속할 수 있어야 한다.
- CUDA가 정상 설치된 Linux 환경이어야 한다.
- Hugging Face 모델 접근 권한이 있어야 한다.
- `HF_TOKEN`은 파일에 쓰지 말고 환경변수로만 사용해야 한다.

## 가상환경 생성 명령

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
python -m pip install --upgrade pip
```

## torch CUDA 12.1 설치 명령

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## vLLM 설치 명령

```bash
pip install vllm openai huggingface_hub
```

## Hugging Face 로그인 및 모델 다운로드 명령

```bash
export HF_TOKEN="<set-in-shell-only>"
huggingface-cli login --token "$HF_TOKEN"
mkdir -p "$HOME/models"
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --local-dir "$HOME/models/deepseek-coder-v2-lite"
```

## start_server.sh 실행 방법

```bash
bash start_server.sh
```

스크립트는 다음 옵션으로 vLLM OpenAI API 서버를 실행한다.

- model path: `$HOME/models/deepseek-coder-v2-lite`
- port: `8000`
- tensor parallel size: `2`
- dtype: `bfloat16`
- GPU memory utilization: `0.90`
- max model length: `32768`

로그는 `vllm_server.log`에 저장된다.

## 서버 health check 방법

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

## test_api.py 실행 방법

```bash
python test_api.py
```

이 스크립트는 OpenAI SDK로 로컬 vLLM 서버에 연결해 Python 퀵소트 구현 요청을 보내고, 응답 텍스트와 usage 정보를 출력한다.

## benchmark.py 실행 방법

```bash
python benchmark.py
```

이 스크립트는 `AsyncOpenAI`로 concurrency `1`, `4`, `8`을 각각 실행하고 결과를 콘솔과 `benchmark_results.json`에 기록한다.

## benchmark 지표 설명

- `Throughput`: 일정 시간 동안 처리한 토큰 수다. 여기서는 tokens per second 기준으로 계산한다.
- `TTFT`: Time To First Token. 요청 시작부터 첫 토큰이 도착할 때까지의 시간이다.
- `TBT`: Time Between Tokens. 토큰 간 간격을 의미한다. 이 과제의 샘플 벤치마크 코드는 TTFT와 전체 TPS 중심으로 구성했고, 실제 DGX 측정 시 TBT를 추가 확장할 수 있다.
- `success rate`: 전체 요청 중 성공한 요청의 비율이다.

## benchmark_results.json 설명

현재 저장된 `benchmark_results.json`은 샘플 placeholder다.  
실제 DGX 서버에서 vLLM을 실행한 뒤 다시 벤치마크를 돌려 실제 수치로 갱신해야 한다.

포함 구조:

- concurrency `1`, `4`, `8`
- `avg_ttft_ms`
- `p50_ttft_ms`
- `p99_ttft_ms`
- `avg_throughput_tps`
- `total_throughput_tps`
- `success_rate`

## local_agent.py 설명

`local_agent.py`는 OpenAI 호환 API를 쓰는 간단한 로컬 에이전트 예제다.  
`ask_coder(task: str)` 함수는 system prompt를 `"You are an expert Python developer."`로 고정하고, 로컬 vLLM 모델에게 코딩 요청을 보낸다.

## Claude API 대비 로컬 vLLM의 장점/단점

장점:

- 데이터 통제가 쉽다.
- 반복 호출 비용을 장기적으로 줄일 수 있다.
- 외부 SaaS 없이 로컬 또는 사내 환경에서 서빙 가능하다.

단점:

- 고성능 GPU가 필요하다.
- 초기 설정이 복잡하다.
- 인프라 운영 비용과 장애 대응 책임이 직접 생긴다.

## 제출 전 실제 DGX에서 갱신해야 할 항목

- `benchmark_results.json`
- `vllm_server.log`

현재 두 파일은 예시/샘플 내용이며, 실제 DGX 실행 후 결과로 교체해야 한다.

