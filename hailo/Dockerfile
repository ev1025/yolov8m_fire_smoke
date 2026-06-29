# =============================================================================
# Hailo-8L 컴파일 환경 (ONNX -> HEF). DFC + Model Zoo 격리 컨테이너.
#
# 빌드 컨텍스트(이 폴더)에 둘 파일:
#   hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl  (DFC)
#   hailort-4.24.0-cp310-cp310-linux_x86_64.whl               (HailoRT Python 바인딩)
#   hailort_4.24.0_amd64.deb                                  (HailoRT 네이티브, libhailort.so)
#
# ⚠ DFC 는 반드시 venv(site-packages)에 설치할 것.
#   시스템 dist-packages 에 깔면 DFC 의 is_release 판정("site-packages" 문자열 검사)이
#   깨져 컴파일러 tools 경로가 None 이 되고, 최종 컴파일이
#   "TypeError: expected str ... not NoneType" 로 실패함.
# ⚠ Windows 는 Docker Desktop(WSL2)에서 빌드/실행. model_zoo v2.19.0 = DFC 3.34.0 매칭.
# =============================================================================
FROM ubuntu:22.04

ARG PY=3.10
ARG MZ_TAG=v2.19.0
ENV DEBIAN_FRONTEND=noninteractive
ENV USER=root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PY} python${PY}-dev python${PY}-venv python3-pip \
        git build-essential unzip graphviz graphviz-dev python3-tk \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# venv 필수: site-packages 경로여야 DFC is_release=True -> 컴파일러 바이너리 경로 정상
RUN python${PY} -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip "setuptools>=70" wheel

WORKDIR /workspace

# 1) HailoRT 네이티브 런타임 (.deb) -> /usr/lib/libhailort.so (시스템, ldconfig 등록)
COPY hailort*amd64*.deb /tmp/
RUN dpkg -i /tmp/hailort*amd64*.deb || (apt-get update && apt-get install -f -y && rm -rf /var/lib/apt/lists/*)
RUN ldconfig

# 2) DFC + HailoRT Python 바인딩 (venv = site-packages)
COPY *.whl /tmp/
RUN pip install /tmp/hailo_dataflow_compiler-*.whl
RUN if ls /tmp/hailort-*.whl >/dev/null 2>&1; then pip install /tmp/hailort-*.whl; fi

# 3) Hailo Model Zoo (hailo8l 지원 v2.x 태그)
#    --no-build-isolation: 격리 빌드 대신 venv(설치된 DFC) 사용
RUN git clone -b ${MZ_TAG} --depth 1 https://github.com/hailo-ai/hailo_model_zoo.git \
    && cd hailo_model_zoo && pip install -e . --no-build-isolation

# 4) 컴파일 스크립트
COPY compile_to_hef.py /workspace/compile_to_hef.py

# best.onnx, calib_data/ 는 docker run -v 로 주입
CMD ["bash"]
