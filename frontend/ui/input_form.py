"""입력 폼 화면 + 순수 로직(태스크 추정, 입력 구성)."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import pandas as pd
import streamlit as st

from backend.interface import DataCard, PipelineInput, PipelineBackend
from ui import theme

LOOP_MIN, LOOP_MAX, LOOP_DEFAULT = 1, 10, 3


def infer_task_type(series: pd.Series) -> str:
    """타깃 컬럼 dtype/고유값으로 분류/회귀 추정."""
    if series.dtype == object or str(series.dtype) == "category":
        return "classification"
    nunique = series.nunique(dropna=True)
    # 고유값이 적으면 분류로 간주
    if nunique <= 10:
        return "classification"
    return "regression"


def build_pipeline_input(csv_path: str, loop_count: int, target_column: str,
                         task_type: str, description: str,
                         llm_instruction: str) -> PipelineInput:
    """폼 값들을 PipelineInput으로 조립."""
    card = DataCard(target_column=target_column, task_type=task_type,
                    description=description)
    return PipelineInput(csv_path=csv_path, loop_count=loop_count,
                         data_card=card, llm_instruction=llm_instruction)


def _save_upload(uploaded) -> str:
    """업로드 파일을 임시 경로에 저장하고 경로 반환."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def _get_csv(uploaded):
    """업로드 파일을 1회만 저장·파싱하고 (path, df) 반환.

    Streamlit은 위젯 상호작용마다 스크립트를 재실행하므로, 캐시 없이 매번
    저장하면 임시파일이 누적되고 CSV를 반복 파싱한다. 업로드 식별자로 캐시한다.
    """
    file_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    cache = st.session_state.get("_csv_cache")
    if cache and cache["file_id"] == file_id:
        return cache["path"], cache["df"]
    if cache:  # 다른 파일로 바뀜 → 이전 임시파일 정리(누수 방지)
        try:
            os.remove(cache["path"])
        except OSError:
            pass
    path = _save_upload(uploaded)
    df = pd.read_csv(path)
    st.session_state["_csv_cache"] = {"file_id": file_id, "path": path, "df": df}
    return path, df


def render(backend: PipelineBackend) -> Optional[PipelineInput]:
    """입력 폼 렌더링. '실행'이 눌리고 검증 통과 시 PipelineInput 반환."""
    theme.step_header(0)
    st.subheader("1. 데이터와 분석 의도를 입력하세요")

    uploaded = st.file_uploader("CSV 업로드", type=["csv"])
    if uploaded is None:
        st.info("CSV 파일을 업로드하면 데이터 카드 입력이 활성화됩니다.")
        return None

    csv_path, df = _get_csv(uploaded)
    st.caption(f"행 {df.shape[0]} · 열 {df.shape[1]}")
    st.dataframe(df.head(), width="stretch")

    columns = list(df.columns)
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("타깃 컬럼(예측 대상)", columns,
                                     index=len(columns) - 1)
    with col2:
        default_task = infer_task_type(df[target_column])
        task_type = st.radio("태스크 종류", ["classification", "regression"],
                             index=0 if default_task == "classification" else 1,
                             horizontal=True)

    description = st.text_area("데이터 카드 — 데이터셋 설명", height=80,
                               placeholder="예) 타이타닉 승객 정보. 생존 여부 예측.")
    llm_instruction = st.text_area(
        "언어모델 입력 — 가설 · 목표 · 평가산식", height=120,
        placeholder="예) 객실 등급(Pclass)이 생존에 미치는 영향 가설을 반영해 "
                    "정확도 기준으로 모델을 개선해줘.",
    )
    loop_count = st.number_input("루프 횟수", min_value=LOOP_MIN, max_value=LOOP_MAX,
                                 value=LOOP_DEFAULT, step=1)

    if st.button("분석 실행 ▶", type="primary"):
        inp = build_pipeline_input(
            csv_path=csv_path, loop_count=int(loop_count),
            target_column=target_column, task_type=task_type,
            description=description, llm_instruction=llm_instruction,
        )
        errors = backend.validate_input(inp)
        if errors:
            for e in errors:
                st.error(e)
            return None
        return inp
    return None
