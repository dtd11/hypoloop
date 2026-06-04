"""입력 검증 공용 로직 (모든 PipelineBackend 구현이 공유)."""
from __future__ import annotations

from typing import List

import pandas as pd

from backend.interface import PipelineInput, TASK_TYPES


def validate_input(inp: PipelineInput) -> List[str]:
    """PipelineInput 검증. 문제 메시지 리스트 반환(빈 리스트면 통과)."""
    errors: List[str] = []
    try:
        columns = list(pd.read_csv(inp.csv_path, nrows=1).columns)
    except Exception as exc:  # noqa: BLE001 - 사용자에게 그대로 안내
        return [f"CSV를 읽을 수 없습니다: {exc}"]

    if inp.data_card.target_column not in columns:
        errors.append(
            f"타깃 컬럼 '{inp.data_card.target_column}'이(가) CSV에 없습니다."
        )
    if inp.data_card.task_type not in TASK_TYPES:
        errors.append(f"태스크 종류가 올바르지 않습니다: {inp.data_card.task_type}")
    if inp.loop_count < 1:
        errors.append("루프 횟수는 1 이상이어야 합니다.")
    if not inp.llm_instruction.strip():
        errors.append("언어모델 입력(가설/목표)을 작성해주세요.")
    return errors
