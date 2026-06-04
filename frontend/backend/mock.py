"""백엔드 미완성 동안 UI를 완성·테스트하기 위한 가짜 구현."""
from __future__ import annotations

import time
from typing import Iterator, List, Optional

from backend import validation
from backend.interface import (
    PipelineInput, ProgressEvent, MetricRecord, PipelineResult,
    PIPELINE_STAGES,
)

# 진행 이벤트 사이 지연(초). 테스트 빠르게 하려고 짧게.
_STEP_DELAY = 0.05


class MockBackend:
    """PipelineBackend 계약을 따르는 가짜 백엔드."""

    def __init__(self, step_delay: float = _STEP_DELAY) -> None:
        self._step_delay = step_delay
        self._result: Optional[PipelineResult] = None

    def validate_input(self, inp: PipelineInput) -> List[str]:
        return validation.validate_input(inp)

    def run(self, inp: PipelineInput) -> Iterator[ProgressEvent]:
        is_clf = inp.data_card.task_type == "classification"
        metric_name = "accuracy" if is_clf else "rmse"
        baseline = 0.70 if is_clf else 0.50
        metrics: List[MetricRecord] = []

        # 베이스라인까지 단계 진행(loop 0)
        for stage in PIPELINE_STAGES[:3]:  # 계획수립, EDA, 베이스라인
            yield ProgressEvent(stage=stage, loop_index=0, status="running",
                                message=f"{stage} 진행 중...")
            time.sleep(self._step_delay)
            yield ProgressEvent(stage=stage, loop_index=0, status="done",
                                message=f"{stage} 완료")
        metrics.append(MetricRecord(loop_index=0, metric_name=metric_name,
                                    baseline=baseline, value=baseline))

        # 개선 루프
        for loop in range(1, inp.loop_count + 1):
            for stage in PIPELINE_STAGES[3:5]:  # 피처엔지니어링, 튜닝
                yield ProgressEvent(stage=stage, loop_index=loop, status="running",
                                    message=f"루프 {loop}: {stage} 진행 중...")
                time.sleep(self._step_delay)
                yield ProgressEvent(stage=stage, loop_index=loop, status="done",
                                    message=f"루프 {loop}: {stage} 완료")
            # 매 루프 가짜 개선(분류는 증가, 회귀는 감소)
            if is_clf:
                value = min(baseline + 0.04 * loop, 0.95)
            else:
                value = max(baseline - 0.03 * loop, 0.10)
            metrics.append(MetricRecord(loop_index=loop, metric_name=metric_name,
                                        baseline=baseline, value=round(value, 4)))

        # 리포트 단계
        yield ProgressEvent(stage="리포트", loop_index=inp.loop_count,
                            status="running", message="리포트 생성 중...")
        time.sleep(self._step_delay)

        self._result = self._build_result(inp, metrics, metric_name)
        yield ProgressEvent(stage="리포트", loop_index=inp.loop_count,
                            status="done", message="완료")

    def get_result(self) -> PipelineResult:
        if self._result is None:
            raise RuntimeError("run()을 먼저 실행해야 결과를 얻을 수 있습니다.")
        return self._result

    def _build_result(self, inp: PipelineInput, metrics: List[MetricRecord],
                      metric_name: str) -> PipelineResult:
        base = metrics[0].value
        best = metrics[-1].value
        delta = round(best - base, 4)
        report_md = (
            f"# 분석 리포트 (Mock)\n\n"
            f"- 데이터: `{inp.csv_path}`\n"
            f"- 타깃: **{inp.data_card.target_column}** "
            f"({inp.data_card.task_type})\n"
            f"- 루프 횟수: {inp.loop_count}\n\n"
            f"## 성능 요약\n\n"
            f"| 구분 | {metric_name} |\n|---|---|\n"
            f"| 베이스라인 | {base} |\n| 최종 | {best} |\n"
            f"| 향상폭(Δ) | {delta} |\n\n"
            f"## 사용자 가설\n\n> {inp.llm_instruction}\n\n"
            f"> ⚠️ 제한된 데이터·환경에서 도출된 결과이므로, "
            f"프로덕션 적용 전 반드시 검토가 필요합니다.\n"
        )
        estimator = (
            "RandomForestClassifier"
            if inp.data_card.task_type == "classification"
            else "RandomForestRegressor"
        )
        final_code = (
            "import pandas as pd\n"
            f"from sklearn.ensemble import {estimator}\n\n"
            f"df = pd.read_csv(r'{inp.csv_path}')\n"
            f"y = df['{inp.data_card.target_column}']\n"
            f"X = df.drop(columns=['{inp.data_card.target_column}'])\n"
            f"model = {estimator}().fit(X, y)\n"
            "print('done')\n"
        )
        experiment_yaml = (
            f"task_type: {inp.data_card.task_type}\n"
            f"target: {inp.data_card.target_column}\n"
            f"metric: {metric_name}\n"
            f"loop_count: {inp.loop_count}\n"
        )
        return PipelineResult(report_md=report_md, final_code=final_code,
                              metrics_history=metrics,
                              experiment_yaml=experiment_yaml)
