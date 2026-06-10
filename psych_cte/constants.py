"""Shared label vocabularies and field aliases."""

EMPATHY_LABELS = [
    "emotion_reflection",
    "deep_meaning_understanding",
    "acceptance_confirmation",
    "exploration_facilitation",
    "blocking_present",
]

BLOCKING_TYPES = [
    "none",
    "premature_advice",
    "judgment_blame",
    "minimization",
    "topic_shift",
    "vague_response",
    "other",
]

FIELD_ALIASES = {
    "audio_id": ["audio_id", "音频编号", "audio_no"],
    "segment_id": ["segment_id", "共情单元编号", "unit_id"],
    "start_ms": ["start_ms", "共情单元起始时间", "segment_start_ms"],
    "end_ms": ["end_ms", "共情单元结束时间", "segment_end_ms"],
    "client_text": ["client_text", "来访者文本", "来访者发言文本"],
    "therapist_text": ["therapist_text", "咨询师文本", "咨询师发言文本"],
    "local_cte_score": ["local_cte_score", "局部CTE评分", "局部 CTE 评分", "单元CTE分数"],
    "local_cte_rationale": ["local_cte_rationale", "评分依据", "局部评分依据", "单元CTE分数依据"],
    "audio_cte_score": ["audio_cte_score", "整体CTE分数", "整段CTE评分", "CTE总评分"],
    "audio_cte_rationale": ["audio_cte_rationale", "整体评分依据", "整段评分依据", "CTE总评分依据"],
    "blocking_type": ["blocking_type", "阻碍类型", "阻碍类型标签"],
}
