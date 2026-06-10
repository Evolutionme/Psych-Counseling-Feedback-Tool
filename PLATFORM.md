# 心理咨询 CTE 前后端平台

## 启动

```bash
pip install -r requirements.txt
python -m uvicorn psych_cte.api:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 已接入功能

- 上传音频并创建预测任务
- 自动调用 `psych_cte.predict_from_audio`
- 选择 `checkpoints/` 里的片段模型和整段模型
- 导入新的 `.pt/.pth/.bin` 训练模型
- 可选上传说话人分离 JSON，并指定来访者/咨询师 speaker
- 可选导出咨询师声学特征
- 查看任务状态、片段预测预览、JSON/CSV/日志下载

## 平台文件位置

- 上传音频：`outputs/platform/uploads/audio/`
- 上传模型：`outputs/platform/uploads/models/`
- 任务输出：`outputs/platform/jobs/<job_id>/`

