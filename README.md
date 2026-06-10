# 心理咨询 CTE 深度学习项目

这套工程面向心理咨询音频标注、结构化数据构建、片段级训练和整段级预测。

## 流程

1. 用 ELAN 标注 `.eaf`
2. 用 `psych_cte.eaf_to_json` 导出结构化 JSON / CSV
3. 用 `psych_cte.train_segment` 训练片段级多任务模型
4. 用 `psych_cte.train_audio` 训练整段级 CTE 模型
5. 用 `psych_cte.predict_segment` / `psych_cte.predict_audio` 做预测

## 模型设计

- 片段级：`BERT/中文预训练模型 + 多任务头`
  - 局部 CTE 评分回归
  - 五个共情反馈维度多标签分类
  - 阻碍类型多分类
- 整段级：`BERT 编码每个共情单元 + 注意力池化 + 回归头`
  - 学习整场咨询中更重要的关键片段

## 依赖

- Python 3.8+
- `torch`
- `transformers`
- `sentencepiece`

建议先安装你机器对应的 `torch` 版本，再安装其余依赖。

## 安装

```bash
pip install -r requirements.txt
```

## ELAN 导出

```bash
python -m psych_cte.eaf_to_json input.eaf --output output.json --csv output.csv
```

如果你的 ELAN 轨道名称不同，可以改成：

```bash
python -m psych_cte.eaf_to_json input.eaf --client-tier 来访者 --therapist-tier 咨询师 --unit-tier 共情单元 --output output.json
```

## 片段级训练

```bash
python -m psych_cte.train_segment --data output.json --output segment_cte.pt
```

## 整段级训练

```bash
python -m psych_cte.train_audio --data audio_level.json --output audio_cte.pt
```

`audio_level.json` 需要保留每个音频下的 `segments` 列表，并包含整段 `audio_cte_score`。

## 预测

先导出说话人样本，手动确认来访者/咨询师：

```bash
python -m psych_cte.export_speaker_samples --audio 220228.wav --backend pyannote --pyannote-model pyannote/speaker-diarization-3.1 --output-dir outputs/speaker_samples --num-speakers 2
```

生成后试听：

- `outputs/speaker_samples/220228/speaker_0_sample.wav`
- `outputs/speaker_samples/220228/speaker_1_sample.wav`

如果使用 pyannote，需要先设置 Hugging Face token：

```bash
set HF_TOKEN=你的token
```

另外你还需要在 Hugging Face 页面手动接受模型条款：

- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

另外 pyannote 这条链路还会用到 `flair`，如果环境里缺它，先装：

```bash
pip install flair
```

在你当前的 Python 3.7 环境里，`numba` 建议装：

```bash
pip install numba==0.56.4
```

如果你当前其实是 Python 3.11，就改装：

```bash
pip install numba==0.57.0
```

片段级：

```bash
python -m psych_cte.predict_segment --checkpoint segment_cte.pt --client-text "我最近很焦虑" --therapist-text "听起来你压力很大"
```

整段级：

```bash
python -m psych_cte.predict_audio --checkpoint audio_cte.pt --input-json audio_level.json
```

直接从音频预测：

```bash
python -m psych_cte.predict_from_audio --audio 220228.wav --segment-checkpoint checkpoints/segment_cte.pt --output-json outputs/220228_prediction.json --output-csv outputs/220228_prediction.csv
```

如果已经导出了说话人样本，可以用同一份 diarization 结果并手动指定角色：

```bash
python -m psych_cte.predict_from_audio --audio 220228.wav --segment-checkpoint checkpoints/segment_cte.pt --diarization-json outputs/speaker_samples/220228/speaker_samples_manifest.json --client-speaker speaker_0 --therapist-speaker speaker_1
```

默认逻辑是“第一个检测到的说话人=来访者，第二个检测到的说话人=咨询师”。

## openSMILE 咨询师声学特征导出

这个模块只做独立的声学统计，不参与前面的训练和预测。

```bash
python -m psych_cte.extract_therapist_acoustics --audio 220228.wav --diarization-json outputs/speaker_samples/220228/speaker_samples_manifest.json --output-csv outputs/220228_therapist_acoustics.csv
```

如果你已经在别的地方确定了咨询师说话人标签，也可以显式指定：

```bash
python -m psych_cte.extract_therapist_acoustics --audio 220228.wav --diarization-json outputs/speaker_samples/220228/speaker_samples_manifest.json --therapist-speaker speaker_1 --output-csv outputs/220228_therapist_acoustics.csv
```

输出 CSV 里会包含平均音高、音高标准差、平均音强、音强标准差，以及一些辅助字段。

## 说明

- 这套实现先把“文本共情判断”作为核心深度学习任务。
- 声学特征分析可以后续独立加一个音频特征模块，再和文本模型结果做融合展示。
- 你现在这台环境是 Python 3.7，实际跑训练建议先升级到 3.8 或更高。

## openSMILE + 语速合并导出

这个命令会先提取咨询师的平均音高、音高标准差、平均音强、音强标准差，再根据咨询师说话区间内的转写文本计算语速，最后统一输出到一个文件夹里。

```bash
python -m psych_cte.extract_therapist_acoustics --audio 220228.wav --diarization-json outputs/speaker_samples/220228/speaker_samples_manifest.json --therapist-speaker speaker_0
```

默认会生成两个文件：
- `outputs/therapist_acoustic_features/220228/220228_therapist_acoustics.csv`
- `outputs/therapist_acoustic_features/220228/220228_therapist_text_segments.csv`

语速默认按“字数 / 有效说话时长”计算；如果你更想按词数算，可以加 `--speech-rate-unit word`。

如果你已经有现成的转写结果，也可以直接传入 `--turns-csv`，这样就不会重新跑 FunASR。
