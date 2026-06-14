const state = {
  models: [],
  jobs: [],
  activeJobId: "",
  pollTimer: null,
  diarizationPollTimer: null,
  generatedDiarizationId: "",
  hfTokenConfigured: false,
};

const $ = (id) => document.getElementById(id);

function setText(id, text) {
  $(id).textContent = text;
}

function api(path, options = {}) {
  return fetch(path, options).then(async (response) => {
    if (!response.ok) {
      let message = response.statusText;
      try {
        const body = await response.json();
        message = body.detail || body.message || message;
      } catch (_) {
        message = await response.text();
      }
      throw new Error(message);
    }
    return response.json();
  });
}

function formatTime(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString();
}

function modelLabel(model) {
  const type = model.type === "segment" ? "片段" : model.type === "audio" ? "整段" : "未知";
  const source = model.source === "uploaded" ? "导入" : "内置";
  const encoder = model.encoder ? ` · ${model.encoder}` : "";
  return `[${type}/${source}] ${model.name} · ${model.size_mb} MB${encoder}`;
}

function fillModelSelects() {
  const segmentSelect = $("segmentModel");
  const audioSelect = $("audioModel");
  segmentSelect.innerHTML = "";
  audioSelect.innerHTML = '<option value="">不使用整段模型</option>';

  const segmentModels = state.models.filter((item) => item.type === "segment");
  const audioModels = state.models.filter((item) => item.type === "audio");
  const unknownModels = state.models.filter((item) => item.type === "unknown");

  [...segmentModels, ...unknownModels].forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = modelLabel(model);
    segmentSelect.appendChild(option);
  });

  audioModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = modelLabel(model);
    audioSelect.appendChild(option);
  });

  setText("modelCount", `${state.models.length} 个模型`);
}

function renderDownloads(job) {
  const box = $("downloadLinks");
  box.innerHTML = "";
  const labels = {
    json: "JSON",
    csv: "片段 CSV",
    turns_csv: "对话 CSV",
    acoustics_csv: "声学 CSV",
    acoustics_text_csv: "声学文本 CSV",
    log: "日志",
  };

  Object.entries(job.downloads || {}).forEach(([key, href]) => {
    const link = document.createElement("a");
    link.href = href;
    link.textContent = labels[key] || key;
    link.target = "_blank";
    box.appendChild(link);
  });
}

function applyAsrDefaults(defaults) {
  if (!defaults) return;
  if (defaults.asr_model) $("asrModel").value = defaults.asr_model;
  if (defaults.vad_model) $("vadModel").value = defaults.vad_model;
  if (defaults.punc_model) $("puncModel").value = defaults.punc_model;
  if (defaults.spk_model) $("spkModel").value = defaults.spk_model;
}

function renderSpeakerPreview(speakers) {
  const list = $("speakerPreviewList");
  list.innerHTML = "";
  const data = speakers || [];
  if (!data.length) {
    list.innerHTML = "";
    return;
  }

  data.forEach((item) => {
    const card = document.createElement("div");
    card.className = "speaker-card";
    const match = String(item.speaker || "").match(/^speaker_(\d+)$/);
    const displayName = match ? `Speaker ${Number(match[1]) + 1}（${item.speaker}）` : item.speaker;

    const title = document.createElement("div");
    title.className = "speaker-card-title";
    const name = document.createElement("strong");
    name.textContent = displayName;
    const meta = document.createElement("span");
    meta.textContent = `${item.segment_count || 0} 段 · ${item.duration_s || 0} 秒`;
    title.append(name, meta);

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = item.sample_url;

    const actions = document.createElement("div");
    actions.className = "speaker-card-actions";
    const clientButton = document.createElement("button");
    clientButton.type = "button";
    clientButton.textContent = "设为来访者";
    clientButton.addEventListener("click", () => {
      $("clientSpeaker").value = item.speaker;
    });
    const therapistButton = document.createElement("button");
    therapistButton.type = "button";
    therapistButton.textContent = "设为咨询师";
    therapistButton.addEventListener("click", () => {
      $("therapistSpeaker").value = item.speaker;
    });
    actions.append(clientButton, therapistButton);

    card.append(title, audio, actions);
    list.appendChild(card);
  });
}

async function submitSpeakerPreview() {
  const audioFiles = $("audioInput").files;
  const diarizationFiles = $("diarizationInput").files;
  if (!audioFiles.length || !diarizationFiles.length) {
    setText("speakerPreviewStatus", "请先选择音频文件和说话人分离 JSON");
    return;
  }

  const data = new FormData();
  data.append("audio", audioFiles[0]);
  data.append("diarization_json", diarizationFiles[0]);
  $("speakerPreviewButton").disabled = true;
  setText("speakerPreviewStatus", "正在生成试听样本...");
  renderSpeakerPreview([]);

  try {
    const result = await api("/api/speaker-preview", {
      method: "POST",
      body: data,
    });
    renderSpeakerPreview(result.speakers || []);
    setText("speakerPreviewStatus", `已生成 ${result.speakers?.length || 0} 个 speaker 样本`);
  } catch (error) {
    setText("speakerPreviewStatus", `生成失败：${error.message}`);
  } finally {
    $("speakerPreviewButton").disabled = false;
  }
}

function renderDiarizationDownloads(result) {
  const box = $("diarizationDownloads");
  box.innerHTML = "";
  [
    ["说话人 JSON", result.manifest_url],
    ["RTTM", result.rttm_url],
    ["日志", result.log_url],
  ].forEach(([label, href]) => {
    if (!href) return;
    const link = document.createElement("a");
    link.href = href;
    link.target = "_blank";
    link.textContent = label;
    box.appendChild(link);
  });
}

async function submitDiarization() {
  const audioFiles = $("audioInput").files;
  if (!audioFiles.length) {
    setText("diarizationStatus", "请先选择音频文件");
    return;
  }

  const data = new FormData();
  data.append("audio", audioFiles[0]);
  data.append("backend", $("diarizationBackend").value);
  data.append("device", $("diarizationDevice").value);
  data.append("num_speakers", $("diarizationSpeakerCount").value || "2");
  data.append("hf_token", $("diarizationToken").value || "");
  if (
    ["pyannote", "whisperx"].includes($("diarizationBackend").value) &&
    !$("diarizationToken").value.trim() &&
    !state.hfTokenConfigured
  ) {
    const message = "pyannote/whisperx 需要 HF Token；请填写 HF Token，或先设置后端环境变量 HF_TOKEN。";
    setText("diarizationStatus", `生成失败：${message}`);
    $("failureReason").textContent = message;
    $("failureReason").classList.add("show");
    $("jobLog").textContent = message;
    $("logMeta").textContent = "1 行";
    return;
  }

  $("diarizationButton").disabled = true;
  state.generatedDiarizationId = "";
  renderSpeakerPreview([]);
  renderDiarizationDownloads({});
  setText("diarizationStatus", "正在生成说话人分离 JSON，音频较长时会花几分钟...");

  try {
    const result = await api("/api/diarization", {
      method: "POST",
      body: data,
    });
    state.generatedDiarizationId = result.manifest_id || "";
    renderDiarizationDownloads(result);
    renderSpeakerPreview(result.speakers || []);
    setText("diarizationStatus", `已生成 JSON，并生成 ${result.speakers?.length || 0} 个试听样本`);
    setText("speakerPreviewStatus", "已使用新生成的说话人分离 JSON 生成试听");
  } catch (error) {
    setText("diarizationStatus", `生成失败：${error.message}`);
  } finally {
    $("diarizationButton").disabled = false;
  }
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  const valueNode = document.createElement("strong");
  valueNode.textContent = value === undefined || value === null || value === "" ? "-" : String(value);
  item.append(labelNode, valueNode);
  return item;
}

function firstFilled(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function empathyLabelStatus(summary, key) {
  return summary?.[key]?.status || "";
}

function empathyLevelFromCte(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return "";
  if (value < 2.5) return "较低共情";
  if (value < 3.5) return "中等共情";
  return "较高共情";
}

function overallFeedbackFromCte(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return null;
  if (value < 2.5) {
    return {
      feedback:
        "咨询师的回应可能与来访者的核心情绪或内在意义贴合度不足。这类回应通常未能准确反映来访者当下的主要感受，也较少帮助来访者继续探索自己的情绪体验。回应中可能存在较明显的建议、评价、安慰或转移焦点，使来访者尚未被充分理解时就被带向解决问题或其他方向。",
      suggestion:
        "建议咨询师优先练习识别来访者此刻最明显的情绪，例如委屈、失落、焦虑、羞愧、无助或愤怒。回应时可以先减少建议、解释或评价，尝试使用“你似乎……”“听起来你……”等情感反映句式，让来访者感到自己的感受被听见。下一次练习时，可先用一句话反映来访者的情绪，再补充对其处境或内在体验的理解",
    };
  }
  if (value < 3.5) {
    return {
      feedback:
        "咨询师能够基本跟随来访者的表达，并对其显性情绪或主要事件作出回应。这类回应通常与来访者当前表达大体一致，但对来访者更深层的情绪、需要、期待或内在冲突探索不足。回应能够体现一定理解，但仍有进一步深化和具体化的空间。",
      suggestion:
        "建议在准确复述来访者内容的基础上，再向前推进一步。不仅说出来访者“发生了什么”和“感到什么”，还可以尝试理解“为什么这件事对他如此重要”。下一次回应可以从情绪背后的需要、期待、自我评价、人际关系感受等方面进行深化，并适当使用开放式问题促进来访者继续表达",
    };
  }
  return {
    feedback:
      "咨询师的回应较准确地贴近了来访者的核心情绪，并进一步触及其情绪背后的内在意义。这类回应不仅能够反映来访者已经表达出的感受，还能够帮助来访者进一步觉察尚未完全说清的体验、需要或冲突。回应整体较具体、贴近且具有促进探索的作用。",
    suggestion:
      "建议继续保持这种先理解、再探索的回应方式。高共情回应的关键不是使用复杂语言，而是准确、具体、贴近来访者的内在体验。后续可以继续关注来访者对该回应的反应，例如是否进一步表达、是否情绪变得更清晰、是否出现新的自我理解。同时，可继续练习在稳定表达共情的基础上，进一步促进来访者探索情绪背后的意义和需要。",
  };
}

function labelAdviceItems(labelSummary) {
  const items = [];
  const emotionReflected = Boolean(labelSummary?.emotion_reflection?.present);
  items.push({
    title: "内容及情感反映",
    text: emotionReflected
      ? "已体现。你的回应能够抓住来访者表达中的主要内容或情绪，使来访者较容易感到自己被听见"
      : "未明显体现。你的回应对来访者核心情绪或主要体验的反映不足，建议下一次先识别来访者最突出的感受，并用简洁语言回应出来。",
  });

  const acceptanceReflected = Boolean(labelSummary?.acceptance_confirmation?.present);
  items.push({
    title: "接纳确认",
    text: acceptanceReflected
      ? "已体现。你的回应能够在一定程度上表达对来访者感受的理解和接纳，有助于降低来访者的防御感。"
      : "未明显体现。你的回应较少体现对来访者感受的确认和接纳，建议下一次加入类似“在这样的情况下你会这样感受是可以理解的”等验证性表达。",
  });

  const explorationReflected = Boolean(labelSummary?.exploration_facilitation?.present);
  items.push({
    title: "促进探索",
    text: explorationReflected
      ? "已体现。你的回应能够帮助来访者进一步停留在自己的体验中，或引导其继续表达尚未完全说清的感受和意义。"
      : "未明显体现。你的回应主要停留在已有内容上，较少推动来访者进一步探索。建议下一次在反映情绪后，使用开放式问题，如“这对你来说最难的部分是什么？”“当时你心里最强烈的感受是什么？”。",
  });

  const blockingReflected = Boolean(labelSummary?.blocking_present?.present);
  items.push({
    title: "共情阻碍",
    text: blockingReflected
      ? "存在明显阻碍。你的回应中可能出现了过早建议、评价、泛泛安慰或转移焦点等表达，使来访者的情绪尚未被充分理解时就被带向解决问题。建议下一次先停留在理解和反映阶段，再考虑是否需要提出建议。"
      : "未见明显阻碍。你的回应中暂未发现明显削弱共情传达的表达方式。",
  });
  return items;
}

function renderOverallFeedback(score, labelSummary) {
  const box = $("overallFeedback");
  if (!box) return;
  const feedback = overallFeedbackFromCte(score);
  box.innerHTML = "";
  if (!feedback) {
    box.hidden = true;
    return;
  }

  const feedbackBlock = document.createElement("div");
  const feedbackTitle = document.createElement("h3");
  const feedbackText = document.createElement("p");
  feedbackTitle.textContent = "总体反馈";
  feedbackText.textContent = feedback.feedback;
  feedbackBlock.append(feedbackTitle, feedbackText);

  const suggestionBlock = document.createElement("div");
  const suggestionTitle = document.createElement("h3");
  const suggestionText = document.createElement("p");
  suggestionTitle.textContent = "建议";
  suggestionText.textContent = feedback.suggestion;
  suggestionBlock.append(suggestionTitle, suggestionText);

  const labelAdviceBlock = document.createElement("div");
  const labelAdviceTitle = document.createElement("h3");
  labelAdviceTitle.textContent = "标签建议";
  labelAdviceBlock.appendChild(labelAdviceTitle);
  labelAdviceItems(labelSummary).forEach((item) => {
    const itemTitle = document.createElement("h4");
    const itemText = document.createElement("p");
    itemTitle.textContent = item.title;
    itemText.textContent = item.text;
    labelAdviceBlock.append(itemTitle, itemText);
  });

  box.append(feedbackBlock, suggestionBlock);
  box.appendChild(labelAdviceBlock);
  box.hidden = false;
}

function renderSummary(preview) {
  const grid = $("summaryGrid");
  grid.innerHTML = "";
  const summary = preview?.summary || {};
  const analysis = summary.analysis_summary || {};
  const audioPrediction = summary.audio_prediction || {};
  const audioSummary = summary.audio_summary || {};
  const roleMap = summary.role_map || {};
  const labelSummary = audioSummary.empathy_label_summary || analysis.empathy_label_summary || {};
  const audioCteScore = firstFilled(audioPrediction.audio_cte_score, audioSummary.audio_cte_score, analysis.audio_cte_score);

  grid.append(
    metric("音频编号", summary.audio_id),
    metric("对话轮次", analysis.turn_count),
    metric("共情单元", analysis.unit_count),
    metric("来访者轮次", analysis.client_turn_count),
    metric("咨询师轮次", analysis.therapist_turn_count),
    metric("整段 CTE", audioCteScore),
    metric("共情等级", empathyLevelFromCte(audioCteScore)),
    metric("来访者映射", Object.entries(roleMap).find(([, value]) => value === "client")?.[0] || ""),
    metric("咨询师映射", Object.entries(roleMap).find(([, value]) => value === "therapist")?.[0] || ""),
    metric("内容及情感反映", empathyLabelStatus(labelSummary, "emotion_reflection")),
    metric("深层意义理解", empathyLabelStatus(labelSummary, "deep_meaning_understanding")),
    metric("接纳确认", empathyLabelStatus(labelSummary, "acceptance_confirmation")),
    metric("促进探索", empathyLabelStatus(labelSummary, "exploration_facilitation")),
    metric("共情阻碍", empathyLabelStatus(labelSummary, "blocking_present"))
  );

  setText("summaryText", summary.audio_id ? "已生成" : "未生成");
  renderOverallFeedback(audioCteScore, labelSummary);
}

function renderTable(rows) {
  const table = $("segmentTable");
  if (!table) return;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  const data = rows || [];
  setText("rowCount", `${data.length} 行`);
  if (!data.length) return;

  const preferred = [
    "segment_id",
    "共情单元起止时间",
    "来访文本",
    "咨询师文本",
    "预测CTE分数",
    "预测阻碍类型",
    "预测共情标签",
  ];
  const keys = preferred.filter((key) => key in data[0]);
  Object.keys(data[0]).forEach((key) => {
    if (!keys.includes(key) && keys.length < 9) keys.push(key);
  });

  const headerRow = document.createElement("tr");
  keys.forEach((key) => {
    const th = document.createElement("th");
    th.textContent = key;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  data.forEach((row) => {
    const tr = document.createElement("tr");
    keys.forEach((key) => {
      const td = document.createElement("td");
      td.textContent = row[key] || "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderAcousticTable(rows) {
  const table = $("acousticTable");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  const empty = $("acousticEmpty");
  const wrap = table.parentElement;
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const data = rows || [];
  setText("acousticRowCount", `${data.length} 行`);
  empty.style.display = data.length ? "none" : "block";
  wrap.style.display = data.length ? "block" : "none";
  if (!data.length) return;

  const preferred = [
    "audio_id",
    "therapist_speaker",
    "therapist_segment_count",
    "therapist_speech_duration_s",
    "speech_rate_unit",
    "speech_rate_per_minute",
    "speech_rate_per_second",
    "speech_rate_text_turn_count",
    "pitch_mean_hz",
    "pitch_std_hz",
    "loudness_mean",
    "loudness_std",
    "pitch_frame_count",
    "loudness_frame_count",
  ];
  const hidden = new Set(["audio_file", "therapist_text", "source_diarization_json", "source_turns_csv"]);
  const keys = preferred.filter((key) => key in data[0]);
  Object.keys(data[0]).forEach((key) => {
    if (!keys.includes(key) && !hidden.has(key) && keys.length < 16) keys.push(key);
  });
  const labels = {
    audio_id: "音频编号",
    therapist_speaker: "咨询师说话人",
    therapist_segment_count: "咨询师片段数",
    therapist_speech_duration_s: "咨询师发言总时长（秒）",
    speech_rate_unit: "语速统计单位",
    speech_unit_count: "语速单位总数",
    speech_rate_per_minute: "平均语速（字/分钟）",
    speech_rate_per_second: "平均语速（字/秒）",
    speech_rate_text_turn_count: "参与语速统计轮次",
    pitch_mean_hz: "平均音高（Hz）",
    pitch_std_hz: "音高标准差（Hz）",
    loudness_mean: "平均响度",
    loudness_std: "响度标准差",
    pitch_frame_count: "音高有效帧数",
    loudness_frame_count: "响度有效帧数",
  };

  const headerRow = document.createElement("tr");
  keys.forEach((key) => {
    const th = document.createElement("th");
    th.textContent = labels[key] || key;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  data.forEach((row) => {
    const tr = document.createElement("tr");
    keys.forEach((key) => {
      const td = document.createElement("td");
      td.textContent = row[key] || "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderLog(log) {
  const box = $("jobLog");
  const meta = $("logMeta");
  const failureBox = $("failureReason");
  if (!log) {
    box.textContent = "等待任务启动后显示实时日志。";
    meta.textContent = "0 行";
    failureBox.classList.remove("show");
    failureBox.textContent = "";
    return;
  }

  const parts = [];
  if (log.status) parts.push(`状态: ${log.status}`);
  if (log.message) parts.push(`消息: ${log.message}`);
  if (log.status === "failed" && log.failure_reason) parts.push(`失败原因: ${log.failure_reason}`);
  if (log.tail_text) parts.push("", log.tail_text.trimEnd());

  box.textContent = parts.join("\n").trim() || "暂无日志";
  box.scrollTop = box.scrollHeight;
  meta.textContent = `${log.line_count || 0} 行`;

  if (log.status === "failed" && log.failure_reason) {
    failureBox.textContent = log.failure_reason;
    failureBox.classList.add("show");
  } else {
    failureBox.classList.remove("show");
    failureBox.textContent = "";
  }
}

async function loadJobLog(jobId) {
  const log = await api(`/api/jobs/${jobId}/log?tail=240`);
  renderLog(log);
}

async function loadModels() {
  const data = await api("/api/models");
  state.models = data.models || [];
  fillModelSelects();
}

async function loadJobs() {
  const data = await api("/api/jobs");
  state.jobs = data.jobs || [];
  renderJobs();
}

function renderJob(job) {
  if (!job) return;
  state.activeJobId = job.id;
  setText("jobStatus", job.status || "unknown");
  $("jobStatus").className = `status-${job.status || "unknown"}`;
  setText("jobMessage", job.message || "");
  renderDownloads(job);
  renderSummary(job.preview || {});
  renderAcousticTable(job.preview?.acoustics || []);
  if (job.status === "failed" && job.failure_reason) {
    const failureBox = $("failureReason");
    failureBox.textContent = job.failure_reason;
    failureBox.classList.add("show");
  }
}

function renderJobs() {
  const list = $("jobList");
  list.innerHTML = "";
  setText("jobCount", `${state.jobs.length} 个`);

  state.jobs.slice(0, 8).forEach((job) => {
    const item = document.createElement("div");
    item.className = "job-item";

    const main = document.createElement("div");
    main.className = "job-main";
    const title = document.createElement("strong");
    title.textContent = job.input?.audio_filename || job.id;
    const sub = document.createElement("span");
    sub.textContent = `${job.status} · ${formatTime(job.created_at)}`;
    main.append(title, sub);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "查看";
    button.addEventListener("click", async () => {
      const fresh = await api(`/api/jobs/${job.id}`);
      renderJob(fresh);
      await loadJobLog(job.id);
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "danger";
    deleteButton.textContent = "删除";
    deleteButton.disabled = job.status === "running" || job.status === "queued";
    deleteButton.addEventListener("click", async () => {
      if (!confirm("确定删除这个历史任务吗？")) return;
      deleteButton.disabled = true;
      try {
        await api(`/api/jobs/${job.id}`, { method: "DELETE" });
        if (state.activeJobId === job.id) {
          state.activeJobId = "";
          setText("jobStatus", "暂无任务");
          setText("jobMessage", "等待提交");
          renderDownloads({});
          renderSummary({});
          renderLog(null);
          renderAcousticTable([]);
        }
        await loadJobs();
      } catch (error) {
        setText("jobMessage", `删除失败：${error.message}`);
        deleteButton.disabled = false;
      }
    });

    const actions = document.createElement("div");
    actions.className = "job-actions";
    actions.append(button, deleteButton);

    item.append(main, actions);
    list.appendChild(item);
  });
}

async function refreshAll() {
  try {
    const health = await api("/api/health");
    state.hfTokenConfigured = Boolean(health.hf_token_configured);
    setText(
      "healthText",
      health.ok
        ? `后端已连接 · 运行环境 ${health.runner_python || "未知"}`
        : "后端异常"
    );
    applyAsrDefaults(health.asr_defaults);
    await Promise.all([loadModels(), loadJobs()]);
  } catch (error) {
    setText("healthText", `连接失败：${error.message}`);
  }
}

function startPolling(jobId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      renderJob(job);
      await loadJobLog(jobId);
      if (job.status === "completed" || job.status === "failed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        $("submitButton").disabled = false;
      }
      await loadJobs();
    } catch (error) {
      setText("jobMessage", error.message);
    }
  }, 2000);
}

async function submitPrediction(event) {
  event.preventDefault();
  const form = $("predictForm");
  const data = new FormData(form);
  if (!$("diarizationInput").files.length) {
    data.delete("diarization_json");
    if (state.generatedDiarizationId) {
      data.set("diarization_file_id", state.generatedDiarizationId);
    }
  }
  data.set("run_acoustics", $("runAcoustics").checked ? "true" : "false");

  $("submitButton").disabled = true;
  setText("jobStatus", "queued");
  setText("jobMessage", "正在提交任务");
  $("failureReason").classList.remove("show");
  $("failureReason").textContent = "";
  $("jobLog").textContent = "任务已提交，等待日志刷新...";
  $("logMeta").textContent = "0 行";

  try {
    const job = await api("/api/predict", {
      method: "POST",
      body: data,
    });
    renderJob(job);
    await loadJobs();
    await loadJobLog(job.id);
    startPolling(job.id);
  } catch (error) {
    $("submitButton").disabled = false;
    setText("jobStatus", "failed");
    $("jobStatus").className = "status-failed";
    setText("jobMessage", `提交失败：${error.message}`);
    $("failureReason").textContent = error.message;
    $("failureReason").classList.add("show");
  }
}

async function submitModel(event) {
  event.preventDefault();
  const data = new FormData($("modelUploadForm"));
  try {
    setText("modelCount", "正在导入");
    await api("/api/models/upload", {
      method: "POST",
      body: data,
    });
    $("modelUploadForm").reset();
    await loadModels();
  } catch (error) {
    setText("modelCount", `导入失败：${error.message}`);
  }
}

async function loadDiarizationLog(previewId) {
  const log = await api(`/api/diarization/${previewId}/log?tail=240`);
  renderLog(log);
}

function renderDiarizationTask(task) {
  if (!task) return;
  if (task.status === "completed") {
    state.generatedDiarizationId = task.manifest_id || "";
    renderDiarizationDownloads(task);
    renderSpeakerPreview(task.speakers || []);
    setText("diarizationStatus", `已生成 JSON，并生成 ${task.speakers?.length || 0} 个试听样本`);
    setText("speakerPreviewStatus", "已使用新生成的说话人分离 JSON 生成试听");
    $("diarizationButton").disabled = false;
  } else if (task.status === "failed") {
    const reason = task.failure_reason || task.message || "未知错误";
    setText("diarizationStatus", `生成失败：${reason}`);
    $("failureReason").textContent = reason;
    $("failureReason").classList.add("show");
    $("diarizationButton").disabled = false;
  } else {
    setText("diarizationStatus", task.message || "正在生成说话人分离 JSON...");
  }
}

function startDiarizationPolling(previewId) {
  if (state.diarizationPollTimer) clearInterval(state.diarizationPollTimer);
  state.diarizationPollTimer = setInterval(async () => {
    try {
      const task = await api(`/api/diarization/${previewId}`);
      renderDiarizationTask(task);
      await loadDiarizationLog(previewId);
      if (task.status === "completed" || task.status === "failed") {
        clearInterval(state.diarizationPollTimer);
        state.diarizationPollTimer = null;
      }
    } catch (error) {
      setText("diarizationStatus", `日志刷新失败：${error.message}`);
    }
  }, 2000);
}

async function submitDiarization() {
  const audioFiles = $("audioInput").files;
  if (!audioFiles.length) {
    setText("diarizationStatus", "请先选择音频文件");
    return;
  }

  const data = new FormData();
  data.append("audio", audioFiles[0]);
  data.append("backend", $("diarizationBackend").value);
  data.append("device", $("diarizationDevice").value);
  data.append("num_speakers", $("diarizationSpeakerCount").value || "2");
  data.append("hf_token", $("diarizationToken").value || "");
  if (
    ["pyannote", "whisperx"].includes($("diarizationBackend").value) &&
    !$("diarizationToken").value.trim() &&
    !state.hfTokenConfigured
  ) {
    const message = "pyannote/whisperx 需要 HF Token；请填写 HF Token，或先设置后端环境变量 HF_TOKEN。";
    setText("diarizationStatus", `生成失败：${message}`);
    $("failureReason").textContent = message;
    $("failureReason").classList.add("show");
    $("jobLog").textContent = message;
    $("logMeta").textContent = "1 行";
    return;
  }

  if (state.diarizationPollTimer) clearInterval(state.diarizationPollTimer);
  $("diarizationButton").disabled = true;
  state.generatedDiarizationId = "";
  renderSpeakerPreview([]);
  renderDiarizationDownloads({});
  $("failureReason").classList.remove("show");
  $("failureReason").textContent = "";
  $("jobLog").textContent = "说话人分离任务已提交，等待实时日志刷新...";
  $("logMeta").textContent = "0 行";
  setText("diarizationStatus", "正在提交说话人分离 JSON 生成任务...");

  try {
    const task = await api("/api/diarization", {
      method: "POST",
      body: data,
    });
    renderDiarizationTask(task);
    await loadDiarizationLog(task.id);
    startDiarizationPolling(task.id);
  } catch (error) {
    setText("diarizationStatus", `生成失败：${error.message}`);
    $("failureReason").textContent = error.message;
    $("failureReason").classList.add("show");
    $("diarizationButton").disabled = false;
  }
}

function hasDiarizationManifest(task) {
  return Boolean(task && (task.manifest_id || task.manifest_url || task.downloads?.manifest));
}

function renderDiarizationTask(task) {
  if (!task) return;
  if (task.status === "completed" && hasDiarizationManifest(task)) {
    state.generatedDiarizationId = task.manifest_id || "";
    renderDiarizationDownloads(task);
    renderSpeakerPreview(task.speakers || []);
    setText("diarizationStatus", `已生成 JSON，并生成 ${task.speakers?.length || 0} 个试听样本`);
    setText("speakerPreviewStatus", "已使用新生成的说话人分离 JSON 生成试听");
    $("diarizationButton").disabled = false;
    return;
  }
  if (task.status === "completed" && !hasDiarizationManifest(task)) {
    const reason = "后端返回完成状态，但未返回说话人分离 JSON 文件。请查看实时日志。";
    setText("diarizationStatus", `生成异常：${reason}`);
    $("failureReason").textContent = reason;
    $("failureReason").classList.add("show");
    $("diarizationButton").disabled = false;
    return;
  }
  if (task.status === "failed") {
    const reason = task.failure_reason || task.message || "未知错误";
    setText("diarizationStatus", `生成失败：${reason}`);
    $("failureReason").textContent = reason;
    $("failureReason").classList.add("show");
    $("diarizationButton").disabled = false;
    return;
  }
  setText("diarizationStatus", task.message || "正在生成说话人分离 JSON...");
}

$("refreshButton").addEventListener("click", refreshAll);
$("diarizationButton").addEventListener("click", submitDiarization);
$("speakerPreviewButton").addEventListener("click", submitSpeakerPreview);
$("predictForm").addEventListener("submit", submitPrediction);
$("modelUploadForm").addEventListener("submit", submitModel);

refreshAll();
