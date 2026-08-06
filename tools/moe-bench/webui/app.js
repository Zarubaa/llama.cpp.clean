(function () {
  "use strict";

  var API_ROOT = "/api";
  var TERMINAL_STATES = new Set(["completed", "failed", "cancelled"]);
  var ACTIVE_STATES = new Set(["queued", "running"]);
  var MAX_LOG_LINES = 10000;
  var RUNS_REFRESH_MS = 5000;
  var DETAIL_POLL_MS = 1500;
  var SSE_RECOVERY_BASE_MS = 1000;
  var SSE_RECOVERY_MAX_MS = 15000;

  var labels = {
    loading: "\u52a0\u8f7d\u4e2d",
    online: "\u670d\u52a1\u5728\u7ebf",
    offline: "\u670d\u52a1\u4e0d\u53ef\u7528",
    empty: "\u672a\u9009\u62e9",
    queued: "\u6392\u961f\u4e2d",
    running: "\u8fd0\u884c\u4e2d",
    completed: "\u5df2\u5b8c\u6210",
    failed: "\u5931\u8d25",
    cancelled: "\u5df2\u53d6\u6d88",
    unknown: "\u672a\u77e5",
    following: "\u8ddf\u968f\u4e2d",
    paused: "\u5df2\u6682\u505c",
    noOutput: "\u6682\u65e0\u8f93\u51fa",
    noData: "\u6682\u65e0\u6570\u636e",
    polling: "SSE \u65ad\u5f00\uff0c\u8f6e\u8be2\u4e2d",
    streaming: "SSE \u5b9e\u65f6",
    connecting: "SSE \u8fde\u63a5\u4e2d"
  };

  var state = {
    config: null,
    runs: [],
    activeRunId: null,
    activeRun: null,
    eventSource: null,
    streamGeneration: 0,
    reconnectTimer: null,
    reconnectAttempt: 0,
    reconnectLogBaseline: null,
    detailPollTimer: null,
    detailPollGeneration: 0,
    detailPollInFlight: false,
    runsRefreshTimer: null,
    elapsedTimer: null,
    loadingRuns: false,
    submitting: false,
    cancelling: false,
    logs: {
      stdout: [],
      stderr: []
    },
    logsInitialized: {
      stdout: false,
      stderr: false
    },
    follow: {
      stdout: true,
      stderr: true
    },
    logSequences: new Set(),
    tokenCache: new Map()
  };

  function byId(id) {
    return document.getElementById(id);
  }

  var elements = {};

  function cacheElements() {
    var ids = [
      "service-state", "service-state-label", "service-detail", "refresh-button",
      "run-form", "defaults-button", "form-error", "submit-button", "cancel-button",
      "prompt-text", "prompt-text-field", "default-prompt-note", "file-prompt-note",
      "pp", "tg", "repeat", "ubatch", "context-size", "gpu-layers", "cache-vram-mb",
      "predictor", "host-cache", "host-cache-preload", "page-cache-policy",
      "reset-cache", "warm-cache", "hot-start", "runtime-paths",
      "current-run-id", "current-run-status", "stream-status", "queue-position",
      "elapsed-time", "started-time", "run-empty-state", "run-content",
      "metric-ttft", "metric-prefill", "metric-tpot", "metric-decode", "metric-total",
      "metric-prefill-hit", "metric-decode-hit", "metric-physical", "validation-banner",
      "stdout-log", "stderr-log", "run-command", "raw-summary", "io-details",
      "token-trace", "artifact-list", "history-count", "history-loading",
      "history-empty", "history-error", "history-table-wrap", "history-body", "toast-region"
    ];
    ids.forEach(function (id) {
      elements[id] = byId(id);
    });
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function firstDefined() {
    for (var i = 0; i < arguments.length; i += 1) {
      if (arguments[i] !== undefined && arguments[i] !== null) {
        return arguments[i];
      }
    }
    return null;
  }

  async function requestJson(path, options) {
    var response;
    try {
      response = await fetch(API_ROOT + path, Object.assign({
        headers: { "Accept": "application/json" }
      }, options || {}));
    } catch (error) {
      throw new Error("\u65e0\u6cd5\u8fde\u63a5 bench \u670d\u52a1");
    }

    var body = null;
    var contentType = response.headers.get("content-type") || "";
    if (contentType.indexOf("application/json") !== -1) {
      try {
        body = await response.json();
      } catch (error) {
        body = null;
      }
    } else {
      var text = await response.text();
      body = text ? { error: text } : null;
    }

    if (!response.ok) {
      var message = body && (body.error || body.message || body.detail);
      var requestError = new Error(message || ("HTTP " + response.status));
      requestError.status = response.status;
      requestError.validationErrors = body && body.validation_errors;
      throw requestError;
    }
    return body || {};
  }

  function unwrapConfig(payload) {
    return isObject(payload) && isObject(payload.config) ? payload.config : (payload || {});
  }

  function unwrapRun(payload) {
    if (!payload) {
      return null;
    }
    return isObject(payload.run) ? payload.run : payload;
  }

  function unwrapRuns(payload) {
    if (Array.isArray(payload)) {
      return payload;
    }
    return payload && Array.isArray(payload.runs) ? payload.runs : [];
  }

  function normalizeStatus(status) {
    var value = String(status || "unknown").toLowerCase();
    if (value === "canceled") {
      return "cancelled";
    }
    return ["queued", "running", "completed", "failed", "cancelled"].indexOf(value) !== -1
      ? value
      : "unknown";
  }

  function isActive(run) {
    return !!run && ACTIVE_STATES.has(normalizeStatus(run.status));
  }

  function isTerminal(run) {
    return !!run && TERMINAL_STATES.has(normalizeStatus(run.status));
  }

  function mergeRun(previous, incoming) {
    if (!previous || previous.id !== incoming.id) {
      return incoming;
    }
    var merged = Object.assign({}, previous, incoming);
    if (previous.request || incoming.request) {
      merged.request = Object.assign({}, previous.request || {}, incoming.request || {});
    }
    if (previous.metrics || incoming.metrics) {
      merged.metrics = Object.assign({}, previous.metrics || {}, incoming.metrics || {});
    }
    return merged;
  }

  function setServiceState(status, detail) {
    elements["service-state"].dataset.state = status;
    elements["service-state-label"].textContent = labels[status] || status;
    elements["service-detail"].textContent = detail || "";
    elements["service-detail"].title = detail || "";
  }

  function serviceDescription(config) {
    var service = config.service || {};
    var address = "";
    if (service.host || service.port) {
      address = [service.host, service.port].filter(function (part) {
        return part !== undefined && part !== null && part !== "";
      }).join(":");
    }
    return [service.version, address].filter(Boolean).join(" | ");
  }

  function populateSelect(select, options, fallback) {
    var values = Array.isArray(options) && options.length ? options : fallback;
    var previous = select.value;
    select.replaceChildren();
    values.forEach(function (optionValue) {
      var value = isObject(optionValue) ? optionValue.value : optionValue;
      var label = isObject(optionValue) ? (optionValue.label || optionValue.value) : optionValue;
      var option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(label);
      select.appendChild(option);
    });
    if (values.some(function (item) {
      return String(isObject(item) ? item.value : item) === previous;
    })) {
      select.value = previous;
    }
  }

  function setNumericLimits(input, limits) {
    if (!isObject(limits)) {
      return;
    }
    if (limits.min !== undefined && limits.min !== null) {
      input.min = String(limits.min);
    }
    if (limits.max !== undefined && limits.max !== null) {
      input.max = String(limits.max);
    }
  }

  function renderRuntimePaths(paths) {
    elements["runtime-paths"].replaceChildren();
    var pathLabels = {
      bench: "bench",
      model: "model",
      prompt_file: "prompt file",
      output_root: "output root"
    };
    Object.keys(pathLabels).forEach(function (key) {
      if (!paths || !paths[key]) {
        return;
      }
      var row = document.createElement("div");
      var term = document.createElement("dt");
      var detail = document.createElement("dd");
      term.textContent = pathLabels[key];
      detail.textContent = paths[key];
      detail.title = paths[key];
      row.append(term, detail);
      elements["runtime-paths"].appendChild(row);
    });

    if (!elements["runtime-paths"].children.length) {
      var empty = document.createElement("div");
      empty.textContent = labels.noData;
      elements["runtime-paths"].appendChild(empty);
    }
  }

  function configureForm(config, applyDefaults) {
    var options = config.options || {};
    populateSelect(elements.predictor, options.predictor, ["lru", "eamc"]);
    populateSelect(elements["host-cache"], options.host_cache, ["off", "pageable", "pinned"]);
    populateSelect(elements["host-cache-preload"], options.host_cache_preload, ["none", "all"]);
    populateSelect(elements["page-cache-policy"], options.page_cache_policy, ["natural", "cold", "hot"]);

    var numericIds = {
      pp: "pp",
      tg: "tg",
      repeat: "repeat",
      ubatch: "ubatch",
      cache_vram_mb: "cache-vram-mb",
      context_size: "context-size",
      gpu_layers: "gpu-layers"
    };
    Object.keys(numericIds).forEach(function (key) {
      setNumericLimits(elements[numericIds[key]], config.limits && config.limits[key]);
    });

    renderRuntimePaths(config.paths || {});
    var promptPath = config.paths && config.paths.prompt_file;
    elements["file-prompt-note"].textContent = promptPath
      ? "\u4f7f\u7528\u670d\u52a1\u6587\u4ef6: " + promptPath
      : "\u670d\u52a1\u672a\u914d\u7f6e prompt file";

    if (applyDefaults) {
      applyFormValues(config.defaults || {});
    } else {
      updateFormDependencies();
    }
  }

  function setRadioValue(name, value) {
    var input = elements["run-form"].querySelector('input[name="' + name + '"][value="' + String(value) + '"]');
    if (input) {
      input.checked = true;
    }
  }

  function setInputValue(id, value) {
    if (value !== undefined && value !== null) {
      elements[id].value = String(value);
    }
  }

  function applyFormValues(values) {
    if (!values) {
      return;
    }
    setRadioValue("prompt_mode", firstDefined(values.prompt_mode, "default"));
    if (values.prompt_text !== undefined && values.prompt_text !== null) {
      elements["prompt-text"].value = String(values.prompt_text);
    }
    setInputValue("pp", values.pp);
    setInputValue("tg", values.tg);
    setInputValue("repeat", values.repeat);
    setInputValue("ubatch", values.ubatch);
    setInputValue("cache-vram-mb", values.cache_vram_mb);
    setInputValue("predictor", values.predictor);
    setInputValue("host-cache", values.host_cache);
    setInputValue("host-cache-preload", values.host_cache_preload);
    setInputValue("page-cache-policy", values.page_cache_policy);
    setInputValue("context-size", values.context_size);
    setInputValue("gpu-layers", values.gpu_layers);
    elements["reset-cache"].checked = Boolean(values.reset_cache_between_repeats);
    elements["warm-cache"].checked = Boolean(values.warm_cache);
    elements["hot-start"].checked = Boolean(values.hot_start);
    updatePromptMode();
    updateFormDependencies();
  }

  function selectedPromptMode() {
    var selected = elements["run-form"].querySelector('input[name="prompt_mode"]:checked');
    return selected ? selected.value : "default";
  }

  function updatePromptMode() {
    var mode = selectedPromptMode();
    elements["default-prompt-note"].classList.toggle("is-hidden", mode !== "default");
    elements["file-prompt-note"].classList.toggle("is-hidden", mode !== "file");
    elements["prompt-text-field"].classList.toggle("is-hidden", mode !== "text");
    elements["prompt-text"].required = mode === "text";
  }

  function updateFormDependencies() {
    var locked = elements["run-form"].dataset.locked === "true";
    var hostCacheOff = elements["host-cache"].value === "off";
    if (hostCacheOff) {
      elements["host-cache-preload"].value = "none";
    }
    elements["host-cache-preload"].disabled = locked || hostCacheOff;

    var supportsHotStart = elements.predictor.value === "eamc";
    if (!supportsHotStart) {
      elements["hot-start"].checked = false;
    }
    elements["hot-start"].disabled = locked || !supportsHotStart;
  }

  function setFormLocked(locked) {
    elements["run-form"].dataset.locked = String(locked);
    elements["run-form"].querySelectorAll("input, select, textarea").forEach(function (control) {
      control.disabled = locked;
    });
    elements["defaults-button"].disabled = locked;
    elements["submit-button"].disabled = locked || state.submitting;
    updateFormDependencies();
  }

  function parseInteger(id) {
    return Number.parseInt(elements[id].value, 10);
  }

  function formPayload() {
    var payload = {
      prompt_mode: selectedPromptMode(),
      pp: parseInteger("pp"),
      tg: parseInteger("tg"),
      repeat: parseInteger("repeat"),
      ubatch: parseInteger("ubatch"),
      cache_vram_mb: parseInteger("cache-vram-mb"),
      predictor: elements.predictor.value,
      host_cache: elements["host-cache"].value,
      host_cache_preload: elements["host-cache-preload"].value,
      page_cache_policy: elements["page-cache-policy"].value,
      context_size: parseInteger("context-size"),
      gpu_layers: parseInteger("gpu-layers"),
      reset_cache_between_repeats: elements["reset-cache"].checked,
      warm_cache: elements["warm-cache"].checked,
      hot_start: elements["hot-start"].checked
    };
    if (payload.prompt_mode === "text") {
      payload.prompt_text = elements["prompt-text"].value;
    }
    return payload;
  }

  function setFormError(message, validationErrors) {
    var parts = [];
    if (message) {
      parts.push(message);
    }
    if (Array.isArray(validationErrors)) {
      validationErrors.forEach(function (error) {
        parts.push(typeof error === "string" ? error : (error.message || JSON.stringify(error)));
      });
    }
    elements["form-error"].textContent = parts.join("\n");
    elements["form-error"].classList.toggle("is-hidden", parts.length === 0);
  }

  function validatePayload(payload) {
    if (!elements["run-form"].checkValidity()) {
      elements["run-form"].reportValidity();
      return "\u8bf7\u586b\u5199\u6240\u6709\u5fc5\u586b\u5b57\u6bb5\uff0c\u5e76\u68c0\u67e5\u53d6\u503c\u8303\u56f4\u3002";
    }
    if (payload.prompt_mode === "text" && !payload.prompt_text.trim()) {
      return "\u6587\u672c\u6a21\u5f0f\u4e0b\u63d0\u793a\u8bcd\u4e0d\u80fd\u4e3a\u7a7a\u3002";
    }
    if (payload.context_size < payload.pp + payload.tg) {
      return "Context size \u4e0d\u80fd\u5c0f\u4e8e PP + TG (" + (payload.pp + payload.tg) + ")\u3002";
    }
    return "";
  }

  async function loadConfig(forceDefaults) {
    var firstLoad = state.config === null;
    try {
      var payload = await requestJson("/config");
      state.config = unwrapConfig(payload);
      configureForm(state.config, firstLoad || forceDefaults);
      setServiceState("online", serviceDescription(state.config));
      return state.config;
    } catch (error) {
      setServiceState("offline", error.message);
      throw error;
    }
  }

  function valueOrDash(value, formatter) {
    return value === undefined || value === null || value === "" ? "-" : formatter(value);
  }

  function formatNumber(value, maximumFractionDigits) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return String(value);
    }
    return numeric.toLocaleString(undefined, {
      maximumFractionDigits: maximumFractionDigits === undefined ? 2 : maximumFractionDigits
    });
  }

  function formatMilliseconds(value) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return String(value);
    }
    if (numeric < 1000) {
      return formatNumber(numeric, numeric < 10 ? 2 : 1) + " ms";
    }
    if (numeric < 60000) {
      return formatNumber(numeric / 1000, 2) + " s";
    }
    var minutes = Math.floor(numeric / 60000);
    var seconds = (numeric % 60000) / 1000;
    return minutes + "m " + formatNumber(seconds, 1) + "s";
  }

  function formatBytes(value) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return String(value);
    }
    if (numeric === 0) {
      return "0 B";
    }
    var units = ["B", "KiB", "MiB", "GiB", "TiB"];
    var index = Math.min(Math.floor(Math.log(Math.abs(numeric)) / Math.log(1024)), units.length - 1);
    index = Math.max(index, 0);
    return formatNumber(numeric / Math.pow(1024, index), 2) + " " + units[index];
  }

  function formatPercent(value) {
    return formatNumber(value, 2) + "%";
  }

  function formatGiB(value) {
    return formatNumber(value, 2) + " GiB";
  }

  function formatDateTime(value) {
    if (!value) {
      return "-";
    }
    var date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function elapsedMilliseconds(run) {
    if (!run) {
      return null;
    }
    if (normalizeStatus(run.status) === "running" && run.started_at) {
      var start = new Date(run.started_at).getTime();
      if (Number.isFinite(start)) {
        var end = run.finished_at ? new Date(run.finished_at).getTime() : Date.now();
        if (Number.isFinite(end)) {
          return Math.max(0, end - start);
        }
      }
    }
    return firstDefined(run.elapsed_ms, run.metrics && run.metrics.total_ms);
  }

  function formatQueuePosition(value) {
    if (value === undefined || value === null) {
      return "-";
    }
    var numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return numeric === 0 ? "\u5373\u5c06\u8fd0\u884c" : "#" + numeric;
    }
    return String(value);
  }

  function renderMetrics(metrics) {
    metrics = metrics || {};
    elements["metric-ttft"].textContent = valueOrDash(metrics.ttft_ms, formatMilliseconds);
    elements["metric-prefill"].textContent = valueOrDash(metrics.prefill_tok_s, function (value) {
      return formatNumber(value, 2);
    });
    elements["metric-tpot"].textContent = valueOrDash(metrics.tpot_ms, formatMilliseconds);
    elements["metric-decode"].textContent = valueOrDash(metrics.decode_tok_s, function (value) {
      return formatNumber(value, 2);
    });
    elements["metric-total"].textContent = valueOrDash(metrics.total_ms, formatMilliseconds);
    elements["metric-prefill-hit"].textContent = valueOrDash(metrics.prefill_hit_pct, formatPercent);
    elements["metric-decode-hit"].textContent = valueOrDash(metrics.decode_hit_pct, formatPercent);
    elements["metric-physical"].textContent = valueOrDash(metrics.physical_read_bytes, formatBytes);
  }

  function shellQuote(value) {
    var text = String(value);
    if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) {
      return text;
    }
    return "'" + text.replace(/'/g, "'\\''") + "'";
  }

  function commandText(command) {
    if (Array.isArray(command)) {
      return command.map(shellQuote).join(" ");
    }
    return command ? String(command) : "-";
  }

  function contentText(content) {
    if (content === undefined || content === null || content === "") {
      return "-";
    }
    return typeof content === "string" ? content : JSON.stringify(content, null, 2);
  }

  function renderValidation(errors) {
    var items = Array.isArray(errors) ? errors : [];
    elements["validation-banner"].classList.toggle("is-hidden", items.length === 0);
    elements["validation-banner"].textContent = items.map(function (item) {
      return typeof item === "string" ? item : (item.message || JSON.stringify(item));
    }).join("\n");
  }

  function renderIoDetails(run) {
    var metrics = run.metrics || {};
    var details = [
      ["Source bytes", metrics.source_bytes, formatBytes],
      ["SSD decode bytes", metrics.ssd_decode_bytes, formatBytes],
      ["Physical read bytes", metrics.physical_read_bytes, formatBytes],
      ["VRAM peak", metrics.vram_peak_gib, formatGiB],
      ["DRAM peak", metrics.dram_peak_gib, formatGiB],
      ["Page cache after decode", metrics.page_cache_after_decode_pct, formatPercent]
    ];
    elements["io-details"].replaceChildren();
    details.forEach(function (detail) {
      var wrapper = document.createElement("div");
      var term = document.createElement("dt");
      var value = document.createElement("dd");
      term.textContent = detail[0];
      value.textContent = valueOrDash(detail[1], detail[2]);
      wrapper.append(term, value);
      elements["io-details"].appendChild(wrapper);
    });
  }

  function safeArtifactUrl(url) {
    if (!url) {
      return null;
    }
    try {
      var parsed = new URL(url, window.location.href);
      if (parsed.origin !== window.location.origin || !/^https?:$/.test(parsed.protocol)) {
        return null;
      }
      return parsed.href;
    } catch (error) {
      return null;
    }
  }

  function renderArtifacts(artifacts) {
    elements["artifact-list"].replaceChildren();
    if (!Array.isArray(artifacts) || !artifacts.length) {
      var empty = document.createElement("div");
      empty.className = "artifact-empty";
      empty.textContent = "\u6682\u65e0\u53ef\u4e0b\u8f7d\u4ea7\u7269";
      elements["artifact-list"].appendChild(empty);
      return;
    }
    artifacts.forEach(function (artifact) {
      var row = document.createElement("div");
      var name = document.createElement("span");
      var size = document.createElement("span");
      row.className = "artifact-row";
      name.className = "artifact-name";
      name.textContent = artifact.name || "artifact";
      name.title = artifact.name || "artifact";
      size.className = "artifact-size";
      size.textContent = artifact.size === undefined || artifact.size === null ? "-" : formatBytes(artifact.size);
      row.append(name, size);

      var href = safeArtifactUrl(artifact.url);
      if (href) {
        var link = document.createElement("a");
        link.href = href;
        link.textContent = "\u4e0b\u8f7d";
        link.setAttribute("download", "");
        row.appendChild(link);
      } else {
        var unavailable = document.createElement("span");
        unavailable.className = "artifact-size";
        unavailable.textContent = "\u4e0d\u53ef\u7528";
        row.appendChild(unavailable);
      }
      elements["artifact-list"].appendChild(row);
    });
  }

  function splitTail(value) {
    if (Array.isArray(value)) {
      return value.map(String).slice(-MAX_LOG_LINES);
    }
    if (value === undefined || value === null || value === "") {
      return [];
    }
    return String(value).replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n").slice(-MAX_LOG_LINES);
  }

  function mergeLogTail(existing, incoming) {
    if (!incoming.length) {
      return existing;
    }
    if (!existing.length) {
      return incoming.slice(-MAX_LOG_LINES);
    }

    var maximumOverlap = Math.min(existing.length, incoming.length);
    var overlap = maximumOverlap;
    while (overlap > 0) {
      var matches = true;
      for (var index = 0; index < overlap; index += 1) {
        if (existing[existing.length - overlap + index] !== incoming[index]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        break;
      }
      overlap -= 1;
    }
    return existing.concat(incoming.slice(overlap)).slice(-MAX_LOG_LINES);
  }

  function syncLogsFromRun(run) {
    ["stdout", "stderr"].forEach(function (stream) {
      var property = stream + "_tail";
      if (run[property] === undefined || run[property] === null) {
        return;
      }
      var incoming = splitTail(run[property]);
      if (!state.logsInitialized[stream]) {
        state.logs[stream] = incoming;
        state.logsInitialized[stream] = true;
      } else {
        state.logs[stream] = mergeLogTail(state.logs[stream], incoming);
      }
      renderLog(stream);
    });
  }

  function renderFollowButton(stream) {
    var button = document.querySelector('.follow-button[data-stream="' + stream + '"]');
    if (!button) {
      return;
    }
    button.setAttribute("aria-pressed", String(state.follow[stream]));
    button.textContent = state.follow[stream] ? labels.following : labels.paused;
  }

  function renderLog(stream) {
    var output = elements[stream + "-log"];
    output.textContent = state.logs[stream].length ? state.logs[stream].join("\n") : labels.noOutput;
    if (state.follow[stream]) {
      output.scrollTop = output.scrollHeight;
    }
    renderFollowButton(stream);
  }

  function appendLog(payload) {
    var stream = payload.stream === "stderr" ? "stderr" : "stdout";
    var sequenceKey = payload.seq === undefined || payload.seq === null
      ? null
      : stream + ":" + String(payload.seq);
    if (sequenceKey && state.logSequences.has(sequenceKey)) {
      return;
    }
    if (sequenceKey) {
      state.logSequences.add(sequenceKey);
    }
    var lines = splitTail(firstDefined(payload.line, payload.text, ""));
    if (!lines.length && firstDefined(payload.line, payload.text, null) === "") {
      lines = [""];
    }
    state.logs[stream] = state.logs[stream].concat(lines).slice(-MAX_LOG_LINES);
    state.logsInitialized[stream] = true;
    renderLog(stream);
  }

  function artifactByName(run, pattern) {
    return (run.artifacts || []).find(function (artifact) {
      return pattern.test(String(artifact.name || ""));
    });
  }

  function renderTokenPlaceholder(run) {
    var container = elements["token-trace"];
    container.replaceChildren();
    if (run.token_trace) {
      var direct = document.createElement("pre");
      direct.className = "code-block trace-output";
      direct.textContent = contentText(run.token_trace);
      container.appendChild(direct);
      return;
    }
    var message = document.createElement("div");
    message.className = "artifact-empty";
    var tokenArtifact = artifactByName(run, /(^|\/)tokens\.csv$/i);
    if (!isTerminal(run)) {
      message.textContent = "\u8fd0\u884c\u5b8c\u6210\u540e\u53ef\u8bfb\u53d6 token trace\u3002";
    } else if (!tokenArtifact) {
      message.textContent = "\u672a\u751f\u6210 tokens.csv\u3002";
    } else {
      message.textContent = "\u5207\u6362\u5230\u672c\u9875\u7b7e\u65f6\u6309\u9700\u8bfb\u53d6 tokens.csv\u3002";
    }
    container.appendChild(message);
  }

  function renderRun(run, options) {
    options = options || {};
    if (!run) {
      elements["current-run-id"].textContent = "-";
      elements["current-run-status"].dataset.status = "empty";
      elements["current-run-status"].textContent = labels.empty;
      elements["queue-position"].textContent = "-";
      elements["elapsed-time"].textContent = "-";
      elements["started-time"].textContent = "-";
      elements["run-empty-state"].classList.remove("is-hidden");
      elements["run-content"].classList.add("is-hidden");
      elements["cancel-button"].disabled = true;
      setFormLocked(false);
      return;
    }

    var status = normalizeStatus(run.status);
    elements["current-run-id"].textContent = run.id || "-";
    elements["current-run-id"].title = run.id || "";
    elements["current-run-status"].dataset.status = status;
    elements["current-run-status"].textContent = labels[status] || status;
    elements["queue-position"].textContent = formatQueuePosition(run.queue_position);
    elements["elapsed-time"].textContent = valueOrDash(elapsedMilliseconds(run), formatMilliseconds);
    elements["started-time"].textContent = formatDateTime(run.started_at);
    elements["run-empty-state"].classList.add("is-hidden");
    elements["run-content"].classList.remove("is-hidden");
    elements["cancel-button"].disabled = !isActive(run) || state.cancelling;
    setFormLocked(isActive(run));

    renderMetrics(run.metrics);
    renderValidation(run.validation_errors);
    elements["run-command"].textContent = commandText(run.command);
    elements["raw-summary"].textContent = contentText(run.raw_summary);
    renderIoDetails(run);
    renderArtifacts(run.artifacts);
    renderTokenPlaceholder(run);
    if (options.syncLogs) {
      syncLogsFromRun(run);
    }
  }

  function applyActiveRun(incoming, options) {
    if (!incoming || !incoming.id || incoming.id !== state.activeRunId) {
      return;
    }
    state.activeRun = mergeRun(state.activeRun, incoming);
    renderRun(state.activeRun, options);
    updateRunInHistory(state.activeRun);
    if (isTerminal(state.activeRun)) {
      closeRunConnection();
      loadRuns(false).catch(function () {});
    }
  }

  function requestSummary(request) {
    if (!request) {
      return "-";
    }
    return [
      "PP " + firstDefined(request.pp, "-"),
      "TG " + firstDefined(request.tg, "-"),
      "UB " + firstDefined(request.ubatch, "-"),
      String(firstDefined(request.predictor, "-")).toUpperCase(),
      "VRAM " + firstDefined(request.cache_vram_mb, "-") + " MiB"
    ].join(" | ");
  }

  function makeCell(text, className) {
    var cell = document.createElement("td");
    cell.textContent = text;
    if (className) {
      cell.className = className;
    }
    return cell;
  }

  function renderHistory() {
    elements["history-loading"].classList.add("is-hidden");
    elements["history-error"].classList.add("is-hidden");
    elements["history-count"].textContent = String(state.runs.length);
    elements["history-body"].replaceChildren();

    if (!state.runs.length) {
      elements["history-empty"].classList.remove("is-hidden");
      elements["history-table-wrap"].classList.add("is-hidden");
      return;
    }

    elements["history-empty"].classList.add("is-hidden");
    elements["history-table-wrap"].classList.remove("is-hidden");
    state.runs.forEach(function (run) {
      var status = normalizeStatus(run.status);
      var metrics = run.metrics || {};
      var row = document.createElement("tr");
      row.tabIndex = 0;
      row.dataset.runId = run.id;
      row.classList.toggle("is-active", run.id === state.activeRunId);
      row.setAttribute("aria-label", "Run " + run.id + ", " + (labels[status] || status));

      var statusCell = makeCell(labels[status] || status, "table-status");
      statusCell.dataset.status = status;
      var idCell = makeCell(run.id || "-", "table-id");
      idCell.title = run.id || "";
      var configCell = makeCell(requestSummary(run.request), "table-config");
      configCell.title = requestSummary(run.request);
      row.append(
        statusCell,
        idCell,
        makeCell(formatDateTime(run.created_at)),
        configCell,
        makeCell(valueOrDash(metrics.prefill_tok_s, function (value) { return formatNumber(value, 2); })),
        makeCell(valueOrDash(metrics.decode_tok_s, function (value) { return formatNumber(value, 2); })),
        makeCell(valueOrDash(elapsedMilliseconds(run), formatMilliseconds))
      );
      row.addEventListener("click", function () {
        selectRun(run.id);
      });
      row.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectRun(run.id);
        }
      });
      elements["history-body"].appendChild(row);
    });
  }

  function updateRunInHistory(run) {
    var index = state.runs.findIndex(function (item) {
      return item.id === run.id;
    });
    if (index === -1) {
      state.runs.unshift(run);
    } else {
      state.runs[index] = mergeRun(state.runs[index], run);
    }
    renderHistory();
  }

  function showHistoryError(message) {
    elements["history-loading"].classList.add("is-hidden");
    elements["history-empty"].classList.add("is-hidden");
    if (!state.runs.length) {
      elements["history-table-wrap"].classList.add("is-hidden");
    }
    elements["history-error"].textContent = message;
    elements["history-error"].classList.remove("is-hidden");
  }

  async function loadRuns(selectInitial) {
    if (state.loadingRuns) {
      return;
    }
    state.loadingRuns = true;
    try {
      var payload = await requestJson("/runs");
      var runs = unwrapRuns(payload).slice();
      runs.sort(function (left, right) {
        return new Date(right.created_at || 0).getTime() - new Date(left.created_at || 0).getTime();
      });
      state.runs = runs;
      renderHistory();

      if (selectInitial && !state.activeRunId && runs.length) {
        var candidate = runs.find(isActive) || runs[0];
        selectRun(candidate.id);
      } else if (state.activeRunId) {
        var summary = runs.find(function (run) {
          return run.id === state.activeRunId;
        });
        if (summary) {
          applyActiveRun(summary, { syncLogs: false });
        }
      }
    } catch (error) {
      showHistoryError(error.message);
      throw error;
    } finally {
      state.loadingRuns = false;
    }
  }

  function resetRunView(runId) {
    state.activeRunId = runId;
    state.activeRun = null;
    state.logs.stdout = [];
    state.logs.stderr = [];
    state.logsInitialized.stdout = false;
    state.logsInitialized.stderr = false;
    state.reconnectLogBaseline = null;
    state.follow.stdout = true;
    state.follow.stderr = true;
    state.logSequences.clear();
    renderLog("stdout");
    renderLog("stderr");
    elements["stream-status"].textContent = "";
    renderHistory();
  }

  async function selectRun(runId) {
    if (!runId || runId === state.activeRunId && state.activeRun) {
      return;
    }
    closeRunConnection();
    resetRunView(runId);
    var listRun = state.runs.find(function (run) {
      return run.id === runId;
    });
    if (listRun) {
      state.activeRun = listRun;
      if (listRun.request) {
        applyFormValues(listRun.request);
      }
      renderRun(listRun, { syncLogs: true });
    }
    elements["stream-status"].textContent = labels.loading;

    try {
      var payload = await requestJson("/runs/" + encodeURIComponent(runId));
      if (state.activeRunId !== runId) {
        return;
      }
      var run = unwrapRun(payload);
      state.activeRun = mergeRun(state.activeRun, run);
      if (run.request) {
        applyFormValues(run.request);
      }
      renderRun(state.activeRun, { syncLogs: true });
      updateRunInHistory(state.activeRun);
      if (isActive(state.activeRun)) {
        openRunStream(runId);
      } else {
        elements["stream-status"].textContent = "";
      }
    } catch (error) {
      if (state.activeRunId === runId) {
        elements["stream-status"].textContent = error.message;
        showToast(error.message, true);
      }
    }
  }

  function clearDetailPoll() {
    state.detailPollGeneration += 1;
    if (state.detailPollTimer) {
      window.clearInterval(state.detailPollTimer);
      state.detailPollTimer = null;
    }
  }

  function clearReconnectTimer(resetAttempt) {
    if (state.reconnectTimer) {
      window.clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
    if (resetAttempt) {
      state.reconnectAttempt = 0;
    }
  }

  function closeRunConnection() {
    state.streamGeneration += 1;
    clearReconnectTimer(true);
    state.reconnectLogBaseline = null;
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    clearDetailPoll();
  }

  async function pollActiveRun(generation) {
    var runId = state.activeRunId;
    if (!runId || generation !== state.detailPollGeneration || state.detailPollInFlight) {
      return;
    }
    state.detailPollInFlight = true;
    try {
      var payload = await requestJson("/runs/" + encodeURIComponent(runId));
      if (runId !== state.activeRunId || generation !== state.detailPollGeneration) {
        return;
      }
      var run = unwrapRun(payload);
      applyActiveRun(run, { syncLogs: true });
      if (isTerminal(run)) {
        clearDetailPoll();
        elements["stream-status"].textContent = "";
      }
    } catch (error) {
      if (runId === state.activeRunId && generation === state.detailPollGeneration) {
        elements["stream-status"].textContent = labels.polling + " | " + error.message;
      }
    } finally {
      state.detailPollInFlight = false;
    }
  }

  function startDetailPolling() {
    if (state.detailPollTimer) {
      return;
    }
    state.detailPollGeneration += 1;
    var generation = state.detailPollGeneration;
    elements["stream-status"].textContent = labels.polling;
    pollActiveRun(generation);
    state.detailPollTimer = window.setInterval(function () {
      pollActiveRun(generation);
    }, DETAIL_POLL_MS);
  }

  function parseEventData(event) {
    if (!event || !event.data) {
      return {};
    }
    try {
      return JSON.parse(event.data);
    } catch (error) {
      return { line: event.data, stream: "stdout", type: "log" };
    }
  }

  function handleStreamEvent(eventName, event) {
    var payload = parseEventData(event);
    var type = payload.type || eventName;
    if (state.reconnectLogBaseline) {
      if (type !== "snapshot") {
        state.logs.stdout = state.reconnectLogBaseline.stdout;
        state.logs.stderr = state.reconnectLogBaseline.stderr;
        state.logsInitialized.stdout = true;
        state.logsInitialized.stderr = true;
        renderLog("stdout");
        renderLog("stderr");
      }
      state.reconnectLogBaseline = null;
    }
    if (type === "log" || payload.line !== undefined) {
      appendLog(payload);
      return;
    }
    var run = unwrapRun(payload);
    if (run && run.id) {
      applyActiveRun(run, { syncLogs: true });
    }
  }

  function scheduleStreamRecovery(runId, generation, source) {
    if (state.reconnectTimer || !isActive(state.activeRun)) {
      return;
    }
    var delay = Math.min(
      SSE_RECOVERY_BASE_MS * Math.pow(2, state.reconnectAttempt),
      SSE_RECOVERY_MAX_MS
    );
    state.reconnectAttempt += 1;
    state.reconnectTimer = window.setTimeout(function () {
      state.reconnectTimer = null;
      if (generation !== state.streamGeneration || runId !== state.activeRunId ||
          state.eventSource !== source || !isActive(state.activeRun)) {
        return;
      }
      if (source.readyState === window.EventSource.OPEN) {
        return;
      }
      startDetailPolling();
      if (source.readyState === window.EventSource.CLOSED) {
        source.close();
        state.eventSource = null;
        connectRunStream(runId, generation);
      }
    }, delay);
  }

  function connectRunStream(runId, generation) {
    if (generation !== state.streamGeneration || runId !== state.activeRunId || !isActive(state.activeRun)) {
      return;
    }
    elements["stream-status"].textContent = labels.connecting;
    var source = new EventSource(API_ROOT + "/runs/" + encodeURIComponent(runId) + "/events");
    state.eventSource = source;
    ["snapshot", "log", "state"].forEach(function (eventName) {
      source.addEventListener(eventName, function (event) {
        if (generation === state.streamGeneration && runId === state.activeRunId &&
            state.eventSource === source) {
          handleStreamEvent(eventName, event);
        }
      });
    });
    source.onmessage = function (event) {
      if (generation === state.streamGeneration && runId === state.activeRunId &&
          state.eventSource === source) {
        handleStreamEvent("message", event);
      }
    };
    source.onopen = function () {
      if (generation === state.streamGeneration && runId === state.activeRunId &&
          state.eventSource === source) {
        clearReconnectTimer(true);
        clearDetailPoll();
        elements["stream-status"].textContent = labels.streaming;
      }
    };
    source.onerror = function () {
      if (generation !== state.streamGeneration || runId !== state.activeRunId ||
          state.eventSource !== source) {
        return;
      }
      if (isActive(state.activeRun)) {
        if (!state.reconnectLogBaseline) {
          state.reconnectLogBaseline = {
            stdout: state.logs.stdout.slice(),
            stderr: state.logs.stderr.slice()
          };
        }
        scheduleStreamRecovery(runId, generation, source);
      } else {
        elements["stream-status"].textContent = "";
      }
    };
  }

  function openRunStream(runId) {
    closeRunConnection();
    if (typeof window.EventSource !== "function") {
      startDetailPolling();
      return;
    }
    connectRunStream(runId, state.streamGeneration);
  }

  async function submitRun(event) {
    event.preventDefault();
    if (state.submitting || isActive(state.activeRun)) {
      return;
    }
    setFormError("");
    var payload = formPayload();
    var validationError = validatePayload(payload);
    if (validationError) {
      setFormError(validationError);
      return;
    }

    state.submitting = true;
    elements["submit-button"].disabled = true;
    try {
      var response = await requestJson("/runs", {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });
      var run = unwrapRun(response);
      if (!run || !run.id) {
        throw new Error("\u670d\u52a1\u672a\u8fd4\u56de run id");
      }
      showToast("\u5df2\u63d0\u4ea4 run " + run.id, false);
      updateRunInHistory(run);
      state.activeRunId = null;
      state.activeRun = null;
      await selectRun(run.id);
    } catch (error) {
      setFormError(error.message, error.validationErrors);
    } finally {
      state.submitting = false;
      elements["submit-button"].disabled = isActive(state.activeRun);
    }
  }

  async function cancelActiveRun() {
    if (!state.activeRunId || !isActive(state.activeRun) || state.cancelling) {
      return;
    }
    state.cancelling = true;
    elements["cancel-button"].disabled = true;
    try {
      var payload = await requestJson("/runs/" + encodeURIComponent(state.activeRunId) + "/cancel", {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json"
        },
        body: "{}"
      });
      var run = unwrapRun(payload);
      if (run && run.id) {
        applyActiveRun(run, { syncLogs: true });
      }
      showToast("\u5df2\u53d1\u9001\u53d6\u6d88\u8bf7\u6c42", false);
    } catch (error) {
      showToast(error.message, true);
    } finally {
      state.cancelling = false;
      elements["cancel-button"].disabled = !isActive(state.activeRun);
    }
  }

  function activateTab(tabName, focus) {
    document.querySelectorAll(".tabs [role=tab]").forEach(function (tab) {
      var selected = tab.dataset.tab === tabName;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) {
        tab.focus();
      }
    });
    document.querySelectorAll(".tab-panel").forEach(function (panel) {
      var selected = panel.id === "panel-" + tabName;
      panel.hidden = !selected;
      panel.classList.toggle("is-hidden", !selected);
    });
    if (tabName === "artifacts" && state.activeRun) {
      loadTokenTrace(state.activeRun);
    }
  }

  function parseCsv(text) {
    var rows = [];
    var row = [];
    var field = "";
    var quoted = false;
    for (var index = 0; index < text.length; index += 1) {
      var character = text[index];
      if (quoted) {
        if (character === '"' && text[index + 1] === '"') {
          field += '"';
          index += 1;
        } else if (character === '"') {
          quoted = false;
        } else {
          field += character;
        }
      } else if (character === '"') {
        quoted = true;
      } else if (character === ",") {
        row.push(field);
        field = "";
      } else if (character === "\n") {
        row.push(field.replace(/\r$/, ""));
        rows.push(row);
        row = [];
        field = "";
      } else {
        field += character;
      }
    }
    if (field !== "" || row.length) {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
    }
    return rows;
  }

  function csvColumnIndex(headers, candidates, fallback) {
    var normalized = headers.map(function (header) {
      return String(header).trim().toLowerCase().replace(/[ -]+/g, "_");
    });
    for (var index = 0; index < candidates.length; index += 1) {
      var found = normalized.indexOf(candidates[index]);
      if (found !== -1) {
        return found;
      }
    }
    return Math.min(fallback, Math.max(headers.length - 1, 0));
  }

  function renderTokenTable(parsed) {
    var container = elements["token-trace"];
    container.replaceChildren();
    if (!parsed.rows.length) {
      var empty = document.createElement("div");
      empty.className = "artifact-empty";
      empty.textContent = "tokens.csv \u4e3a\u7a7a\u3002";
      container.appendChild(empty);
      return;
    }

    var note = document.createElement("p");
    note.className = "trace-note";
    note.textContent = parsed.total > parsed.rows.length
      ? "\u5171 " + parsed.total + " \u884c\uff0c\u663e\u793a\u524d " + parsed.rows.length + " \u884c\u3002"
      : "\u5171 " + parsed.total + " \u884c\u3002";
    var wrapper = document.createElement("div");
    wrapper.className = "trace-table-wrap";
    var table = document.createElement("table");
    var head = document.createElement("thead");
    var headingRow = document.createElement("tr");
    ["Repeat", "Step", "Token"].forEach(function (heading) {
      var cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = heading;
      headingRow.appendChild(cell);
    });
    head.appendChild(headingRow);
    var body = document.createElement("tbody");
    parsed.rows.forEach(function (values) {
      var row = document.createElement("tr");
      values.forEach(function (value) {
        row.appendChild(makeCell(value));
      });
      body.appendChild(row);
    });
    table.append(head, body);
    wrapper.appendChild(table);
    container.append(note, wrapper);
  }

  async function loadTokenTrace(run) {
    if (run.token_trace || !isTerminal(run)) {
      return;
    }
    var artifact = artifactByName(run, /(^|\/)tokens\.csv$/i);
    var href = artifact && safeArtifactUrl(artifact.url);
    if (!href) {
      return;
    }
    if (state.tokenCache.has(run.id)) {
      renderTokenTable(state.tokenCache.get(run.id));
      return;
    }
    var container = elements["token-trace"];
    container.replaceChildren();
    var loading = document.createElement("div");
    loading.className = "artifact-empty";
    loading.textContent = "\u6b63\u5728\u8bfb\u53d6 tokens.csv";
    container.appendChild(loading);
    try {
      var response = await fetch(href, { headers: { "Accept": "text/csv,text/plain" } });
      if (!response.ok) {
        throw new Error("HTTP " + response.status);
      }
      var csvRows = parseCsv(await response.text());
      if (state.activeRunId !== run.id) {
        return;
      }
      var headers = csvRows.shift() || [];
      var repeatIndex = csvColumnIndex(headers, ["repeat", "repetition", "run"], 0);
      var stepIndex = csvColumnIndex(headers, ["step", "token_index", "position"], 1);
      var tokenIndex = csvColumnIndex(headers, ["token", "token_text", "token_id"], 2);
      var parsed = {
        total: csvRows.length,
        rows: csvRows.slice(0, 500).map(function (row) {
          return [row[repeatIndex] || "", row[stepIndex] || "", row[tokenIndex] || ""];
        })
      };
      state.tokenCache.set(run.id, parsed);
      renderTokenTable(parsed);
    } catch (error) {
      if (state.activeRunId === run.id) {
        container.replaceChildren();
        var failed = document.createElement("div");
        failed.className = "artifact-empty";
        failed.textContent = "\u65e0\u6cd5\u8bfb\u53d6 tokens.csv: " + error.message;
        container.appendChild(failed);
      }
    }
  }

  function showToast(message, isError) {
    var toast = document.createElement("div");
    toast.className = "toast" + (isError ? " error" : "");
    toast.textContent = message;
    elements["toast-region"].appendChild(toast);
    window.setTimeout(function () {
      toast.remove();
    }, 4500);
  }

  async function refreshAll() {
    elements["refresh-button"].disabled = true;
    try {
      await Promise.all([loadConfig(false), loadRuns(false)]);
      showToast("\u5df2\u5237\u65b0", false);
    } catch (error) {
      showToast(error.message, true);
    } finally {
      elements["refresh-button"].disabled = false;
    }
  }

  function bindEvents() {
    elements["run-form"].addEventListener("submit", submitRun);
    elements["cancel-button"].addEventListener("click", cancelActiveRun);
    elements["refresh-button"].addEventListener("click", refreshAll);
    elements["defaults-button"].addEventListener("click", function () {
      if (state.config) {
        applyFormValues(state.config.defaults || {});
        setFormError("");
      }
    });
    elements["run-form"].querySelectorAll('input[name="prompt_mode"]').forEach(function (input) {
      input.addEventListener("change", updatePromptMode);
    });
    elements["host-cache"].addEventListener("change", updateFormDependencies);
    elements.predictor.addEventListener("change", updateFormDependencies);

    document.querySelectorAll(".follow-button").forEach(function (button) {
      button.addEventListener("click", function () {
        var stream = button.dataset.stream;
        state.follow[stream] = !state.follow[stream];
        if (state.follow[stream]) {
          elements[stream + "-log"].scrollTop = elements[stream + "-log"].scrollHeight;
        }
        renderFollowButton(stream);
      });
    });

    ["stdout", "stderr"].forEach(function (stream) {
      var output = elements[stream + "-log"];
      output.addEventListener("scroll", function () {
        var atBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 24;
        if (state.follow[stream] !== atBottom) {
          state.follow[stream] = atBottom;
          renderFollowButton(stream);
        }
      }, { passive: true });
    });

    var tabs = Array.from(document.querySelectorAll(".tabs [role=tab]"));
    tabs.forEach(function (tab, index) {
      tab.addEventListener("click", function () {
        activateTab(tab.dataset.tab, false);
      });
      tab.addEventListener("keydown", function (event) {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight" && event.key !== "Home" && event.key !== "End") {
          return;
        }
        event.preventDefault();
        var targetIndex = index;
        if (event.key === "ArrowLeft") {
          targetIndex = (index - 1 + tabs.length) % tabs.length;
        } else if (event.key === "ArrowRight") {
          targetIndex = (index + 1) % tabs.length;
        } else if (event.key === "Home") {
          targetIndex = 0;
        } else if (event.key === "End") {
          targetIndex = tabs.length - 1;
        }
        activateTab(tabs[targetIndex].dataset.tab, true);
      });
    });
  }

  function startTimers() {
    state.elapsedTimer = window.setInterval(function () {
      if (state.activeRun && normalizeStatus(state.activeRun.status) === "running") {
        elements["elapsed-time"].textContent = valueOrDash(elapsedMilliseconds(state.activeRun), formatMilliseconds);
      }
    }, 1000);
    state.runsRefreshTimer = window.setInterval(function () {
      loadRuns(false).catch(function () {});
    }, RUNS_REFRESH_MS);
  }

  async function initialize() {
    cacheElements();
    bindEvents();
    updatePromptMode();
    renderRun(null);
    renderLog("stdout");
    renderLog("stderr");
    setServiceState("loading", "");
    try {
      await loadConfig(true);
    } catch (error) {
      showToast(error.message, true);
    }
    try {
      await loadRuns(true);
    } catch (error) {
      if (state.config) {
        showToast(error.message, true);
      }
    }
    startTimers();
  }

  document.addEventListener("DOMContentLoaded", initialize);
}());
