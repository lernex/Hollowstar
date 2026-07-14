const $ = (id) => document.getElementById(id);

const chatLog = $("chatLog");
const sendButton = $("sendButton");
const userInput = $("userInput");
const checkpointSelect = $("checkpoint");
const clearChatButton = $("clearChat");
const seedPromptButton = $("seedPrompt");
const modelName = $("modelName");
const modelTagline = $("modelTagline");
const modelDescription = $("modelDescription");
const topbarModel = $("topbarModel");
const rawOutput = $("rawOutput");
const rawStatus = $("rawStatus");
const rawToggle = $("rawToggle");
const rawDrawer = $("rawDrawer");
const genStat = $("genStat");
const statusPill = $("statusPill");
const statusText = $("statusText");

const temperature = $("temperature");
const temperatureValue = $("temperatureValue");
const topK = $("topK");
const topKValue = $("topKValue");
const maxTokens = $("maxTokens");
const maxTokensValue = $("maxTokensValue");

const SAMPLE_PROMPTS = [
  "What is a tokenizer, explained simply?",
  "Write a short poem about the ocean.",
  "Explain why the sky is blue in two sentences.",
  "List three tips for studying effectively.",
];

let messages = [];          // [{role, content}]
let currentController = null; // AbortController while generating
let rawText = "";

/* ---------- status ---------- */
function setStatus(text, kind = "ready") {
  statusText.textContent = text;
  statusPill.className = `status ${kind}`;
}
function setRawStatus(text) { rawStatus.textContent = text; }

/* ---------- model card ---------- */
const checkpointMap = new Map(
  (window.__TINYLMS__.checkpoints || []).map((c) => [c.value, c]),
);
function selectedMeta() { return checkpointMap.get(checkpointSelect.value) || null; }
function updateModelCard() {
  const meta = selectedMeta();
  if (!meta) return;
  modelName.textContent = meta.label;
  modelTagline.textContent = meta.tagline || "Checkpoint";
  modelDescription.textContent = meta.description || "";
  topbarModel.textContent = meta.label;
}
function applyCheckpointDefaults() {
  const meta = selectedMeta();
  if (meta && typeof meta.default_temperature === "number") {
    temperature.value = meta.default_temperature;
    updateRangeLabels();
  }
}
function updateRangeLabels() {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
  topKValue.textContent = topK.value;
  maxTokensValue.textContent = maxTokens.value;
}

/* ---------- rendering ---------- */
function nearBottom() {
  return chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 90;
}
function stickToBottom(force) {
  if (force || nearBottom()) chatLog.scrollTop = chatLog.scrollHeight;
}

function renderEmptyState() {
  chatLog.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "empty-state";
  wrap.innerHTML = `
    <div class="glyph">M</div>
    <h3>Talk to Metis</h3>
    <p>A small model trained from scratch — it answers simple questions and reasons at its scale. Start with something basic.</p>
    <div class="chips"></div>`;
  const chips = wrap.querySelector(".chips");
  SAMPLE_PROMPTS.forEach((p) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.textContent = p;
    b.addEventListener("click", () => { userInput.value = p; autoGrow(); userInput.focus(); });
    chips.appendChild(b);
  });
  chatLog.appendChild(wrap);
}

// Create a bubble node and return handles for incremental updates.
function addBubble(role, content) {
  const empty = chatLog.querySelector(".empty-state");
  if (empty) empty.remove();

  const el = document.createElement("article");
  el.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "You" : "M";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = content;
  const meta = document.createElement("div");
  meta.className = "msg-meta";
  bubble.append(body, meta);
  el.append(avatar, bubble);
  chatLog.appendChild(el);
  stickToBottom(true);
  return { el, body, meta };
}

/* ---------- reply extraction ---------- */
function extractReply(full) {
  let text = full;
  if (text.includes("Assistant:")) text = text.split("Assistant:").at(-1);
  for (const stop of ["\nUser:", "\nUser :", "\nSystem:"]) {
    const i = text.indexOf(stop);
    if (i !== -1) text = text.slice(0, i);
  }
  return text.replace(/^[\s\n]+/, "").replace(/\s+$/, "");
}

/* ---------- raw drawer ---------- */
function setRaw(text) {
  rawText = text;
  rawOutput.textContent = text;
  rawOutput.scrollTop = rawOutput.scrollHeight;
}
rawToggle.addEventListener("click", () => {
  const open = rawDrawer.hidden;
  rawDrawer.hidden = !open;
  rawToggle.setAttribute("aria-expanded", String(open));
});

/* ---------- send / stream ---------- */
function setSending(on) {
  if (on) {
    sendButton.classList.add("stop");
    sendButton.setAttribute("aria-label", "Stop");
    sendButton.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>';
  } else {
    sendButton.classList.remove("stop");
    sendButton.setAttribute("aria-label", "Send");
    sendButton.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M3.4 20.4 21 12 3.4 3.6 3 10l12 2-12 2z"/></svg>';
  }
}

async function sendMessage() {
  // If already generating, this button acts as Stop.
  if (currentController) { currentController.abort(); return; }

  const content = userInput.value.trim();
  if (!content) return;

  messages.push({ role: "user", content });
  addBubble("user", content);
  const requestMessages = messages.map((m) => ({ ...m }));
  userInput.value = "";
  autoGrow();

  const assistant = addBubble("assistant", "");
  const caret = document.createElement("span");
  caret.className = "caret";
  assistant.body.appendChild(caret);

  setSending(true);
  setStatus("Generating…", "busy");
  setRaw("");
  setRawStatus("Streaming");
  genStat.textContent = "";

  currentController = new AbortController();
  let full = "";
  let nTokens = 0;
  let t0 = 0;
  let aborted = false;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        checkpoint: checkpointSelect.value || window.__TINYLMS__.defaultCheckpoint,
        messages: requestMessages,
        temperature: Number(temperature.value),
        top_k: Number(topK.value),
        max_new_tokens: Number(maxTokens.value),
      }),
      signal: currentController.signal,
    });

    if (!response.ok || !response.body) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        let event;
        try { event = JSON.parse(line); } catch { continue; }

        if (event.type === "error") throw new Error(event.error || "Generation failed");
        if (event.type === "meta") { setRawStatus("Streaming"); continue; }
        if (event.type === "start") { setRaw(event.raw_text || ""); continue; }

        if (event.type === "token") {
          if (!t0) t0 = performance.now();
          nTokens += 1;
          full += event.delta || "";
          setRaw(rawText + (event.delta || ""));
          // incremental: only touch this bubble's text node + reposition caret
          assistant.body.textContent = extractReply(full);
          assistant.body.appendChild(caret);
          stickToBottom();
          continue;
        }

        if (event.type === "done") {
          full = event.raw_text || full;
          assistant.body.textContent = event.reply || extractReply(full) || "(empty reply)";
          setRaw(full);
          setRawStatus("Complete");
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      aborted = true;
    } else {
      caret.remove();
      assistant.body.textContent = (full && extractReply(full)) || `⚠ ${err.message}`;
      assistant.el.classList.add("assistant");
      setStatus("Error", "error");
      setRawStatus("Error");
    }
  } finally {
    caret.remove();
    currentController = null;
    setSending(false);

    const replyText = assistant.body.textContent.trim();
    messages.push({ role: "assistant", content: replyText });

    if (t0 && nTokens) {
      const secs = (performance.now() - t0) / 1000;
      const meta = `${nTokens} tok · ${(nTokens / secs).toFixed(1)} tok/s`;
      assistant.meta.textContent = aborted ? `stopped · ${meta}` : meta;
      genStat.textContent = meta;
    }
    if (aborted) { setStatus("Stopped", "ready"); setRawStatus("Stopped"); }
    else if (statusPill.classList.contains("busy")) setStatus("Ready", "ready");
    userInput.focus();
  }
}

/* ---------- textarea autogrow ---------- */
function autoGrow() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
}

/* ---------- events ---------- */
sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("input", autoGrow);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

clearChatButton.addEventListener("click", () => {
  if (currentController) currentController.abort();
  messages = [];
  renderEmptyState();
  setRaw("");
  setRawStatus("Waiting");
  setStatus("Ready", "ready");
  genStat.textContent = "";
});

seedPromptButton.addEventListener("click", () => {
  userInput.value = SAMPLE_PROMPTS[Math.floor(Math.random() * SAMPLE_PROMPTS.length)];
  autoGrow();
  userInput.focus();
});

temperature.addEventListener("input", updateRangeLabels);
topK.addEventListener("input", updateRangeLabels);
maxTokens.addEventListener("input", updateRangeLabels);
checkpointSelect.addEventListener("change", () => {
  updateModelCard();
  applyCheckpointDefaults();
  setStatus("Ready", "ready");
});

/* ---------- init ---------- */
updateRangeLabels();
updateModelCard();
applyCheckpointDefaults();
renderEmptyState();
setStatus("Ready", "ready");
