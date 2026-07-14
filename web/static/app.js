const chatLog = document.getElementById("chatLog");
const statusPill = document.getElementById("statusPill");
const sendButton = document.getElementById("sendButton");
const userInput = document.getElementById("userInput");
const checkpointSelect = document.getElementById("checkpoint");
const clearChatButton = document.getElementById("clearChat");
const seedPromptButton = document.getElementById("seedPrompt");
const modelName = document.getElementById("modelName");
const modelTagline = document.getElementById("modelTagline");
const modelDescription = document.getElementById("modelDescription");
const rawOutput = document.getElementById("rawOutput");
const rawStatus = document.getElementById("rawStatus");

const temperature = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperatureValue");
const topK = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const maxTokens = document.getElementById("maxTokens");
const maxTokensValue = document.getElementById("maxTokensValue");

let messages = [];
let lastRawText = "";
const checkpointMap = new Map(
  (window.__TINYLMS__.checkpoints || []).map((checkpoint) => [checkpoint.value, checkpoint]),
);

function setStatus(text) {
  statusPill.textContent = text;
}

function setRawStatus(text) {
  rawStatus.textContent = text;
}

function setRawOutput(text) {
  lastRawText = text;
  rawOutput.textContent = text;
  rawOutput.scrollTop = rawOutput.scrollHeight;
}

function appendRawOutput(delta) {
  if (!delta) return;
  lastRawText += delta;
  rawOutput.textContent = lastRawText;
  rawOutput.scrollTop = rawOutput.scrollHeight;
}

function selectedCheckpointMeta() {
  return checkpointMap.get(checkpointSelect.value) || null;
}

function updateRangeLabels() {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
  topKValue.textContent = topK.value;
  maxTokensValue.textContent = maxTokens.value;
}

function updateModelCard() {
  const meta = selectedCheckpointMeta();
  if (!meta) return;
  modelName.textContent = meta.label;
  modelTagline.textContent = meta.tagline || "Checkpoint";
  modelDescription.textContent = meta.description || "";
}

function applyCheckpointDefaults() {
  const meta = selectedCheckpointMeta();
  if (!meta) return;
  if (typeof meta.default_temperature === "number") {
    temperature.value = meta.default_temperature;
    updateRangeLabels();
  }
}

function renderEmptyState() {
  chatLog.innerHTML = `
    <div class="empty-state">
      <p class="eyebrow">First contact</p>
      <h3>Start simple and watch the model’s habits show up fast.</h3>
      <p class="subtle">Good tiny-model prompts: “What is a tokenizer?”, “Tell me a short story about a rabbit.”, “Explain loss in simple words.”</p>
    </div>
  `;
}

function renderMessages() {
  if (!messages.length) {
    renderEmptyState();
    return;
  }

  chatLog.innerHTML = "";
  for (const message of messages) {
    const bubble = document.createElement("article");
    bubble.className = `message ${message.role}`;
    bubble.innerHTML = `
      <span class="message-label">${message.role}</span>
      <div>${escapeHtml(message.content)}</div>
    `;
    chatLog.appendChild(bubble);
  }
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendMessage(role, content = "") {
  const message = { role, content };
  messages.push(message);
  renderMessages();
  return message;
}

function updateLastMessage(message, content) {
  message.content = content;
  renderMessages();
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
    .replaceAll("\n", "<br>");
}

async function readStreamingResponse(response, assistantMessage) {
  if (!response.body) {
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Generation failed");
    }
    setRawOutput(data.raw_text || data.reply || "");
    updateLastMessage(assistantMessage, data.reply || "(empty reply)");
    const label = data.checkpoint_meta?.label || "Model";
    setStatus(`${label} on ${data.device}`);
    setRawStatus("Complete");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamedReply = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "error") {
        throw new Error(event.error || "Generation failed");
      }
      if (event.type === "meta") {
        const label = event.checkpoint_meta?.label || "Model";
        setStatus(`${label} on ${event.device}`);
        setRawStatus("Streaming");
        continue;
      }
      if (event.type === "start") {
        setRawOutput(event.raw_text || "");
        continue;
      }
      if (event.type === "token") {
        appendRawOutput(event.delta || "");
        streamedReply = extractVisibleAssistantReply(lastRawText);
        updateLastMessage(assistantMessage, streamedReply || "(streaming raw output...)");
        continue;
      }
      if (event.type === "done") {
        setRawOutput(event.raw_text || lastRawText);
        updateLastMessage(assistantMessage, event.reply || extractVisibleAssistantReply(lastRawText) || "(empty reply)");
        setRawStatus("Complete");
      }
    }
  }
}

function extractVisibleAssistantReply(fullText) {
  let text = fullText;
  if (text.includes("Assistant:")) {
    text = text.split("Assistant:").at(-1);
  }
  for (const stop of ["\nUser:", "\nSystem:", "\nAssistant:"]) {
    if (text.includes(stop)) {
      text = text.split(stop)[0];
    }
  }
  return text.replace(/^[\s:;,.!?\-"'`]+/, "").trim();
}

async function sendMessage() {
  const content = userInput.value.trim();
  if (!content) return;

  messages.push({ role: "user", content });
  const requestMessages = messages.map((message) => ({ ...message }));
  userInput.value = "";
  renderMessages();
  const assistantMessage = appendMessage("assistant", "(streaming...)");
  setRawOutput("");
  setRawStatus("Starting");
  sendButton.disabled = true;
  setStatus("Generating");

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
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Generation failed");
    }
    await readStreamingResponse(response, assistantMessage);
  } catch (error) {
    updateLastMessage(assistantMessage, `Error: ${error.message}`);
    renderMessages();
    setStatus("Error");
    setRawStatus("Error");
  } finally {
    sendButton.disabled = false;
  }
}

sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    sendMessage();
  }
});

clearChatButton.addEventListener("click", () => {
  messages = [];
  renderMessages();
  setRawOutput("");
  setRawStatus("Waiting");
  setStatus("Ready");
});

seedPromptButton.addEventListener("click", () => {
  userInput.value = "What is a tokenizer, explained like I am a beginner?";
  userInput.focus();
});

temperature.addEventListener("input", updateRangeLabels);
topK.addEventListener("input", updateRangeLabels);
maxTokens.addEventListener("input", updateRangeLabels);
checkpointSelect.addEventListener("change", () => {
  updateModelCard();
  applyCheckpointDefaults();
  setStatus("Ready");
});

updateRangeLabels();
updateModelCard();
applyCheckpointDefaults();
renderMessages();
