const chatLog = document.getElementById("chatLog");
const statusPill = document.getElementById("statusPill");
const sendButton = document.getElementById("sendButton");
const userInput = document.getElementById("userInput");
const systemPrompt = document.getElementById("systemPrompt");
const checkpointSelect = document.getElementById("checkpoint");
const clearChatButton = document.getElementById("clearChat");
const seedPromptButton = document.getElementById("seedPrompt");
const modelName = document.getElementById("modelName");
const modelTagline = document.getElementById("modelTagline");
const modelDescription = document.getElementById("modelDescription");

const temperature = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperatureValue");
const topK = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const maxTokens = document.getElementById("maxTokens");
const maxTokensValue = document.getElementById("maxTokensValue");

let messages = [];
const checkpointMap = new Map(
  (window.__TINYLMS__.checkpoints || []).map((checkpoint) => [checkpoint.value, checkpoint]),
);

function setStatus(text) {
  statusPill.textContent = text;
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

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
    .replaceAll("\n", "<br>");
}

async function sendMessage() {
  const content = userInput.value.trim();
  if (!content) return;

  messages.push({ role: "user", content });
  userInput.value = "";
  renderMessages();
  sendButton.disabled = true;
  setStatus("Generating");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        checkpoint: checkpointSelect.value || window.__TINYLMS__.defaultCheckpoint,
        system_prompt: systemPrompt.value,
        messages,
        temperature: Number(temperature.value),
        top_k: Number(topK.value),
        max_new_tokens: Number(maxTokens.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Generation failed");
    }

    messages.push({ role: "assistant", content: data.reply || "(empty reply)" });
    renderMessages();
    const label = data.checkpoint_meta?.label || "Model";
    setStatus(`${label} on ${data.device}`);
  } catch (error) {
    messages.push({ role: "assistant", content: `Error: ${error.message}` });
    renderMessages();
    setStatus("Error");
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
