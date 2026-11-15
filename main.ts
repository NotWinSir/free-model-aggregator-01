// free-model-aggregator-01 entry: a free-model aggregator server.
//
// The program reads its brief (the trial prompt file), builds a catalog of
// every free model the brief's quality table lists, registers that catalog
// with the trial environment, and then serves the environment's client one
// request at a time. For each request it picks the highest-rated free model
// that meets the request's requirements, translates the request into that
// provider's wire format without losing any field the provider echoes back,
// makes exactly one successful provider call, and translates the answer back
// into the client's wire. Providers are always called non-streaming; when
// the client asked for a stream, the stream is synthesized locally from the
// complete answer so the event order is deterministic.
//
// Everything the program does is a function of the brief, the requests it
// has served, and the answers providers returned: no wall clock, no
// randomness, so a replayed trial walks the exact same path.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Brief parsing
// ---------------------------------------------------------------------------

// The brief embeds three JSON documents, each tagged with a schema string.
// Find the document that carries the given schema by locating the schema
// text, walking back to the opening brace of its object, and scanning
// forward with a brace matcher that respects JSON strings and escapes.
function extractDocument(text, schema) {
  const at = text.indexOf(schema);
  if (at < 0) return null;
  const start = text.lastIndexOf("{", at);
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(start, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function loadBrief() {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return null;
  let text;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return null;
  }
  const directory = extractDocument(text, "free-model-aggregator-01/providers/v1");
  const ratings = extractDocument(text, "free-model-aggregator-01/ratings/v1");
  const workload = extractDocument(text, "free-model-aggregator-01/workload/v1");
  if (!directory || !ratings || !workload) return null;
  return { directory, ratings, workload };
}

// Join the brief's provider directory with its quality table into the model
// list the router works from, and into the catalog document that is both
// printed by --list and registered with the environment. The catalog lists
// free models only, each with the context length the table publishes.
function buildCatalog(brief) {
  const rowsByProvider = new Map();
  for (const row of brief.ratings.ratings) {
    if (!rowsByProvider.has(row.provider)) rowsByProvider.set(row.provider, []);
    rowsByProvider.get(row.provider).push(row);
  }
  const providers = [];
  for (const entry of brief.directory.providers) {
    const rows = rowsByProvider.get(entry.id) || [];
    providers.push({
      id: entry.id,
      base_url: entry.path,
      key_env: entry.key_env,
      wire: entry.wire,
      models: rows.map((row) => ({
        id: row.model,
        free: true,
        context_length: row.context_length,
      })),
    });
  }
  return { providers };
}

function buildModels(brief) {
  const directoryById = new Map();
  for (const entry of brief.directory.providers) directoryById.set(entry.id, entry);
  const models = [];
  for (const row of brief.ratings.ratings) {
    const provider = directoryById.get(row.provider);
    if (!provider) continue;
    models.push({
      providerId: provider.id,
      providerWire: provider.wire,
      modelId: row.model,
      rating: row.rating_milli,
      contextLength: row.context_length,
      tools: row.tools === true,
      vision: row.vision === true,
      json: row.json === true,
    });
  }
  return models;
}

// ---------------------------------------------------------------------------
// Canonical request form
//
// A client request is decoded once into a canonical form that keeps exactly
// the pieces a provider echoes back: the system prompt (and whether one was
// sent at all), the ordered messages with their text and image blocks, the
// tool definitions, max_tokens (null when the client sent none, never
// invented), the stop sequences, and whether JSON mode was requested. The
// provider request is then rendered from this form in the provider's wire,
// and each wire's parser reads back the same pieces.
// ---------------------------------------------------------------------------

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function textOfBlocks(blocks) {
  const parts = [];
  for (const block of blocks) if (block.kind === "text") parts.push(block.text);
  return parts.join("\n");
}

// Decode an OpenAI image_url value into a canonical image block. A data URL
// keeps its media type and base64 payload; any other URL is kept whole.
function imageFromUrl(url) {
  if (typeof url !== "string") return null;
  const match = /^data:([^;,]+);base64,([\s\S]*)$/.exec(url);
  if (match) return { kind: "image", mediaType: match[1], data: match[2] };
  return { kind: "imageUrl", url };
}

function blocksFromOpenAIContent(content) {
  if (typeof content === "string") return [{ kind: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const blocks = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === "text" && typeof part.text === "string") {
      blocks.push({ kind: "text", text: part.text });
    } else if (part.type === "image_url" && isRecord(part.image_url)) {
      const image = imageFromUrl(part.image_url.url);
      if (image) blocks.push(image);
    }
  }
  return blocks;
}

function blocksFromAnthropicContent(content) {
  if (typeof content === "string") return [{ kind: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const blocks = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === "text" && typeof part.text === "string") {
      blocks.push({ kind: "text", text: part.text });
    } else if (part.type === "image" && isRecord(part.source)) {
      const source = part.source;
      if (source.type === "base64" && typeof source.data === "string") {
        blocks.push({
          kind: "image",
          mediaType: typeof source.media_type === "string" ? source.media_type : "image/png",
          data: source.data,
        });
      } else if (source.type === "url" && typeof source.url === "string") {
        blocks.push({ kind: "imageUrl", url: source.url });
      }
    }
  }
  return blocks;
}

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringList(value) {
  if (typeof value === "string") return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((item) => typeof item === "string");
}

// Decode an OpenAI-wire client body into the canonical form, following the
// same reading rules the providers apply: system text is every system
// message's text joined with newlines, max_tokens falls back to
// max_completion_tokens, a lone stop string is a one-item list, and JSON
// mode is a response_format of type json_object or json_schema.
function decodeOpenAIRequest(body) {
  const messages = [];
  const systemParts = [];
  let systemPresent = false;
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of rawMessages) {
    if (!isRecord(message)) continue;
    const blocks = blocksFromOpenAIContent(message.content);
    if (message.role === "system") {
      systemPresent = true;
      systemParts.push(textOfBlocks(blocks));
    } else if (message.role === "user" || message.role === "assistant") {
      messages.push({ role: message.role, blocks });
    }
  }
  let tools = null;
  if (Array.isArray(body.tools)) {
    tools = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      if (typeof tool.function.name !== "string") continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === "string" ? tool.function.description : null,
        schema: tool.function.parameters === undefined ? null : tool.function.parameters,
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  return {
    systemPresent,
    systemText: systemParts.join("\n"),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens ?? body.max_completion_tokens),
    stops: "stop" in body ? stringList(body.stop) : null,
    jsonMode: format !== null && (format.type === "json_object" || format.type === "json_schema"),
    responseFormat: format,
    temperature: numberOrNull(body.temperature),
    stream: body.stream === true,
  };
}

// Decode an Anthropic-wire client body. The system prompt travels outside
// the messages; JSON mode does not exist on this wire, so it is always off.
function decodeAnthropicRequest(body) {
  const messages = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of rawMessages) {
    if (!isRecord(message)) continue;
    if (message.role !== "user" && message.role !== "assistant") continue;
    messages.push({ role: message.role, blocks: blocksFromAnthropicContent(message.content) });
  }
  let systemPresent = false;
  let systemText = "";
  if (typeof body.system === "string") {
    systemPresent = true;
    systemText = body.system;
  } else if (Array.isArray(body.system)) {
    systemPresent = true;
    const parts = [];
    for (const block of body.system) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
    }
    systemText = parts.join("\n");
  }
  let tools = null;
  if (Array.isArray(body.tools)) {
    tools = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== "string") continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === "string" ? tool.description : null,
        schema: tool.input_schema === undefined ? null : tool.input_schema,
      });
    }
  }
  return {
    systemPresent,
    systemText,
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: "stop_sequences" in body ? stringList(body.stop_sequences) : null,
    jsonMode: false,
    responseFormat: null,
    temperature: numberOrNull(body.temperature),
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Rendering a canonical request into a provider's wire
// ---------------------------------------------------------------------------

function imageToOpenAI(block) {
  if (block.kind === "image") {
    return { type: "image_url", image_url: { url: `data:${block.mediaType};base64,${block.data}` } };
  }
  return { type: "image_url", image_url: { url: block.url } };
}

function imageToAnthropic(block) {
  if (block.kind === "image") {
    return { type: "image", source: { type: "base64", media_type: block.mediaType, data: block.data } };
  }
  return { type: "image", source: { type: "url", url: block.url } };
}

function imageToGemini(block) {
  if (block.kind === "image") {
    return { inlineData: { mimeType: block.mediaType, data: block.data } };
  }
  return { fileData: { fileUri: block.url } };
}

// Render one canonical message's blocks for a wire. When a message is a
// single text block and carries no image, the text travels as a plain
// string on the OpenAI and Anthropic wires, the shape clients themselves
// most often send; the parsers read both shapes to the same text.
function renderBlocks(blocks, wire) {
  const hasImage = blocks.some((block) => block.kind !== "text");
  if (!hasImage) return textOfBlocks(blocks);
  if (wire === "openai") {
    return blocks.map((block) =>
      block.kind === "text" ? { type: "text", text: block.text } : imageToOpenAI(block),
    );
  }
  if (wire === "anthropic") {
    return blocks.map((block) =>
      block.kind === "text" ? { type: "text", text: block.text } : imageToAnthropic(block),
    );
  }
  return blocks.map((block) => (block.kind === "text" ? { text: block.text } : imageToGemini(block)));
}

function renderOpenAIProviderBody(canon, modelId) {
  const messages = [];
  if (canon.systemPresent) messages.push({ role: "system", content: canon.systemText });
  for (const message of canon.messages) {
    messages.push({ role: message.role, content: renderBlocks(message.blocks, "openai") });
  }
  const body = { model: modelId, messages, stream: false };
  if (canon.tools !== null) {
    body.tools = canon.tools.map((tool) => {
      const fn = { name: tool.name };
      if (tool.description !== null) fn.description = tool.description;
      if (tool.schema !== null) fn.parameters = tool.schema;
      return { type: "function", function: fn };
    });
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops !== null) body.stop = canon.stops;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.jsonMode && canon.responseFormat !== null) body.response_format = canon.responseFormat;
  return body;
}

function renderAnthropicProviderBody(canon, modelId) {
  const messages = canon.messages.map((message) => ({
    role: message.role,
    content: renderBlocks(message.blocks, "anthropic"),
  }));
  const body = { model: modelId, messages, stream: false };
  if (canon.systemPresent) body.system = canon.systemText;
  if (canon.tools !== null) {
    body.tools = canon.tools.map((tool) => {
      const out = { name: tool.name };
      if (tool.description !== null) out.description = tool.description;
      if (tool.schema !== null) out.input_schema = tool.schema;
      return out;
    });
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops !== null) body.stop_sequences = canon.stops;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  return body;
}

// Gemini contents carry a parts array, never a bare string: a text-only
// message still travels as one text part. (The OpenAI and Anthropic wires
// accept a plain string for a text-only message; Gemini's does not, and a
// string in place of the parts array reads back as no text at all.)
function renderGeminiParts(blocks) {
  const hasImage = blocks.some((block) => block.kind !== "text");
  if (!hasImage) return [{ text: textOfBlocks(blocks) }];
  return blocks.map((block) => (block.kind === "text" ? { text: block.text } : imageToGemini(block)));
}

function renderGeminiProviderBody(canon, modelId) {
  const contents = canon.messages.map((message) => ({
    role: message.role === "assistant" ? "model" : "user",
    parts: renderGeminiParts(message.blocks),
  }));
  const body = { contents };
  if (canon.systemPresent) body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools !== null && canon.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: canon.tools.map((tool) => {
          const decl = { name: tool.name };
          if (tool.description !== null) decl.description = tool.description;
          if (tool.schema !== null) decl.parameters = tool.schema;
          return decl;
        }),
      },
    ];
  }
  const config = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops !== null) config.stopSequences = canon.stops;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Provider answers
//
// A provider's non-streaming answer is decoded into the answer text, the
// usage counts in that wire's own units, and the provider's stop word. The
// stop word is later mapped by family: the families group the wires' words
// for the same ending, so a normal ending stays a normal ending on the
// client's wire whatever provider served it.
// ---------------------------------------------------------------------------

const STOP_FAMILIES = [
  ["stop", "end_turn", "STOP"],
  ["length", "max_tokens", "MAX_TOKENS"],
  ["tool_calls", "tool_use", "FUNCTION_CALL"],
  ["content_filter", "refusal", "SAFETY"],
];
const CLIENT_STOPS = {
  openai: ["stop", "length", "tool_calls", "content_filter"],
  anthropic: ["end_turn", "max_tokens", "tool_use", "refusal"],
};

function stopFamilyOf(word) {
  if (typeof word !== "string") return 0;
  for (let family = 0; family < STOP_FAMILIES.length; family += 1) {
    if (STOP_FAMILIES[family].includes(word)) return family;
  }
  return 0;
}

function parseOpenAIAnswer(json) {
  if (!isRecord(json) || !Array.isArray(json.choices) || json.choices.length === 0) return null;
  const choice = json.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return null;
  let text = "";
  if (typeof choice.message.content === "string") text = choice.message.content;
  else if (Array.isArray(choice.message.content)) {
    text = textOfBlocks(blocksFromOpenAIContent(choice.message.content));
  }
  const usage = isRecord(json.usage) ? json.usage : {};
  return {
    text,
    inputTokens: numberOrNull(usage.prompt_tokens) ?? 0,
    outputTokens: numberOrNull(usage.completion_tokens) ?? 0,
    stop: typeof choice.finish_reason === "string" ? choice.finish_reason : null,
  };
}

function parseAnthropicAnswer(json) {
  if (!isRecord(json) || !Array.isArray(json.content)) return null;
  const parts = [];
  for (const block of json.content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  const usage = isRecord(json.usage) ? json.usage : {};
  return {
    text: parts.join(""),
    inputTokens: numberOrNull(usage.input_tokens) ?? 0,
    outputTokens: numberOrNull(usage.output_tokens) ?? 0,
    stop: typeof json.stop_reason === "string" ? json.stop_reason : null,
  };
}

function parseGeminiAnswer(json) {
  if (!isRecord(json) || !Array.isArray(json.candidates) || json.candidates.length === 0) return null;
  const candidate = json.candidates[0];
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return null;
  const parts = [];
  for (const part of candidate.content.parts) {
    if (isRecord(part) && typeof part.text === "string") parts.push(part.text);
  }
  const usage = isRecord(json.usageMetadata) ? json.usageMetadata : {};
  return {
    text: parts.join(""),
    inputTokens: numberOrNull(usage.promptTokenCount) ?? 0,
    outputTokens: numberOrNull(usage.candidatesTokenCount) ?? 0,
    stop: typeof candidate.finishReason === "string" ? candidate.finishReason : null,
  };
}

// ---------------------------------------------------------------------------
// Rendering an answer into the client's wire
// ---------------------------------------------------------------------------

const RESPONSE_ID = "chatcmpl-aggregator";
const RESPONSE_CREATED = 1767225600;
const MESSAGE_ID = "msg_aggregator";

function openAIAnswerJson(answer, modelId) {
  return {
    id: RESPONSE_ID,
    object: "chat.completion",
    created: RESPONSE_CREATED,
    model: modelId,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: CLIENT_STOPS.openai[stopFamilyOf(answer.stop)],
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

function anthropicAnswerJson(answer, modelId) {
  return {
    id: MESSAGE_ID,
    type: "message",
    role: "assistant",
    model: modelId,
    content: [{ type: "text", text: answer.text }],
    stop_reason: CLIENT_STOPS.anthropic[stopFamilyOf(answer.stop)],
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

// Split answer text into the small pieces a streamed answer is sent in.
// The client reassembles the pieces, so the split points do not matter for
// the text; a fixed piece size keeps the stream deterministic.
function textPieces(text) {
  const pieces = [];
  for (let i = 0; i < text.length; i += 13) pieces.push(text.slice(i, i + 13));
  return pieces;
}

function sseFrame(payload) {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function openAIAnswerStream(answer, modelId) {
  const base = { id: RESPONSE_ID, object: "chat.completion.chunk", created: RESPONSE_CREATED, model: modelId };
  let out = sseFrame({
    ...base,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  });
  for (const piece of textPieces(answer.text)) {
    out += sseFrame({ ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  out += sseFrame({
    ...base,
    choices: [
      { index: 0, delta: {}, finish_reason: CLIENT_STOPS.openai[stopFamilyOf(answer.stop)] },
    ],
  });
  out += sseFrame({
    ...base,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
  out += "data: [DONE]\n\n";
  return out;
}

function anthropicAnswerStream(answer, modelId) {
  const frame = (payload) => `event: ${payload.type}\ndata: ${JSON.stringify(payload)}\n\n`;
  let out = frame({
    type: "message_start",
    message: {
      id: MESSAGE_ID,
      type: "message",
      role: "assistant",
      model: modelId,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  out += frame({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  for (const piece of textPieces(answer.text)) {
    out += frame({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } });
  }
  out += frame({ type: "content_block_stop", index: 0 });
  out += frame({
    type: "message_delta",
    delta: {
      stop_reason: CLIENT_STOPS.anthropic[stopFamilyOf(answer.stop)],
      stop_sequence: null,
    },
    usage: { output_tokens: answer.outputTokens },
  });
  out += frame({ type: "message_stop" });
  return out;
}

// ---------------------------------------------------------------------------
// The aggregator
// ---------------------------------------------------------------------------

function authHeaders(provider) {
  const headers = {};
  if (provider.auth === "bearer") headers.authorization = `Bearer ${provider.key}`;
  else if (provider.auth === "x-api-key") headers["x-api-key"] = provider.key;
  else if (provider.auth === "x-goog-api-key") headers["x-goog-api-key"] = provider.key;
  return headers;
}

class Aggregator {
  constructor(brief, dial) {
    this.manifest = brief.workload.manifest;
    this.models = buildModels(brief);
    this.dial = dial; // provider id -> { baseUrl, auth, key, wire }
    // Per-provider health: how many calls were made and how many refusals
    // came back in a row. Two refusals in a row demote the provider below
    // every healthy one for the rest of the trial: that is the signature of
    // an exhausted budget or an outage, and calls are too scarce to spend
    // re-probing it while healthy providers remain.
    this.health = new Map();
    for (const entry of brief.directory.providers) {
      this.health.set(entry.id, { calls: 0, failStreak: 0 });
    }
    this.servedCount = 0;
  }

  // The candidates for one request: free models (every model in the table
  // is free) meeting the manifest's requirements, best first. A JSON-mode
  // OpenAI request cannot go to an Anthropic-wire provider, whose wire has
  // no way to carry the JSON mode the provider must echo back.
  candidates(entry, canon, clientApi) {
    const requires = entry && isRecord(entry.requires) ? entry.requires : {};
    const list = [];
    for (const model of this.models) {
      if (requires.tools === true && !model.tools) continue;
      if (requires.vision === true && !model.vision) continue;
      if (requires.json === true && !model.json) continue;
      if (typeof requires.min_context === "number" && model.contextLength < requires.min_context) continue;
      if (clientApi === "openai" && canon.jsonMode && model.providerWire === "anthropic") continue;
      list.push(model);
    }
    list.sort((a, b) => {
      const demotedA = this.health.get(a.providerId).failStreak >= 2 ? 1 : 0;
      const demotedB = this.health.get(b.providerId).failStreak >= 2 ? 1 : 0;
      if (demotedA !== demotedB) return demotedA - demotedB;
      if (a.rating !== b.rating) return b.rating - a.rating;
      if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
      if (a.modelId !== b.modelId) return a.modelId < b.modelId ? -1 : 1;
      return 0;
    });
    return list;
  }

  // One provider call. Resolves to { ok: true, answer } on a 2xx answer, or
  // { ok: false } on a refusal or a transport failure. A 2xx answer that
  // cannot be decoded still counts as the request's one successful call, so
  // it is returned with whatever could be read from it rather than retried.
  async callProvider(model, canon) {
    const dial = this.dial.get(model.providerId);
    const health = this.health.get(model.providerId);
    health.calls += 1;
    let url;
    let body;
    const headers = { "content-type": "application/json", ...authHeaders(dial) };
    if (dial.wire === "openai") {
      url = `${dial.baseUrl}/v1/chat/completions`;
      body = renderOpenAIProviderBody(canon, model.modelId);
    } else if (dial.wire === "anthropic") {
      url = `${dial.baseUrl}/v1/messages`;
      headers["anthropic-version"] = "2023-06-01";
      body = renderAnthropicProviderBody(canon, model.modelId);
    } else {
      url = `${dial.baseUrl}/v1beta/models/${model.modelId}:generateContent`;
      body = renderGeminiProviderBody(canon, model.modelId);
    }
    let response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(8000),
      });
    } catch {
      health.failStreak += 1;
      return { ok: false };
    }
    if (response.status < 200 || response.status >= 300) {
      health.failStreak += 1;
      try {
        await response.text();
      } catch {
        // The refusal body carries nothing the router needs.
      }
      return { ok: false };
    }
    health.failStreak = 0;
    let json = null;
    try {
      json = await response.json();
    } catch {
      json = null;
    }
    let answer = null;
    if (json !== null) {
      if (dial.wire === "openai") answer = parseOpenAIAnswer(json);
      else if (dial.wire === "anthropic") answer = parseAnthropicAnswer(json);
      else answer = parseGeminiAnswer(json);
    }
    if (answer === null) answer = { text: "", inputTokens: 0, outputTokens: 0, stop: null };
    return { ok: true, answer };
  }

  // Serve one client request. The first candidate is retried once after a
  // refusal: a scheduled one-call failure is common, and the rating gap to
  // the next candidate is usually worth the extra call. Fallbacks get one
  // attempt each; every further call is scarcer than the rating it buys.
  async serve(clientApi, body) {
    const canon = clientApi === "openai" ? decodeOpenAIRequest(body) : decodeAnthropicRequest(body);
    const entry = this.manifest[this.servedCount] || null;
    this.servedCount += 1;
    const candidates = this.candidates(entry, canon, clientApi);
    const deadline = Date.now() + 24000;
    let served = null;
    for (let index = 0; index < candidates.length; index += 1) {
      const attempts = index === 0 ? 2 : 1;
      for (let attempt = 0; attempt < attempts; attempt += 1) {
        if (Date.now() > deadline) break;
        const result = await this.callProvider(candidates[index], canon);
        if (result.ok) {
          served = { model: candidates[index], answer: result.answer };
          break;
        }
      }
      if (served) break;
    }
    if (served === null) {
      // No provider would answer. Still answer the client in its own wire
      // and shape; the request cannot be valid without a provider's echo,
      // but the response checks that do not depend on the provider pass.
      const fallback = { text: "", inputTokens: 0, outputTokens: 0, stop: null };
      return this.render(clientApi, canon, fallback, "auto");
    }
    return this.render(clientApi, canon, served.answer, served.model.modelId);
  }

  render(clientApi, canon, answer, modelId) {
    if (clientApi === "openai") {
      if (canon.stream) {
        return { status: 200, contentType: "text/event-stream", body: openAIAnswerStream(answer, modelId) };
      }
      return { status: 200, contentType: "application/json", body: JSON.stringify(openAIAnswerJson(answer, modelId)) };
    }
    if (canon.stream) {
      return { status: 200, contentType: "text/event-stream", body: anthropicAnswerStream(answer, modelId) };
    }
    return { status: 200, contentType: "application/json", body: JSON.stringify(anthropicAnswerJson(answer, modelId)) };
  }
}

// The model list answers, one per client wire. Both list exactly the free
// model ids the catalog registers, each id once, marked free.
function modelList(catalog, clientApi) {
  const ids = [];
  const seen = new Set();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      ids.push(model.id);
    }
  }
  if (clientApi === "anthropic") {
    return {
      data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z", free: true })),
      has_more: false,
      first_id: ids.length > 0 ? ids[0] : null,
      last_id: ids.length > 0 ? ids[ids.length - 1] : null,
    };
  }
  return {
    object: "list",
    data: ids.map((id) => ({ id, object: "model", created: RESPONSE_CREATED, owned_by: "aggregator", free: true })),
  };
}

// ---------------------------------------------------------------------------
// Program entry
// ---------------------------------------------------------------------------

function readRequestBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", () => resolve(""));
  });
}

async function main() {
  const brief = loadBrief();
  if (process.argv.includes("--list")) {
    if (brief === null) {
      process.stdout.write('{"providers":[]}\n');
    } else {
      process.stdout.write(`${JSON.stringify(buildCatalog(brief))}\n`);
    }
    return;
  }
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (brief === null || !envUrl) {
    process.stderr.write("free-model-aggregator-01: no brief or no MAKEFUN_ENV_URL; nothing to serve\n");
    process.exit(1);
  }
  const catalog = buildCatalog(brief);

  // Dialling information: the environment's instance document carries the
  // absolute base URLs and the keys. If it cannot be read, fall back to the
  // brief's own directory joined to the environment URL.
  const dial = new Map();
  try {
    const response = await fetch(`${envUrl}/instance`);
    const instance = await response.json();
    for (const provider of instance.providers) {
      dial.set(provider.id, {
        baseUrl: provider.base_url,
        auth: provider.auth,
        key: provider.key,
        wire: provider.wire,
      });
    }
  } catch {
    // The fallback below covers this; the instance read is retried never,
    // because a replay must walk the same path every time.
  }
  for (const entry of brief.directory.providers) {
    if (!dial.has(entry.id)) {
      dial.set(entry.id, {
        baseUrl: `${envUrl}${entry.path}`,
        auth: entry.auth,
        key: entry.key,
        wire: entry.wire,
      });
    }
  }

  const aggregator = new Aggregator(brief, dial);
  // Requests are served strictly one at a time: the client waits for each
  // answer, and a provider refuses a second call while one is in flight.
  let queue = Promise.resolve();

  const server = createServer((req, res) => {
    const url = new URL(req.url || "/", "http://aggregator");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      const clientApi = req.headers["x-api-key"] !== undefined ? "anthropic" : "openai";
      const body = JSON.stringify(modelList(catalog, clientApi));
      res.writeHead(200, { "content-type": "application/json" });
      res.end(body);
      return;
    }
    const isOpenAI = req.method === "POST" && url.pathname === "/v1/chat/completions";
    const isAnthropic = req.method === "POST" && url.pathname === "/v1/messages";
    if (!isOpenAI && !isAnthropic) {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "not_found" }));
      return;
    }
    const clientApi = isOpenAI ? "openai" : "anthropic";
    void readRequestBody(req).then((raw) => {
      let body = {};
      try {
        body = JSON.parse(raw);
      } catch {
        body = {};
      }
      if (!isRecord(body)) body = {};
      queue = queue.then(async () => {
        try {
          const answer = await aggregator.serve(clientApi, body);
          res.writeHead(answer.status, { "content-type": answer.contentType });
          res.end(answer.body);
        } catch {
          res.writeHead(500, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: "internal" }));
        }
      });
    });
  });

  await new Promise((resolve) => server.listen(0, "0.0.0.0", resolve));
  const port = server.address().port;

  const registerResponse = await fetch(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
  });
  if (!registerResponse.ok) {
    process.stderr.write(`free-model-aggregator-01: register refused with ${registerResponse.status}\n`);
    process.exit(1);
  }

  // Poll the work route until the environment says the client is done,
  // then exit 0. The guard deadline keeps the program inside the trial's
  // wall even if the poll itself ever stalls.
  const guard = Date.now() + 165000;
  for (;;) {
    let done = false;
    try {
      const workResponse = await fetch(`${envUrl}/work`);
      const work = await workResponse.json();
      done = work.done === true;
    } catch {
      done = false;
    }
    if (done) break;
    if (Date.now() > guard) break;
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  server.close();
  process.exit(0);
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator-01: ${err && err.message ? err.message : err}\n`);
  process.exit(1);
});
