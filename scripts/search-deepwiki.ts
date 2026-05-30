import { randomUUID } from "crypto";

const query = process.argv[2] || "Explain the database schema of OpenCouncil";
console.log(`Searching DeepWiki for: "${query}"...\n`);

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

const uuid = randomUUID();
const querySlug = `${slugify(query.substring(0, 50))}_${uuid}`;

const postUrl = "https://api.devin.ai/ada/query";
const wsUrl = `wss://api.devin.ai/ada/ws/query/${querySlug}`;

const payload = {
  mode: "deep",
  user_query: query,
  keywords: [],
  repo_names: ["schemalabz/opencouncil"],
  additional_context: "",
  query_id: querySlug,
  use_notes: false,
  generate_summary: false,
  source: "ada.deepwiki_public"
};

// 1. Submit the query via HTTP POST
const response = await fetch(postUrl, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Origin": "https://deepwiki.com"
  },
  body: JSON.stringify(payload)
});

if (!response.ok) {
  console.error("Failed to submit query", await response.text());
  process.exit(1);
}

console.log("Query submitted successfully. Opening WebSocket connection...\n");

// 2. Open WebSocket connection
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
  console.log("WebSocket connected. Streaming results:\n");
};

socket.onmessage = (event) => {
  try {
    const frame = JSON.parse(event.data);
    
    if (frame.type === "chunk" && frame.data) {
      process.stdout.write(frame.data);
    } else if (frame.type === "tool_call_start") {
      console.log(`\n\n[Devin running tool: ${frame.tool_name}]`);
    } else if (frame.type === "tool_call_complete") {
      console.log(`[Devin finished running tool: ${frame.tool_name}]`);
    } else if (frame.type === "thoughts_start") {
      console.log("\n[Devin thinking...]");
    } else if (frame.type === "thoughts_end") {
      console.log("[Devin stopped thinking]");
    }
  } catch (err) {
    // Raw frame or non-JSON
    console.log(event.data);
  }
};

socket.onclose = () => {
  console.log("\n\nStream closed.");
  process.exit(0);
};

socket.onerror = (error) => {
  console.error("\nWebSocket error:", error);
  process.exit(1);
};
