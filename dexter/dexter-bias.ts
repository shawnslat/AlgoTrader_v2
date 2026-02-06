import 'dotenv/config';
import { Agent } from './src/agent/agent.ts';
import { MessageHistory } from './src/utils/message-history.ts';

function getTickers(): string[] {
  const raw = process.env.DEXTER_TICKERS || '';
  return raw
    .split(',')
    .map((t) => t.trim().toUpperCase())
    .filter(Boolean);
}

function buildPrompt(tickers: string[]): string {
  const list = tickers.join(', ');
  return `You are helping manage a small trading account with limited trades/day. 

For the following tickers: ${list}

Return JSON only (no markdown) in this exact shape:
{
  "bias": {
    "AAPL": { "decision": "ok" | "avoid", "reason": "short reason" },
    ...
  }
}

Rules:
- Mark "avoid" if there is an imminent high-risk event (earnings in next 5 days), major negative catalyst, or unusually elevated volatility risk.
- Mark "ok" otherwise.
- Keep reasons short, 1 sentence each.
`;
}

async function run() {
  if (!process.env.OPENAI_API_KEY) {
    console.log(JSON.stringify({ error: 'Missing OPENAI_API_KEY (set dexter/.env or environment).' }));
    return;
  }

  const tickers = getTickers();
  if (!tickers.length) {
    console.log(JSON.stringify({ error: 'Missing DEXTER_TICKERS env var.' }));
    return;
  }

  const query = buildPrompt(tickers);

  let doneResolve: (() => void) | null = null;
  const done = new Promise<void>((resolve) => (doneResolve = resolve));
  let output = '';

  const agent = new Agent({
    model: process.env.DEXTER_MODEL || 'gpt-4.1-mini',
    callbacks: {
      onAnswerStream: (stream) => {
        (async () => {
          try {
            for await (const chunk of stream) output += chunk;
          } finally {
            doneResolve?.();
          }
        })();
      },
    },
  });

  await agent.run(query, new MessageHistory());
  await done;

  const trimmed = output.trim();
  if (!trimmed) {
    console.log(JSON.stringify({ error: 'No output returned.' }));
    return;
  }

  try {
    const parsed = JSON.parse(trimmed);
    console.log(JSON.stringify(parsed));
  } catch {
    console.log(JSON.stringify({ error: 'Non-JSON response returned.' }));
  }
}

run().catch((err) => {
  console.log(JSON.stringify({ error: String((err as any)?.message || err) }));
});

