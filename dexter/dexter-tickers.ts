import 'dotenv/config';
import { Agent } from './src/agent/agent.ts';
import { MessageHistory } from './src/utils/message-history.ts';

function getQuery(): string {
  return (
    process.env.DEXTER_QUERY ||
    `Suggest 5–8 liquid US tickers for 1–5 day swing longs today. Avoid earnings within 5 days, low volume, and obvious negative catalysts. Output JSON: {"tickers":[...]} only.`
  );
}

async function run() {
  if (!process.env.OPENAI_API_KEY) {
    console.log(JSON.stringify({ tickers: [], error: 'Missing OPENAI_API_KEY (set dexter/.env or environment).' }));
    return;
  }

  const query = getQuery();

  let answerStreamResolve: (() => void) | null = null;
  const answerStreamDone = new Promise<void>((resolve) => (answerStreamResolve = resolve));

  let output = '';
  const agent = new Agent({
    model: process.env.DEXTER_MODEL || 'gpt-4.1-mini',
    callbacks: {
      onAnswerStream: (stream) => {
        (async () => {
          try {
            for await (const chunk of stream) output += chunk;
          } finally {
            answerStreamResolve?.();
          }
        })();
      },
    },
  });

  await agent.run(query, new MessageHistory());
  await answerStreamDone;

  try {
    const parsed = JSON.parse(output);
    const tickers = parsed?.tickers && Array.isArray(parsed.tickers) ? parsed.tickers : [];
    console.log(JSON.stringify({ tickers }));
  } catch {
    const trimmed = output.trim();
    console.log(JSON.stringify({ tickers: [], error: trimmed ? 'Non-JSON response returned.' : 'No output returned.' }));
  }
}

run().catch((err) => {
  console.log(JSON.stringify({ tickers: [], error: String((err as any)?.message || err) }));
});
