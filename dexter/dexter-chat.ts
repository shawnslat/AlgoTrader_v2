import 'dotenv/config';
import { Agent } from './src/agent/agent.ts';
import { MessageHistory } from './src/utils/message-history.ts';

function getQuery(): string {
  const envQ = process.env.DEXTER_QUERY;
  if (envQ && envQ.trim()) return envQ.trim();
  const argvQ = process.argv.slice(2).join(' ').trim();
  if (argvQ) return argvQ;
  return 'Give a concise answer.';
}

async function run() {
  if (!process.env.OPENAI_API_KEY) {
    console.log(JSON.stringify({ error: 'Missing OPENAI_API_KEY (set dexter/.env or environment).' }));
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

  const trimmed = output.trim();
  if (!trimmed) {
    console.log(JSON.stringify({ error: 'No answer returned (check stderr for provider/tool errors).' }));
    return;
  }
  console.log(JSON.stringify({ answer: trimmed }));
}

run().catch((err) => {
  console.log(JSON.stringify({ error: String((err as any)?.message || err) }));
});
