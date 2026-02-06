import { TavilySearch } from '@langchain/tavily';

// Avoid constructing Tavily tool unless API key is present.
// The wrapper throws at construction time if TAVILY_API_KEY is missing.
export const tavilySearch = process.env.TAVILY_API_KEY
  ? new TavilySearch({ maxResults: 5 })
  : (null as unknown as TavilySearch);
