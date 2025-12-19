"""Wikipedia retrieval tool for factual grounding."""
from __future__ import annotations
from typing import Any, Dict
from .base import BaseBrainTool

try:
    import wikipedia
except ImportError:
    wikipedia = None


class WikipediaTool(BaseBrainTool):
    """Retrieves summaries from Wikipedia."""

    def __init__(self):
        super().__init__(
            name="wikipedia",
            description="Search Wikipedia for information on a topic and return a summary."
        )

    async def execute(self, query: str, sentences: int = 3) -> Any:
        """Execute the search."""
        if not wikipedia:
            return {"error": "wikipedia package not installed. Cannot search."}
        
        try:
            # Set language to english
            wikipedia.set_lang("en")
            summary = wikipedia.summary(query, sentences=sentences)
            return {
                "query": query,
                "summary": summary,
                "source": "wikipedia"
            }
        except wikipedia.exceptions.DisambiguationError as e:
            return {
                "error": "Query is ambiguous. Possible topics found.",
                "options": e.options[:5],
                "query": query
            }
        except wikipedia.exceptions.PageError:
            return {"error": "No page found for query.", "query": query}
        except Exception as e:
            return {"error": f"Search error: {str(e)}", "query": query}

    def _get_params_spec(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or topic."
                },
                "sentences": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of sentences to return in the summary."
                }
            },
            "required": ["query"]
        }
