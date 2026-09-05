import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.embeddings import embed_text
from app.generation import generate_answer


async def main() -> None:
    vec = await embed_text("The quick brown fox jumps over the lazy dog.")
    print("embedding length:", len(vec))
    print("embedding sample:", vec[:5])

    answer, thinking = await generate_answer("In one sentence, what is the capital of France?")
    print()
    print("answer:", repr(answer))
    print("has <think> tag in answer:", "<think>" in answer)
    print("thinking (first 150 chars):", repr(thinking[:150]))


if __name__ == "__main__":
    asyncio.run(main())
