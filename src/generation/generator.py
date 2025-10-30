import os
from openai import OpenAI

from src.retrieval.database import QdrantDatabase
from .prompts import SYSTEM_PROMPT


class Generator:
    def __init__(self, db: QdrantDatabase):
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        if not db:
            db = QdrantDatabase()

        self.client = OpenAI(api_key=api_key)
        self.db = db

    def generate_response(self, user_query: str) -> str:
        """
        Generates a response to a user query using RAG.
        """

        context_chunks = self.db.search(user_query)
        context_str = "\n---\n".join(context_chunks)

        user_prompt = f"""
        Context from 10-Q form:
        ---
        {context_str}
        ---
        Question: {user_query}
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content


if __name__ == '__main__':

    try:
        rag_generator = Generator(db=QdrantDatabase())

        # 3. Define a user query and generate a response
        query = "What are the company's cash and cash equivalents?"
        answer = rag_generator.generate_response(query)

        print("\n--- User Query ---")
        print(query)
        print("\n--- LLM Response ---")
        print(answer)

    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have set the OPENAI_API_KEY in your environment variables.")
