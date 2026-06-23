import os 
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from ingest import load_faq_data, build_index
# pyrefly: ignore [missing-import]
from rag_helper import RAGBase
from openai import OpenAI

load_dotenv()

documents = load_faq_data()
index = build_index(documents)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

assistant = RAGBase(index=index, llm_client=openai_client)

answer = assistant.rag("I just discovered this course is it available for me to join now?")
print(answer)


# can override system instruction if we want to
custom_instructions = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
""".strip()

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
    instructions=custom_instructions,
)

another_answer = assistant.rag("Would i get the certificate if i join the course late?")
print(another_answer)