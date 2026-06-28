from pydantic.deprecated import tools
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS, prompt_template=PROMPT_TEMPLATE, course="llm-zoomcamp", model="openai/gpt-oss-120b:free"):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions 
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    # Index Searching from course json
    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query=query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )
    
    # Context formatting for the search results
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()


    #  Prompt formatting from search results
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    
    # The LLM methods that sends prompts to the LLM
    def llm(self, prompt):
        
        # making the llm agentic and be able to call functions
        search_tool = {
            "type": "function",
            "name": "search",
            "description": "search the FAQ database for entries matching teh query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "decsription": "search query text to look up in the course FAQ"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }

        input_message = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_message,
            tools=[search_tool]
        )

        return response.output
        # return response.output_text


    # The RAG method that wires all the  methods together
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)

        

        return answer