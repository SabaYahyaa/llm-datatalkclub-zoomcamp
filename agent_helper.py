from smolagents import CodeAgent, LiteLLMModel, Tool
from langchain_core.messages import HumanMessage, AIMessage
import re


#########################################################
#   Agent with memory
#########################################################
class FAQSearchTool(Tool):
    name = "text_search_tuned"
    description = (
        "Searches the LLM Zoomcamp FAQ database using optimized keyword weights. "
        "Use this for any course registration, schedules, or prerequisite questions."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The simple search keywords or queries to pass into the FAQ database."
        }
    }

    output_type = "string"

    def __init__(self, search_index, **kwargs):
        super().__init__(**kwargs)
        self.search_index = search_index

    def forward(self, query: str) -> str:

        boost_dict = {
            "question": 3.0,
            "section": 0.5,
            "answer": 10.0,
        }

        filter_dict = {
            "course": "llm-zoomcamp"
        }
        search_results = self.search_index.search(
            query,
            num_results=5,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )
        lines = []

        for doc in search_results:
            lines.append(f"Section: {doc['section']}")
            lines.append(f"Q: {doc['question']}")
            lines.append(f"A: {doc['answer']}\n")
        return "\n".join(lines).strip() if lines else "No matching FAQ entries found."


class FAQAgent:
    def __init__(self, search_index):
        # Memory
        self.chat_history = []
    
        # Model
        self.model = LiteLLMModel(
            model_id="ollama_chat/qwen2.5:3b-instruct",
            api_base="http://127.0.0.1:11434"
        )
        # Tool
        self.faq_tool = FAQSearchTool(
            search_index=search_index
        )
        # Agent
        self.agent = CodeAgent(
            model=self.model,
            tools=[self.faq_tool],
            max_steps=5,
            verbosity_level=0
        )

    def get_history(self):

        if not self.chat_history:
            return "No previous conversation."

        history = []
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                history.append(
                    f"User: {message.content}"
                )
            elif isinstance(message, AIMessage):
                history.append(
                    f"Assistant: {message.content}"
                )
        return "\n".join(history)
    
    def clear_memory(self):
        """I want to clear memory after each query to reduce the cost"""
        self.chat_history = []

    def update_memory(self, question, answer):
        self.chat_history.append(
            HumanMessage(content=question)
        )
        self.chat_history.append(
            AIMessage(content=answer)
        )
        # keep last 10 messages
        self.chat_history = self.chat_history[-10:]

    def run(self, question):
        history = self.get_history()
        task_prompt = f"""
You are an FAQ assistant for LLM Zoomcamp.
Conversation history:
{history}

You have one tool:
text_search_tuned(query)

Follow these steps:
1. Call text_search_tuned with the user's question.
2. Read the returned FAQ information.
3. Answer ONLY using the returned information.
4. Do not copy the search results.
5. Keep the answer concise.

IMPORTANT:
Your final output MUST be inside a code block exactly like this:

<code>
final_answer("your answer here")
</code>

Never write final_answer outside the code block.

User question:
{question}
"""
        result = self.agent.run(
            task_prompt,
            return_full_result=True
        )
        answer = result.output

        # Update memory
        self.update_memory(
            question,
            answer
        )
        return result
    

def get_query_cost_by_agent_with_memeory(agent_result):
    usage_agent_result = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for step in agent_result.steps:
        if isinstance(step, dict):
            step_usage = step.get("token_usage")
            if step_usage:
                for key in usage_agent_result:
                    usage_agent_result[key] += step_usage.get(key, 0)
    return(usage_agent_result)



def get_tool_call_by_agent_with_memeory(agent_result):
    tool_calls = []

    for step in agent_result.steps:
        if not isinstance(step, dict):
            continue
        code = step.get("code_action")

        if not code:
            continue
        matches = re.findall(
            r'(\w+)\((.*?)\)',
            code,
            re.DOTALL
        )
        for name, args in matches:
            if name != "print":
                tool_calls.append(
                    {
                        "tool": name,
                        "arguments": args
                    }
                )
        return tool_calls