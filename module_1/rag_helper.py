
from langchain_ollama import ChatOllama 
from langchain_core.prompts import (
    ChatPromptTemplate,  # to set the prompt and specify the system, human, ai, and developper
    MessagesPlaceholder, # to keep the conversion on and take care from the previous conversion
)
from langchain_core.messages import (
    HumanMessage, # this is the user and we need to save the questions 
    AIMessage    # this the ai response, and we need to save it (also (save the anwsers from the AI))
)
from langchain_core.output_parsers import StrOutputParser  # set the output parser only string

COURSE = "llm-zoomcamp"

INSTRUCTIONS = """
You are a helpful course assistant.
Answer questions ONLY using the provided context.

If the answer is not in the context, say:
"I don't know."
"""

PROMPT_INPUT =  """
                    Context:
                    {context}.strip()

                    Question:
                    {question}
                    """



class RAGBase:
        def __init__(
                        self,
                        index,
                        model = "gemma3:4b",
                        num_results=5,
        ):
                self.index = index
                self.model =  model
                self.prompt = self.build_prompt_template()
                self.num_results = num_results
                self.models = {}
                self.chat_history = []
        
        def build_prompt_template(self):
                # set the prompt
                prompt = ChatPromptTemplate.from_messages([
                                ("system",INSTRUCTIONS), # ai assistant
                                MessagesPlaceholder(variable_name="chat_history"),  # takecare from the history
                                # Current question + retrieved context (you need to put the question and also the clean context)
                                ("human", PROMPT_INPUT)
                                ])
                return prompt


        def search(self, query):
                boost_dict = {"question": 2.0, "section": 0.5}
                filter_dict = {"course": COURSE}
                return self.index.search(
                        query,
                        boost_dict=boost_dict,
                        filter_dict=filter_dict,
                        num_results=self.num_results
        )


        def build_context(self, search_results):
                lines = []

                for doc in search_results:
                        lines.append(doc["section"])
                        lines.append("Q: " + doc["question"])
                        lines.append("A: " + doc["answer"])
                        lines.append("")
                return "\n".join(lines).strip()


        def get_chain(self):
                if self.model not in self.models:
                        llm = ChatOllama(
                                model=self.model,
                                temperature=0.7,
                                base_url="http://localhost:11434",
                        )
                        self.models[self.model] = (  # this is used to save the model and and we do not need to build it 
                                self.prompt
                                | llm
                                | StrOutputParser()
                        )
                return self.models[self.model]


        def llm(self,  query, context):
                # get the chain
                chain = self.get_chain()

                # get response
                response = chain.invoke({
                        "context": context,
                        "question":  query,
                        "chat_history": self.chat_history
                })


                self.chat_history.extend([ # Save conversation history
                                        HumanMessage(content=query),
                                        AIMessage(content=response)
                ])
                # chat_history = chat_history[-10:] # this is to keep only the last 10 chats, so that I do not need a lot of memory to save all the chat
                del self.chat_history[:-10]
                return response


        def rag(self, query):
                # get the context according to the question
                search_results = self.search(query)
                # clean the context to be suitable for llm
                context = self.build_context(search_results)
                answer = self.llm(query=query, context=context)
                return answer
        
# # Note if I want to change to openAI or HuggingFace, I can create another class that imort the parent class, then I can override the class, as the following
# class OpenAIRAG(RAGBase):
#         def llm(self):