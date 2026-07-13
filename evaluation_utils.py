
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import textwrap
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from  pydantic import BaseModel
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from langchain_core.prompts import (
    ChatPromptTemplate,  # to set the prompt and specify the system, human, ai, and developper
    MessagesPlaceholder, # to keep the conversion on and take care from the previous conversion
)
from rag_helper import RAGBase


MODEL="gemma3:4b"
MAX_RETRIES = 3
MAX_WORKERS = 4
Question_INSTRUCTION = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()


class Questions(BaseModel):
    """
        I want the model llm to return sth like this: { "questions": [..., ..., ]}
        a key questions that has list value, each list has n elements
    """
    questions: list[str]


def llm_structured(instructions, user_prompt, output_type=Questions, model="gemma3:4b"):
    messages = [
    SystemMessage(content=instructions),
    HumanMessage(content=user_prompt)
    ]

    llm_model = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature = 0,
        )
    structured_llm_model = llm_model.with_structured_output(
                                        output_type, 
                                        include_raw=True, # to get the metadata including the input and output tokens
    )
    result = structured_llm_model.invoke(messages)

    # return result['parsed'].questions, result['raw'].usage_metadata
    parsed_output = result.get('parsed')
    raw_message = result.get('raw')
    
    usage_metadata = {}
    if raw_message and hasattr(raw_message, 'usage_metadata'):
        usage_metadata = raw_message.usage_metadata

    return parsed_output, usage_metadata


def llm_structured_retry(
    instructions,
    user_prompt,
    output_type=Questions,
    model=MODEL,
    max_retries=MAX_RETRIES,
):
    """if there is an error form a document, just retry 3 times if the error still available, just return None"""
    for attempt in range(max_retries):
        try:
            possible_questions, token_usage = llm_structured(
                instructions=instructions,
                user_prompt=user_prompt,
                output_type=output_type,
                model=model,
            )
            return (possible_questions, token_usage)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def generate_ground_truth(doc):
    """
        Prepare ground truth, 
        for each document find 5 questions and usage_token, then
        for each question attach the document id (report in a dictionary)
    """
    user_prompt = json.dumps(doc)
    quesion_with_metadata = dict()
    result = list()
    questions = list()
    # apply the llm_structed
    questions, usage_token = llm_structured_retry(
        instructions=Question_INSTRUCTION ,
        user_prompt=user_prompt,
        output_type=Questions, 
        model=MODEL,
        max_retries=MAX_RETRIES,
        )
    if questions:
        for question in questions:
            quesion_with_metadata = dict()
            quesion_with_metadata["question"] = question
            quesion_with_metadata["document"] = doc['id']
            result.append(quesion_with_metadata)
        return result, usage_token
    return None, None



def run_in_parallel(func, items, max_workers=MAX_WORKERS, desc="Processing"):
    """Process 4 documents in parallel to save time"""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(func, item)
            for item in items
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
        ):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Unexpected error: {e}")
                # results.append(None)
                continue
    return results


def compute_relevance(ground_truth_document, search_function):
    """
        ground_truth_document is a document that contains document_id and question
        we check if the search_function is able to fetch the same document_id of the ground_truth_document
        If the search_function fetches the document we report the position (we set 1 to the position)
    """
    doc_id = ground_truth_document["document"]
    results = search_function(query=ground_truth_document["question"])

    relevance = []
    for d in results:
        relevance.append(int(d["id"] == doc_id))
    return relevance


def compute_relevance_total(ground_truth, search_function):
    """find the relevance matrix for all the docuements in the ground_truth"""
    relevance_total = list()

    for query_with_id in ground_truth:
        relevance = compute_relevance(query_with_id, search_function=search_function)
        relevance_total.append(relevance)

    return relevance_total


def hit_rate(relavent_matrix, num_results=5):
    """Hit Rate (when there is 1 in a line, means the search_index is able to fetch the documents)
        so the hit rate is the percentage of fetching documents
    """
    hit_rate = sum([sum(line) for line in relavent_matrix])/len(relavent_matrix)
    # Note: some lines do not have 5 
    # relavent_matrix = [row + [0] * (num_results - len(row)) for row in relavent_matrix]
    # relavent_matrix = np.array(relavent_matrix)
    # hit_rate = np.sum(relavent_matrix)/len(relavent_matrix)

    # if we use 
    return (hit_rate)


def mean_reciprocal_rank_numpy(relavent_matrix, num_results=5):
    relavent_matrix = [row + [0] * (num_results - len(row)) for row in relavent_matrix]
    arr = np.asarray(relavent_matrix)
    
    # Find the column index of the first '1' for each row. 
    # If no '1' is found, argmax returns 0, so we must mask them out.
    has_hit = (arr == 1).any(axis=1)
    first_hit_ranks = (arr == 1).argmax(axis=1) + 1
    
    # Calculate reciprocal rank only where a hit exists, otherwise 0
    reciprocal_ranks = np.where(has_hit, 1.0 / first_hit_ranks, 0.0)
    
    return reciprocal_ranks.mean()


def mean_reciprocal_rank(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance)

def evaluate(ground_truth, search_function):
    relavent_matrix  = compute_relevance_total(ground_truth, search_function)

    return {
        "hit_rate": hit_rate(relavent_matrix ),
        "mean_reciprocal_rank": mean_reciprocal_rank(relavent_matrix ),
    }



class RAGWithUsage(RAGBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def search(self, query, question_boost=3.0, section_boost=0.5, answer_boost=10.0):
        """
        Updated search method using your tuned parameters.
        """
        boost_dict = {
            "question": question_boost,
            "section": section_boost,
            "answer": answer_boost,
        }
        filter_dict = {"course": self.course}
        
        return self.index.search(
            query,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=self.num_results
        )

    def get_chain(self):
        """
        Overriding get_chain to REMOVE StrOutputParser().
        This ensures chain.invoke() returns the full AIMessage object.
        """
        if self.model not in self.models:
            llm = ChatOllama(
                model=self.model,
                temperature=0.7,
                base_url="http://localhost:11434",
            )
            # Removed | StrOutputParser() so we get metadata
            self.models[self.model] = self.prompt | llm
            
        return self.models[self.model]

    def llm(self, query, context):
        # 1. Get the chain (without the string parser)
        chain = self.get_chain()

        # 2. Get full AIMessage response object
        response = chain.invoke({
            "context": context,
            "question": query,
            "chat_history": self.chat_history
        })

        # 3. Extract content and maintain history exactly like your code
        answer = response.content
        self.chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer)
        ])
        del self.chat_history[:-10]

        #  Extract token tracking numbers from the metadata
        metadata = response.usage_metadata or {}
        input_tokens = metadata.get("input_tokens", 0) 
        output_tokens = metadata.get("output_tokens", 0)

        # Update cumulative totals
        self.last_input_tokens = input_tokens
        self.last_output_tokens = output_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return response
    
    def get_last_tokens(self):
        """
        Dedicated method to retrieve only the input and output token 
        counts from the most recent LLM invocation.
        """
        return {
            "input_tokens": self.last_input_tokens,
            "output_tokens": self.last_output_tokens
        }
    def get_total_tokens(self):
            """
            Dedicated method to retrieve cumulative token usage 
            across the whole lifecycle of this class object.
            """
            return {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens
            }
    def rag(self, query):
        search_results = self.search(query)
        context = self.build_context(search_results).strip()
        response = self.llm(query=query, context=context)
        return response.content