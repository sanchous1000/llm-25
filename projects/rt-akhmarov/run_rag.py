import os
from langfuse import Langfuse, Evaluation
from src.rag import RAGService
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self):
        self.llm = ChatOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="sk-no-key-required",
        model="llama-3.2-3b-instruct",
        temperature=0.05,
        max_tokens=64
        )

    def is_relevant(self, question, chunk):
        prompt = f"""
        Task: Grade the relevance of a retrieved document chunk to a user's question.
        Question: {question}
        Chunk: {chunk}
        
        Is this chunk relevant and helpful for answering the question? 
        Respond only with 'YES' or 'NO'.
        """
        response = self.llm.invoke(prompt).content.strip().upper()
        return 1.0 if "YES" in response else 0.0

judge = LLMJudge()

def rag_evaluator(input, output, expected_output, **kwargs):
    eval_results = []
    question = input 
    retrieved_docs = output.get("retrieved_contexts", [])
    
    relevance_scores = [judge.is_relevant(question, doc) for doc in retrieved_docs]
    
    k = len(relevance_scores)
    if k == 0:
        return []

    p_at_k = sum(relevance_scores) / k
    eval_results.append(Evaluation(name="precision_at_k", value=p_at_k))

    r_at_k = 1.0 if sum(relevance_scores) > 0 else 0.0
    eval_results.append(Evaluation(name="recall_at_k", value=r_at_k))

    mrr = 0.0
    for i, score in enumerate(relevance_scores):
        if score == 1.0:
            mrr = 1.0 / (i + 1)
            break
    eval_results.append(Evaluation(name="mrr", value=mrr))

    return eval_results

def accuracy_evaluator(input, output, expected_output, **kwargs):
    if not expected_output:
        return []
    
    generated_answer = output.get("answer", "")
    
    judge_prompt = f"""
    Task: Is the Generated Answer factually correct based on the Ground Truth?
    Ground Truth: {expected_output}
    Generated Answer: {generated_answer}

    Analyze and compare them. Be strict. If the answer is correct information, reply '1'. If it is wrong or missing info, reply '0'.
    Reply ONLY with the number.
    """
    
    result = judge.llm.invoke(judge_prompt).content.strip()
    score = 1.0 if result == "1" else 0.0
    
    return [Evaluation(name="accuracy", value=score)]


def run_experiment():
    config_name = "test_rag_experiment-latest"
    rag_params = {
        "chunk_size": 512,       
        "embedding": "e5-large", 
        "top_k": 3            
    }

    langfuse = Langfuse()
    
    rag = RAGService(top_k=rag_params["top_k"])
    
    dataset_name = "verl-aq-v2"
    print(f"📥 Loading dataset '{dataset_name}'...")
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Error getting dataset: {e}")
        return

    print(f"🧪 Running Experiment: {config_name}")

    _ = dataset.run_experiment(
        name=f"rag_exp_{config_name}-no_rag",
        task=rag.answer_without_rag,
        evaluators=[accuracy_evaluator], 
        metadata=rag_params,  
        max_concurrency=5,      
        description="Testing accuracy without retrieval"
    )
    
    _ = dataset.run_experiment(
        name=f"rag_exp_{config_name}-with_rag",
        task=rag.answer_question,
        evaluators=[rag_evaluator, accuracy_evaluator], 
        metadata=rag_params,  
        max_concurrency=5,      
        description="Testing retrieval metrics recall/mrr"
    )

    langfuse.flush()
    print("🏁 Experiment finished! Check 'Datasets' tab in Langfuse.")
    print("👉 Use 'Compare' button in UI to see how this config performs against others.")

if __name__ == "__main__":
    run_experiment()