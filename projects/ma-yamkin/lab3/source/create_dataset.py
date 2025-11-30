import os
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# Инициализация Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)


def create_sentiment_dataset():
    # 1. Создаем сам датасет
    dataset = langfuse.create_dataset(
        name="arxiv-dataset",
        description="Тестовые примеры для агента анализа тональности"
    )
    print(f"✅ Датасет создан: {dataset.name} (ID: {dataset.id})")

    # 2. Добавляем примеры
    test_cases = [
          {
              "input": "What is the main contribution of the proposed method in addressing data scarcity for Named Entity Recognition (NER) in low-resource settings?",
              "metadata": ["2510.17720v1_page_002"],
              "expected_output": "The main contribution is a paraphrase-based data augmentation framework called PANER, which preserves entity positions while rewriting only the surrounding context using a large language model (LLaMA 3.3-70B). This approach expands linguistic variety without breaking semantic relationships, improves entity recognition in domain-specific and low-resource settings, and is combined with an efficient instruction-tuning template that incorporates negative examples, definitions, and annotation guidelines."
          },
          {
              "input": "What advantages does the proposed PANER framework offer over existing zero-shot NER models like GNER and SLIMER in terms of computational efficiency and out-of-domain performance?",
              "metadata": ["2510.17720v1_page_008"],
              "expected_output": "PANER achieves competitive zero-shot performance (e.g., F1 = 0.648 with Falcon-3, close to GNER’s scores) using only one fine-tuning epoch with LoRA, whereas GNER requires three full fine-tuning epochs. In out-of-domain evaluation on the BUSTER dataset, PANER (F1 = 0.336) outperforms both GNER variants and approaches SLIMER (SOTA, F1 = 0.4527), while requiring significantly fewer computational resources."
          },
          {
              "input": "What is the primary purpose of the MT-Video-Bench benchmark introduced in the paper, and how does it differ from previous evaluation datasets for multimodal large language models (MLLMs)?",
              "metadata": ["2510.17722v1_page_002"],
              "expected_output": "MT-Video-Bench is the first holistic benchmark designed to evaluate MLLMs in multi-turn, video-grounded dialogues, assessing both perceptual and interactive capabilities across six dimensions (e.g., memory recall, topic shifting, proactive interaction). Unlike prior datasets that focus on single-turn factual perception, MT-Video-Bench emphasizes cross-scene reasoning, long-range dependencies, and interactive adaptability, using 987 dialogues and 5,805 QA pairs to better reflect real-world usage."
          },
          {
              "input": "How does the number of video frames affect model performance across different capabilities in the MT-Video-Bench evaluation, and what does this reveal about the role of visual context in multi-turn video dialogue reasoning?",
              "metadata": ["2510.17722v1_page_009"],
              "expected_output": "Performance varies by task: Topic Shifting is unaffected by frame count, indicating reliance on dialogue-level reasoning rather than visual detail. Answer Refusal improves with fewer frames, as models become more cautious and avoid hallucination. The other four capabilities (Object Reference, Memory Recall, Content Summary, Proactive Interaction) improve with more frames, showing that richer visual context supports accurate reasoning. This reveals that visual context is beneficial but task-dependent: excessive frames can harm tasks requiring uncertainty awareness, while aiding those needing detailed perception."
          },
          {
              "input": "What distinguishes AcademicEval from related frameworks like ResearchTown and WildLong in terms of its design goals, data sourcing, and approach to mitigating label leakage in long-context LLM evaluation?",
              "metadata": ["2510.17725v1_page_003"],
              "expected_output": "AcademicEval is a live, real-world evaluation benchmark based on authentic academic papers, designed to assess hierarchical academic writing (Title, Abstract, etc.) under leakage-resistant conditions. Unlike ResearchTown (a multi-agent simulation for research dynamics) and WildLong (a synthetic data generator for instruction tuning), AcademicEval uses a co-author graph to retrieve high-quality few-shot examples and performs periodic data updates to prevent training data contamination. This dynamic updating actively mitigates label leakage, a problem most static benchmarks suffer from."
          },
          {
              "input": "How do human-judged preferences for academic writing tasks (e.g., Title, Abstract, Introduction, Related Work) differ from automatic metrics like BERTScore and ROUGE-L, and what does this reveal about the strengths of RALM versus non-RALM models across different tasks?",
              "metadata": ["2510.17725v1_page_012"],
              "expected_output": "Human judges prioritize coherence, academic style, feasibility, and appropriate citation over lexical/semantic overlap. Consequently: Non-RALM models (e.g., Qwen, Mixtral) are preferred for Title, Abstract, and Introduction, where holistic quality matters. RALM models (e.g., LLaMA†) excel in Related Work, where retrieval of prior studies and domain terminology aligns with judged quality. This divergence shows that automatic metrics favor surface similarity, while human evaluation captures deeper academic writing competence, revealing that RALM’s strength is task-specific and not universally superior.The kernel enables edge-based neuromorphic processing without cloud dependency by introducing dynamic core assignment (DCA). It achieves 4× speed-up on moderate SNNs and 1.7× on Synfire networks, load-balances across all CPU cores (e.g., ARM Cortex), and is up to 70% more energy-efficient than static core assignment by leveraging CPU DVFS and minimizing idle cores—making it suitable for low-SWaP (Size, Weight, and Power) edge devices."
          },
          {
              "input": "What is the primary advantage of the proposed multi-threading kernel for Spiking Neural Networks (SNNs) in neuromorphic edge applications, and how does it improve upon previous implementations in terms of performance, energy efficiency, and core utilization?",
              "metadata": ["2510.17745v1_page_001"],
              "expected_output": "Human judges prioritize coherence, academic style, feasibility, and appropriate citation over lexical/semantic overlap. Consequently: Non-RALM models (e.g., Qwen, Mixtral) are preferred for Title, Abstract, and Introduction, where holistic quality matters. RALM models (e.g., LLaMA†) excel in Related Work, where retrieval of prior studies and domain terminology aligns with judged quality. This divergence shows that automatic metrics favor surface similarity, while human evaluation captures deeper academic writing competence, revealing that RALM’s strength is task-specific and not universally superior."
          },
          {
              "input": "How does the multi-threading kernel in CARLsim achieve performance gains on neuromorphic edge applications, and what are the observed speed-up factors for the Chainfire and Synfire networks when scaling across multiple CPU cores?",
              "metadata": ["2510.17745v1_page_003"],
              "expected_output": "The kernel parallelizes neuron state updates using numerical integration (Euler/RK4) across threads, with dynamic load balancing to match real-time constraints. For the Chainfire network (2000 neurons), it achieves up to 4.0× performance gain (16 threads vs. 1). For the Synfire network (1200 neurons, 77K synapses), it achieves 1.7× speed-up (4–8 threads), limited by the current focus on neuron computation rather than synaptic processing."
          },
          {
              "input": "How does the proposed approach in this study enhance the predictability of Sea Ice Concentration (SIC) and Sea Ice Velocity (SIV) compared to prior deep learning methods, and what role does the physics-informed training scheme play in improving the HIS-Unet framework?",
              "metadata": ["2510.17756v1_page_009"],
              "expected_output": "The study enhances predictability by adopting the multi-task HIS-Unet architecture and integrating a physics-informed machine learning (PIML) training scheme. Unlike prior CNN/LSTM models that rely solely on data-driven patterns, this approach embeds physical constraints (e.g., conservation laws, ice dynamics) into the loss function or training process, ensuring predictions adhere to known physical principles—thereby improving generalization, stability, and accuracy for both SIC and SIV forecasts."
          },
          {
              "input": "How do the Weighting Attention Modules (WAMs) facilitate information sharing between the Sea Ice Velocity (SIV) and Sea Ice Concentration (SIC) branches in the proposed multi-task U-Net architecture, and what role do channel and spatial attention mechanisms play within each WAM?",
              "metadata": ["2510.17756v1_page_017"],
              "expected_output": "Each WAM receives feature maps from both SIV and SIC branches, computes a learned weighted sum to create shared information, and refines it through channel attention (identifying informative feature channels) and spatial attention (highlighting relevant spatial regions). This two-stage attention allows the model to selectively exchange task-relevant features across branches at multiple scales (3 encoder + 3 decoder levels), improving joint prediction accuracy by capturing cross-task dependencies in both feature importance and location."
          }
    ]

    for case in test_cases:
        item = langfuse.create_dataset_item(
            dataset_name=dataset.name,
            input=case["input"],
            expected_output=case["expected_output"],
            metadata=case["metadata"]
        )
        print(f"  ➕ Пример добавлен: {item.id} | Вход: '{case['input'][:30]}...'")

    return dataset


if __name__ == "__main__":
    create_sentiment_dataset()