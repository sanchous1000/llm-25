---
source: "..\data\raw\arxiv\2510.17720v1.pdf"
arxiv_id: "2510.17720v1"
page: 3
total_pages: 8
date_converted: "2025-11-05"
---

## Paraphrasing Prompt
Task Description:
You are a helpful assistant. I have a sentence with
certain entities that I want to preserve in spirit, but
you may modify the sentence slightly to add variety.
## Your task is:
1) Read the Original Sentence provided.
2) Create 2 new sentences (variants) that:
• DO NOT MODIFY any word enclosed in
<<>> tags or move them around (do not
introduce any new <<>> tags that weren’t
in the original).
• May adjust phrasing, structure, or add con-
textual details while maintaining logical co-
herence and meaning.
• Minor modifications are allowed, but retain
the core entity references and do not trans-
form them into something else.
3) Return the output in a valid JSON format with
the generated variants.
## Original Sentence: Input
Fig. 2. Prompt used for generating paraphrases.
A. Paraphrasing Framework
The core of our approach involves transforming input
sentences into masked templates where named entities are
replaced with semantic placeholders. For example, given the
input sentence “John visited the supermarket on Tuesday,”
the system generates a masked version: “<PER>visited the
<LOC>on Tuesday.” This masking preserves the essential
structure of the sentence while marking entity positions for
consistent paraphrasing. Using these masked templates, the
LLAMA 3.3-70B model generates variations that maintain the
original entity relationships while introducing some diversity.
## Our experiments produced paraphrases such as:
• <PER> stopped by <LOC> last Tuesday
• On Tuesday, <PER> went to <LOC>
• Tuesday saw <PER> traveling to <LOC>
Through experimental evaluation, we determined that gen-
erating two paraphrased versions per input sentence achieved
the optimal balance between diversity and quality. Attempts
to produce three or more variations frequently resulted in re-
dundancy or significant deviations from the original meaning.
## Furthermore, the additional paraphrases often failed to adhere
to the specified output formatting requirements (JSON format)
and occasionally produced sentences with an inconsistent
number of <ENT>tags compared to the input. Based on
these observations, we limited the generation to two variants.
## However, the system remains configurable, allowing users to
generate additional variations by adjusting the temperature
parameter during generation.
B. Implementation and Optimization
We have iterated over the paraphrasing prompt multiple
times and included optimizations to enhance the reliability of
the proposed paraphrasing system. One key improvement in-
volved the handling of consecutive entity tags. When multiple
words belonging to the same entity type appear in sequence
(for example, a four-word organization name), we consolidate
them into a single entity tag rather than using multiple
consecutive tags. This simplification reduced the complexity
of the paraphrasing task and improved the model’s ability to
maintain entity consistency.
## Initially, we used generic <ENTITY>tags instead of spe-
cific tags like <PER>or <LOC>. However, this approach
proved less effective as the model lacked sufficient context
about the type of entity being masked. Specific entity tags
provided better guidance for the paraphrasing process, partic-
ularly in domain-specific contexts where entity relationships
are more nuanced.
C. Quality Control and Validation
To ensure the quality of generated paraphrases, a structured
validation pipeline was developed using the instructor package
and a locally hosted version of LLAMA. Our system processes
the model’s output in JSON format, allowing efficient parsing
and validation of the generated paraphrases. For each para-
phrase, we verify that:
1) The number of entity tags matches the input sentences.
2) The semantic relationships between entities are pre-
served (cosine similarity).
## When a generated paraphrase fails these validation checks,
the system either triggers a regeneration with adjusted param-
eters or attempts to map the entities correctly based on their
position and context in the original sentence. This validation
process helps maintain the integrity of the augmented dataset
while allowing for natural variations in sentence structure and
word choice. The complete paraphrasing prompt is shown in
Fig. 2, where we instruct the model to maintain entity refer-
ences while allowing for structural variations and additional
context.
IV. INSTRUCTION TUNING AND ADAPTATION OF PROMPT
DESIGN
In this work, we revisit and refine the instruction-tuning
methodologies outlined in GNER [5], diverging from the
traditional BIO tagging schema in favour of a word/tag repre-
sentation format. The proposed method annotates each word
with its corresponding entity tag using a forward slash (/)
separator, simplifying the tagging process by removing the
complexity of distinguishing between “B-” and “I-” labels.
## Notably, all words within multi-word entities are assigned the
same tag. Furthermore, we provide detailed definitions and
guidelines, the same way as SLIMER [4], for each entity type
to enhance the extraction.
## We continue to leverage the effective strategy of incor-
porating negative instances, as outlined by GNER [5]. This
technique helps the model differentiate between entity and
