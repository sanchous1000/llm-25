---
source: "..\data\raw\arxiv\2510.17722v1.pdf"
arxiv_id: "2510.17722v1"
page: 4
total_pages: 9
date_converted: "2025-11-05"
---

## Selected
High similarity
to the left frame
Video Datasets
2. Extract Frames
3. Object Extraction
4. Object Memory Bank
5. Relevant Scene Merging
6. MTQA Generation
7. Human Verification
Filter
Criteria
•
Sharpness
•
Similariy
Object Detection & Caption
ID: ID 1
Information: ......
## Scene Settings
•
Single Scene
•
Cross Scene
...
...
...
...
1. Scene Splitting
Low Sharpness
Person Caption
Animal Caption
...
...
Time
ID: ID 2
Information: ......
......
•
MTQA Accuracy Checking
•
Task Quality Verification
•
Contextual Coherence
•
......
## Annotation Criteria
6 Core Abilities
•
Perceptivity
•
Interactivity
...
...
## Scene i
Scene j
Scene k
Object-Based Scene Merging
Figure 2: An overview of the semi-automatic data construction process of MT-Video-Bench.
includes:
• Object Reference (OR) evaluates the model’s ability to resolve references and pronouns in the user’s
input, ensuring that entities mentioned implicitly are correctly mapped to the appropriate objects,
characters, or concepts.
• Memory Recall (MR) measures the model’s capacity to retrieve, retain, and integrate relevant informa-
tion from prior conversational turns or long-term history, enabling coherent reasoning and continuity
across interactions.
• Content Summary (CS) assesses the model’s effectiveness in condensing conversational and video
content into succinct yet comprehensive summaries, while preserving essential details, coherent
structure, and semantic fidelity.
## Interactivity evaluates the model’s capacity to conduct coherent, adaptive, and user-aware dialogues
based on the video content. It focuses on appropriately refusing unanswerable questions, smoothly
adapting to topic changes, and proactively maintaining engagement. It includes:
• Answer Refusal (AR) tests the ability to recognize unanswerable queries based on available evidence
and explicitly decline or indicate insufficiency without hallucination.
• Topic Shifting (TS) evaluates how effectively the model can track and adapt to user-initiated changes in
conversational focus or subject matter, while maintaining coherence, fluency, and relevance throughout
the dialogue.
• Proactive Interaction (PI) probes the model’s capacity to sustain or restore engagement through
clarifications, elaborations, or novel insights when signs of disinterest or disengagement are detected,
thereby fostering renewed interest and continuation of the dialogue.
3.3
Data Collection
As shown in Figure 2, the data collection process for MT-Video-Bench involves both automated con-
struction and human verification. We first acquire videos from online platforms and split them into
single-scene segments. Next, we retrieve and merge relevant scenes by extracting frames, performing
object detection, and constructing an object memory bank. Multi-turn dialogues are then generated
automatically for diverse evaluation tasks. Finally, human annotators are involved to ensure the accuracy
and quality of the generated dialogues.
Video Collection and Single-Scene Splitting. The data collection process begins with the manual acqui-
sition of 135 videos from various online platforms, such as YouTube, within the past year. Subsequently,
we employ PySceneDetect1 to divide the videos into shorter clips. Recognizing that these clips are often
too brief to represent complete scenes, we then use the Gemini 2.5 Flash model (Team, 2025) to generate
descriptive captions for each clip. Finally, the caption-based clip merging method is iteratively applied
twice to combine related clips into a coherent, single-scene video, ensuring a seamless and contextually
accurate representation of the scene. These refined single-scene videos serve as the core visual content for
the subsequent task of generating single-scene, multi-turn dialogues.
1https://github.com/Breakthrough/PySceneDetect
4
