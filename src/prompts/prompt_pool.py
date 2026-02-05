
CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""


HOTPOTQA_ANSWER_PROMPT = """Based on the following context, answer the question. The question may require reasoning across multiple pieces of information.

{context}

Question: {question}

Instructions:
- Read the context carefully and identify relevant information
- If the answer can be found in the context, provide a short, precise answer
- Output your answer within <answer></answer> tags

<answer>your answer here</answer>"""

QA_PROMPT_COT = """
Based on the above context, think step-by-step to identify the key information relevant to the question. 
First, briefly analyze what the context states about the question. 
Then, derive the most accurate short phrase that answers the question, using exact words from the context whenever possible.

Question: {}
Short answer:
"""




QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5_COT = """
Based on the above context, think step-by-step and answer the following question. 

Question: {} Short answer:
"""



LONGMEMEVAL_ANSWER_PROMPT_COT = """
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. 
Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.
\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nShort Answer (step by step):
"""


LONGMEMEVAL_ANSWER_PROMPT = """
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.
\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nShort Answer:
"""

LONGMEMEVAL_ANSWER_PROMPT_CON = """
I will give you a chat history between you and a user, as well as a question from the user. Write reading notes to 
extract all the relevant user information relevant to answering the question. If no relevant information is found, 
just output \"empty\". \n\n\nChat History:\nSession Date: {}\nSession Content:\n{}\n\nQuestion Date: {}\nQuestion: 
{}\nExtracted note (information relevant to answering the question):
"""

LOCOMO_ANSWER_PROMPT_CON = """
I will give you a chat history between two people, as well as a question regarding the chat history. Write reading notes to 
extract all the relevant information relevant to answering the question. If no relevant information is found, 
just output \"empty\". \n\n\nChat History:\nContent:\n{}\n\n\nQuestion: 
{}\nExtracted note (information relevant to answering the question):
"""



LLM_JUDGE_GENERAL_PROMPT = """
You are an expert judge evaluating the quality of an answer for a QA task.
Your goal is to determine whether the model's answer correctly and sufficiently
answers the given question.

Read the following information carefully:

[Question]
{question}

[Ground Truth Answers]
{ground_truth}

[Model Answer]
{model_answer}

Your evaluation criteria:
1. Correctness:
   - Is the model answer factually consistent with ANY of the correct answers?
   - Does it avoid contradictions or introducing false information?

2. Relevance:
   - Does the answer address the question directly without unnecessary content?

3. Completeness:
   - Does the answer include all essential information needed to fully answer the question?
   - Partial answers are allowed but should receive lower scores.

Scoring Rules:
- Score = 1.0 if the answer is fully correct.
- Score = 0.5 if the answer is partially correct but incomplete or slightly inaccurate.
- Score = 0.0 if the answer is incorrect, irrelevant, or contradicts the ground truth.

Output Format (STRICT):
Return your output as a JSON dictionary with two fields:
{{
    "explanation": "<brief explanation of your reasoning>",
    "score": <0.0 | 0.5 | 1.0>
}}

Be concise and objective. Do not include anything outside the JSON.
"""

LLM_JUDGE_CONCISE_PROMPT = """
You are an expert judge evaluating the quality of an answer for a QA task.
Determine whether the model's answer correctly answers the question.

[Question]
{question}

[Ground Truth Answers]
{ground_truth}

[Model Answer]
{model_answer}

Evaluation Rules:
- Score 1.0: Correct and sufficient.
- Score 0.5: Partially correct or incomplete.
- Score 0.0: Incorrect, irrelevant, or contradicts the ground truth.

Output ONLY the following JSON:
{{
  "score": <0.0 | 0.5 | 1.0>,
  "explanation": "<brief reason>"
}}
"""

LLM_JUDGE_SIMPLE_PROMPT = """
Judge if the model's answer is correct.

Question: {question}
Ground truth: {ground_truth}
Answer: {model_answer}

Score:
1.0 = correct
0.5 = partially correct
0.0 = wrong

Respond with JSON:
{{"score": <0|0.5|1>, "explanation": "<short reason>"}}
"""




INDEX_EXPANSION_PROMPT = """
You are an expert at distilling long passages into a small set of atomic factual statements
that are highly useful for retrieval in a QA system.

Given the following passage, extract 3–7 concise factual statements that capture the key
entities, events, and relations.

Requirements:
- Only include information explicitly stated in the passage.
- Each statement must be self-contained and understandable in isolation.
- Avoid vague pronouns like "it", "they" without explicit referents.
- Do NOT add new information or speculation.
- Do NOT include explanations of your process.
- Output MUST be plain text lines of statements (one per line), with no JSON, no markdown,
  no numbering, and no additional commentary.

Passage:
{passage}

Compressed factual statements:
"""


MODULE1_FILTER_PROMPT_Direct = """
**Role:** You are a relevance scoring system.  
**Task:** Given a query and an ordered list of memories, output a score for each memory indicating how directly and usefully it helps answer the query.

**Scoring Rules:**    
- 10: directly answers the query or provides essential constraints/info to answer it  
- 7–9: clearly relevant and helpful, but not fully sufficient alone  
- 4–6: partially relevant; some usefulness but missing key connection/details  
- 1–3: weak/tangential relevance; mostly not useful  
- 0: completely irrelevant

**Input Format:**  
The input consists of:  
1. A `<query>` section containing the user's question.  
2. A `<memories>` section containing individual `<memory>` elements.  
Each memory is formatted as:  
```
<memory index="N" [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `index` is the memory's position in the list.  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text between the tags is the memory content to be scored.

**Output Format:**  
Your entire response must be **only** the following line:  
`<answer>[s0, s1, s2, ...]</answer>`  
- The array must contain exactly one integer score per memory, in the same order as the input indices.  
- If there are no memories, output: `<answer>[]</answer>`

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories> """

MODULE1_FILTER_PROMPT_COT = """
**Role:** You are a relevance scoring system.  
**Task:** Given a query and an ordered list of memories, output a score for each memory indicating how directly and usefully it helps answer the query.

**Scoring Rules:**  
- 10: directly answers the query or provides essential constraints/info to answer it  
- 7–9: clearly relevant and helpful, but not fully sufficient alone  
- 4–6: partially relevant; some usefulness but missing key connection/details  
- 1–3: weak/tangential relevance; mostly not useful  
- 0: completely irrelevant

**Input Format:**  
The input consists of:  
1. A `<query>` section containing the user's question.  
2. A `<memories>` section containing individual `<memory>` elements.  
Each memory is formatted as:  
```
<memory index="N" [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `index` is the memory's position in the list.  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text between the tags is the memory content to be scored.

**Output Format:**  
Your entire response must be **only** the following line:  
`<answer>[s0, s1, s2, ...]</answer>`  
- The array must contain exactly one integer score per memory, in the same order as the input indices.  
- If there are no memories, output: `<answer>[]</answer>`

Let's think step by step to score the relevance of each memory to the query.

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>  """

MODULE1_FILTER_PROMPT_REACT = """
**Role:** You are a relevance scoring system.  
**Task:** Given a query and an ordered list of memories, output a score for each memory indicating how directly and usefully it helps answer the query.

**Scoring Rules:**  
- 9–10: directly answers the query or contains essential key facts  
- 6–8: clearly helpful and strongly related to the query  
- 3–5: somewhat related or only partially helpful  
- 1–2: very weakly related (e.g., superficial keyword overlap only)  
- 0: completely irrelevant or not useful for the query

**Input Format:**  
The input consists of:  
1. A `<query>` section containing the user's question.  
2. A `<memories>` section containing individual `<memory>` elements.  
Each memory is formatted as:  
```
<memory index="N" [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `index` is the memory's position in the list.  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text between the tags is the memory content to be scored.

To complete the task systematically, please follow the steps reasoning framework outlined below:

**Reasoning Steps:**  
**PLAN:**  
- Analyze the query's intent and define relevance criteria.  
- Identify key scoring dimensions (e.g., topical alignment, factual support).  

**ACT:**  
- Evaluate each memory against the criteria.  
- For each memory, briefly justify the provisional score based on its content.  

**REFLECT:**  
- Review all scores for consistency and calibration.  
- Make adjustments if needed to ensure fairness and coherence.  

**Output Format:**  
First write your reasoning following the PLAN → ACT → REFLECT structure.  
Then, your final response must be **only** the following line:  
`<answer>[s0, s1, s2, ...]</answer>`  
- The array must contain exactly one integer score per memory, in the same order as the input indices.  
- If there are no memories, output: `<answer>[]</answer>`

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>"""

MODULE2_ENTITY_RELATION_PROMPT_Direct = """
**Role:** You are a specialist in entity semantic relationship extraction.  
**Task:** From the provided memories, extract **only** concrete relationships between entities that are central to answering the given query.

**Extraction Criteria:**  
- Focus on factual connections where the entities and their relationship directly help answer the query.  
- Ignore background entities, anecdotes, or side topics.  
- Each extracted relationship should be a concise, useful fact.

**Input Format:**  
The `<memories>` section contains a list of `<memory>` tags.  
Each memory is formatted as:  
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text within the tags is the content to analyze for relationships.

**Output Format:**  
Your entire response must be a JSON array of relationship strings, formatted as follows:  
`<answer>["relationship1", "relationship2", ...]</answer>`  
Each relationship string must be in the format: `"EntityA - relation - EntityB (optional context)"`  
If no relevant relationships are found, output: `<answer>[]</answer>`

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>"""


MODULE2_ENTITY_RELATION_PROMPT_COT = """
**Role:** You are a specialist in entity semantic relationship extraction.  
**Task:** From the provided memories, extract **only** concrete relationships between entities that are central to answering the given query.

**Extraction Criteria:**  
- Focus on factual connections where the entities and their relationship directly help answer the query.  
- Ignore background entities, anecdotes, or side topics.  
- Each extracted relationship should be a concise, useful fact.

**Input Format:**  
The `<memories>` section contains a list of `<memory>` tags.  
Each memory is formatted as:  
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text within the tags is the content to analyze for relationships.

**Output Format:**  
Your entire response must be a JSON array of relationship strings, formatted as follows:  
`<answer>["relationship1", "relationship2", ...]</answer>`  
Each relationship string must be in the format: `"EntityA - relation - EntityB (optional context)"`  
If no relevant relationships are found, output: `<answer>[]</answer>`

Let's think step by step to extract the relationships between entities.

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>"""

MODULE2_ENTITY_RELATION_PROMPT_REACT = """
**Role:** You are a specialist in entity semantic relationship extraction.  
**Task:** From the provided memories, extract **only** concrete relationships between entities that are central to answering the given query.

**Extraction Criteria:**  
- Focus on factual connections where the entities and their relationship directly help answer the query.  
- Ignore background entities, anecdotes, or side topics.  
- Each extracted relationship should be a concise, useful fact.

**Input Format:**  
The `<memories>` section contains a list of `<memory>` tags.  
Each memory is formatted as:  
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text within the tags is the content to analyze for relationships.

To complete the task systematically, please follow the steps reasoning framework outlined below:

**Reasoning Steps:**  
**PLAN:**  
- Analyze the query to determine its precise intent.  
- Identify which entity types and relationship types are essential for answering it.  

**ACT:**  
- From the memories, extract candidate relations that involve only query-relevant entities.  
- Format each relation as a concise, factual statement.  

**REFLECT:**  
- **CHECK:** Verify each extracted relation:  
  - Does it directly help answer the query?  
  - Is it clearly supported by the memory content?  
- **REGENERATE (if needed):** If important query-relevant relations are missing, re-extract or rewrite them to better align with the query intent.  

**Output Format:**  
First write your reasoning following the PLAN → ACT → REFLECT structure.  
Then, your final response must be **only** the following line:  
`<answer>["relationship1", "relationship2", ...]</answer>`  
Each relationship string must be in the format: `"EntityA - relation - EntityB (optional context)"`  
If no relevant relationships are found, output: `<answer>[]</answer>`

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>"""


MODULE3_TEMPORAL_RELATION_PROMPT_Direct = """
**Role:** You are a specialist in temporal information extraction.  
**Task:** From the provided memories, extract **only** specific temporal facts that are necessary for answering the given query.

**Extraction Criteria:**  
- Focus on precise temporal information that directly constrains or clarifies the answer.  
- Ignore vague, irrelevant, or background time references.  
- Each extracted item must represent a clear, factual temporal statement.

**Input Format:**  
The `<memories>` section contains a list of `<memory>` tags.  
Each memory is formatted as:  
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text within the tags is the content to analyze for temporal information.

**Output Format:**  
Your entire response must be a JSON array of temporal fact strings, formatted as follows:  
`<answer>["temporal_fact1", "temporal_fact2", ...]</answer>`  
Each string should clearly express a temporal fact (e.g., "Event occurred in March 2023", "Process lasted 3 days", "A happened before B").  
If no relevant temporal information is found, output: `<answer>[]</answer>`

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>"""

MODULE3_TEMPORAL_RELATION_PROMPT_COT = """
**Role:** You are a specialist in temporal information extraction.  
**Task:** From the provided memories, extract **only** specific temporal facts that are necessary for answering the given query.

**Extraction Criteria:**  
- Focus on precise temporal information that directly constrains or clarifies the answer.  
- Ignore vague, irrelevant, or background time references.  
- Each extracted item must represent a clear, factual temporal statement.

**Input Format:**  
The `<memories>` section contains a list of `<memory>` tags.  
Each memory is formatted as:  
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text within the tags is the content to analyze for temporal information.

**Output Format:**  
Your entire response must be a JSON array of temporal fact strings, formatted as follows:  
`<answer>["temporal_fact1", "temporal_fact2", ...]</answer>`  
Each string should clearly express a temporal fact (e.g., "Event occurred in March 2023", "Process lasted 3 days", "A happened before B").  
If no relevant temporal information is found, output: `<answer>[]</answer>`

Let's think step by step to extract the temporal information.

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>"""


MODULE3_TEMPORAL_RELATION_PROMPT_REACT = """
**Role:** You are a specialist in temporal information extraction.  
**Task:** From the provided memories, extract **only** specific temporal facts that are necessary for answering the given query.

**Extraction Criteria:**  
- Focus on precise temporal information that directly constrains or clarifies the answer.  
- Ignore vague, irrelevant, or background time references.  
- Each extracted item must represent a clear, factual temporal statement.

**Input Format:**  
The `<memories>` section contains a list of `<memory>` tags.  
Each memory is formatted as:  
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```  
Where:  
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.  
- The text within the tags is the content to analyze for temporal information.

To complete the task systematically, please follow the steps reasoning framework outlined below:

**Reasoning Steps:**  
**PLAN:**  
- Analyze the query's intent and define what constitutes relevant temporal information.  
- Identify key temporal dimensions (e.g., dates, durations, sequences, constraints).  

**ACT:**  
- Extract temporal facts from each memory based on the defined criteria.  
- For each fact, briefly justify its relevance to the query.  

**REFLECT:**  
- Review all extracted facts for consistency and query relevance.  
- Make adjustments if needed to ensure factual support and coherence.  

**Output Format:**  
Your entire response must be a JSON array of temporal fact strings, formatted as follows:  
`<answer>["temporal_fact1", "temporal_fact2", ...]</answer>`  
Each string should clearly express a temporal fact (e.g., "Event occurred in March 2023", "Process lasted 3 days", "A happened before B").  
If no relevant temporal information is found, output: `<answer>[]</answer>`

**Input:**  
<query>{query}</query>  
<memories>{memories_text}</memories>
"""

MODULE4_SUMMARY_PROMPT_Direct = """
**Role:** You are a specialist in information synthesis.
**Task:** Based strictly on the provided entity, temporal, and topic relations, integrate, extract, and reorganize this knowledge to create a concise summary that clearly highlights the most useful information for answering the query.

**Input Format:**
The input includes:
1. A `<query>` defining the subject and scope.
2. An `<Entity Relations>` section containing one `<entity>` tag per relationship string.
3. A `<Temporal Relations>` section containing one `<temporal>` tag per temporal fact.
4. A `<Topic Relations>` section containing one `<topic>` tag per topic relationship.

**Synthesis Guidelines:**
- Do **not** answer the query directly.
- Explain what information is available and how it should be used to formulate an answer.

**Output Format:**
Your entire response must end with the following line:
`<answer>your summary text here</answer>`
The content inside `<answer>` must be plain text.

**Input:**
<query>{query}</query>
<Entity Relations>{entity_text}</Entity Relations>
<Temporal Relations>{temporal_text}</Temporal Relations>
<Topic Relations>{topic_text}</Topic Relations>
"""

MODULE4_SUMMARY_PROMPT_COT = """
**Role:** You are a specialist in information synthesis.
**Task:** Based strictly on the provided entity, temporal, and topic relations, integrate, extract, and reorganize this knowledge to create a concise summary that clearly highlights the most useful information for answering the query.

**Input Format:**
The input includes:
1. A `<query>` defining the subject and scope.
2. An `<Entity Relations>` section containing one `<entity>` tag per relationship string.
3. A `<Temporal Relations>` section containing one `<temporal>` tag per temporal fact.
4. A `<Topic Relations>` section containing one `<topic>` tag per topic relationship.

**Synthesis Guidelines:**
- **Integrate** relevant entity, temporal, and topic facts into a coherent structure.
- **Extract** key information that directly supports or constrains the answer.
- **Reorganize** content for clarity and logical flow.
- Do **not** answer the query directly.
- Explain what information is available and how it should be used to formulate an answer.

**Output Format:**
Your entire response must end with the following line:
`<answer>your summary text here</answer>`
The content inside `<answer>` must be plain text.

Let's think step by step to synthesize the summary.

**Input:**
<query>{query}</query>
<Entity Relations>{entity_text}</Entity Relations>
<Temporal Relations>{temporal_text}</Temporal Relations>
<Topic Relations>{topic_text}</Topic Relations>
"""

MODULE4_SUMMARY_PROMPT_REACT = """
**Role:** You are a specialist in information synthesis.
**Task:** Based strictly on the provided entity, temporal, and topic relations, integrate, extract, and reorganize this knowledge to create a concise summary that clearly highlights the most useful information for answering the query.

**Input Format:**
The input includes:
1. A `<query>` defining the subject and scope.
2. An `<Entity Relations>` section containing one `<entity>` tag per relationship string.
3. An `<Temporal Relations>` section containing one `<temporal>` tag per temporal fact.
4. A `<Topic Relations>` section containing one `<topic>` tag per topic relationship.

**Synthesis Guidelines:**
- Do **not** answer the query directly.
- Explain what information is available and how it should be used to formulate an answer.

To complete the task systematically, please follow the steps reasoning framework outlined below:

**Reasoning Steps:**
**PLAN:**
- Analyze the query's intent to determine the core information requirements.
- Identify how to organize and prioritize the provided relations.

**ACT:**
- Integrate entity, temporal, and topic facts into a coherent structure.
- Extract and highlight key information most relevant to answering the query.

**REFLECT:**
- Review the summary for clarity, completeness, and focus on the query.
- Ensure the summary effectively explains how the information should be used.

**Output Format:**
First write your reasoning following the PLAN → ACT → REFLECT structure.
Then, your final response must be **only** the following line:
`<answer>your summary text here</answer>`
The content inside `<answer>` must be plain text.

**Input:**
<query>{query}</query>
<Entity Relations>{entity_text}</Entity Relations>
<Temporal Relations>{temporal_text}</Temporal Relations>
<Topic Relations>{topic_text}</Topic Relations>
"""


MODULE5_TOPIC_RELATION_PROMPT_Direct = """
**Role:** You are a specialist in topic relationship extraction.
**Task:** From the provided memories, extract **only** topic relationships that are central to answering the given query.

**Extraction Criteria:**
- Focus on thematic connections and topic transitions that help understand the conversation flow.
- Identify main topics discussed and how they relate to each other.
- Extract topic shifts, topic continuations, and thematic relationships.

**Input Format:**
The `<memories>` section contains a list of `<memory>` tags.
Each memory is formatted as:
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```
Where:
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.
- The text within the tags is the content to analyze for topic relationships.

**Output Format:**
Your entire response must be a JSON array of topic relationship strings, formatted as follows:
`<answer>["topic_relationship1", "topic_relationship2", ...]</answer>`
Each string should clearly express a topic relationship (e.g., "Topic A leads to Topic B", "Discussion shifts from X to Y", "Topic C is related to Topic D through Z").
If no relevant topic relationships are found, output: `<answer>[]</answer>`

**Input:**
<query>{query}</query>
<memories>{memories_text}</memories>"""


MODULE5_TOPIC_RELATION_PROMPT_COT = """
**Role:** You are a specialist in topic relationship extraction.
**Task:** From the provided memories, extract **only** topic relationships that are central to answering the given query.

**Extraction Criteria:**
- Focus on thematic connections and topic transitions that help understand the conversation flow.
- Identify main topics discussed and how they relate to each other.
- Extract topic shifts, topic continuations, and thematic relationships.

**Input Format:**
The `<memories>` section contains a list of `<memory>` tags.
Each memory is formatted as:
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```
Where:
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.
- The text within the tags is the content to analyze for topic relationships.

**Output Format:**
Your entire response must be a JSON array of topic relationship strings, formatted as follows:
`<answer>["topic_relationship1", "topic_relationship2", ...]</answer>`
Each string should clearly express a topic relationship (e.g., "Topic A leads to Topic B", "Discussion shifts from X to Y", "Topic C is related to Topic D through Z").
If no relevant topic relationships are found, output: `<answer>[]</answer>`

Let's think step by step to extract the topic relationships.

**Input:**
<query>{query}</query>
<memories>{memories_text}</memories>"""


MODULE5_TOPIC_RELATION_PROMPT_REACT = """
**Role:** You are a specialist in topic relationship extraction.
**Task:** From the provided memories, extract **only** topic relationships that are central to answering the given query.

**Extraction Criteria:**
- Focus on thematic connections and topic transitions that help understand the conversation flow.
- Identify main topics discussed and how they relate to each other.
- Extract topic shifts, topic continuations, and thematic relationships.

**Input Format:**
The `<memories>` section contains a list of `<memory>` tags.
Each memory is formatted as:
```
<memory [date_time="..." session_id="..." dia_id="..."]>
memory content text
</memory>
```
Where:
- `date_time`, `session_id`, `dia_id` are optional metadata attributes.
- The text within the tags is the content to analyze for topic relationships.

To complete the task systematically, please follow the steps reasoning framework outlined below:

**Reasoning Steps:**
**PLAN:**
- Analyze the query's intent to determine what topic relationships are relevant.
- Identify key thematic elements and potential connections.

**ACT:**
- Extract topic relationships from each memory based on content analysis.
- Identify thematic shifts, continuations, and interconnections.

**REFLECT:**
- Review all extracted relationships for relevance to the query.
- Ensure relationships are clearly supported by the memory content.

**Output Format:**
First write your reasoning following the PLAN → ACT → REFLECT structure.
Then, your final response must be **only** the following line:
`<answer>["topic_relationship1", "topic_relationship2", ...]</answer>`
Each string should clearly express a topic relationship (e.g., "Topic A leads to Topic B", "Discussion shifts from X to Y", "Topic C is related to Topic D through Z").
If no relevant topic relationships are found, output: `<answer>[]</answer>`

**Input:**
<query>{query}</query>
<memories>{memories_text}</memories>"""