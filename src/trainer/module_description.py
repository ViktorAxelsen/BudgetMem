# description.py

Module1FilterDescription = "Module 1 (Filter): Selects the top-k memories that are most relevant to the query intent by scoring semantic relevance, reducing noise and limiting downstream computation to high-signal memories."

Module2EntityRelationDescription = "Module 2 (Entity Relation): Extracts query-relevant entities and their relationships from the filtered memories, focusing on factual, structured relations while removing redundant or weakly supported information."

Module3TemporalRelationDescription = "Module 3 (Temporal Relation): Identifies and organizes temporal information from the filtered memories, extracting time constraints, event orderings, and temporal dependencies that affect how entities and relations should be interpreted."

Module4SummaryDescription = "Module 4 (Summary): Synthesizes filtered memories together with extracted entity, temporal, and topic relations into a coherent knowledge summary, and provides explicit reasoning steps that a later language model can follow to answer the query, without directly answering the query itself."

Module5TopicRelationDescription = "Module 5 (Topic Relation): Analyzes thematic connections and topic transitions within the filtered memories, identifying main discussion topics, topic shifts, and thematic relationships that provide context for understanding the conversation flow and intent."