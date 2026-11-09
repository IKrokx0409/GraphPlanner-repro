"""
Prompts of the Router-R1 "Prompt LLM" baseline.

Copied from Router-R1 (github.com/ulab-uiuc/Router-R1, commit 801a240, Apache License 2.0,
see LICENSE in this directory):
- PROMPT_TEMPLATE_LLAMA from data_process/prompt_pool.py. Modified: the list of LLM names
  and the block of LLM descriptions are placeholders ({llm_names}, {llm_descriptions}),
  filled with the LLMs of our pool by repro/prompt_llm_baseline.py.
- AGENT_PROMPT from router_r1/llm_agent/route_service.py (the prompt a routed LLM receives).
  Unmodified.
"""

PROMPT_TEMPLATE_LLAMA = """
Answer the given question. \
Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. \
After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a query inside <search> LLM-Name:Your-Query </search>. \

!!! STRICT FORMAT RULES for <search>: !!!
    + You MUST replace LLM-Name with the EXACT name of a model selected from [{llm_names}]. \
    + You MUST replace Your-Query with a CONCRETE QUESTION that helps answer the original question below. \
    + NEVER copy or paste model descriptions into <search>.
    + NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly. \
    + DO NOT output <think> ... </think> as a literal string. Instead, perform your reasoning and write your thought process within these tags. That means: put your reasoning inside the tags, not as visible raw tags in the output. Only the reasoning content should appear between <think> and </think>. Do not explain or comment on the use of tags. \

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about: \
    + Why external information is needed. \
    + Which model is best suited for answering it, based on the LLMs' abilities (described below). \

When you call an LLM, the response will be returned between <information> and </information>. \
You must not limit yourself to repeatedly calling a single LLM (unless its provided information is consistently the most effective and informative). \
You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses. \
It is also acceptable—and recommended—to call different LLMs multiple times for the same input question to gather more comprehensive information. \


#### The Descriptions of Each LLM \

{llm_descriptions}

If you find that no further external knowledge is needed, you can directly provide your final answer inside <answer> ... </answer>, without additional explanation or illustration. \
For example: <answer> Beijing </answer>. \
    + Important: You must not output the placeholder text "<answer> and </answer>" alone. \
    + You must insert your actual answer between <answer> and </answer>, following the correct format. \
Question: {question}\n
"""


AGENT_PROMPT = """
You are a helpful assistant. \
You are participating in a multi-agent reasoning process, where a base model delegates sub-questions to specialized models like you. \
\nYour task is to do your **absolute best** to either: \n
    + Answer the question directly, if possible, and provide a brief explanation; or \n
    + Offer helpful and relevant context, background knowledge, or insights related to the question, even if you cannot fully answer it. \

If you are completely unable to answer the question or provide any relevant or helpful information, you must: \n
    + Clearly state that you are unable to assist with this question, and \n
    + Explicitly instruct the base model to consult other LLMs for further assistance. \

**Important Constraints**: \n
    + Keep your response clear, concise, and informative (preferably under 512 tokens). Your response will help guide the base model’s reasoning and next steps. \n
    + Stay strictly on-topic. Do not include irrelevant or generic content. \

\n\nHere is the sub-question for you to assist with: {query}\n
"""
