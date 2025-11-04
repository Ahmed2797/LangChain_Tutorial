# LangChain_Tutorial 
https://python.langchain.com/docs/integrations/components/


# Chat_model
https://python.langchain.com/docs/integrations/chat/

# Dccument_loader
https://python.langchain.com/docs/integrations/document_loaders/

# Retriver 
https://python.langchain.com/docs/integrations/retrievers/ 

# Lang_Graph 
https://docs.langchain.com/oss/python/langgraph/quickstart 


# graph-level conditional edge
It’s a conditional branch, but handled by LangGraph’s graph logic,
not by normal if ... else statements inside Python code.

        ┌─────────────┐
        │   START     │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  llm_tool   │   ← calls your LLM (model.invoke)
        └──────┬──────┘
               │
     condition_state ?
        ┌──────┴──────────┐
        │                 │
        ▼                 ▼
┌─────────────┐       ┌──────────┐
│  tool_node  │  or   │   END    │
└──────┬──────┘       └──────────┘
       │
       ▼
  ┌─────────────┐
  │  llm_tool   │  ← loops back to ask again if needed
  └─────────────┘

⚙️ Explanation

add_node('llm_tool', llm_tool) → defines your LLM node.

add_node('tool_node', tool_node) → defines your tool execution node.

add_edge(START, 'llm_tool') → starts from the LLM.

add_conditional_edges() → uses condition_state() to decide whether to:

    go to the tool_node, or

    end the process (END).

add_edge('tool_node', 'llm_tool') → loops back after the tool runs.