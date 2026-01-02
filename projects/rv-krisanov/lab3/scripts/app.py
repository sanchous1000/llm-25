import gradio as gr
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
import sys
sys.path.append('scripts')

from agentic import agent_executor, langfuse_handler, system_prompt


checkpointer = InMemorySaver()
agent = agent_executor.with_config({"callbacks": [langfuse_handler]})

type History = list[list[str | None]]


def gradio_to_langchain_messages(history: History) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append(HumanMessage(content=user_msg))
        if bot_msg:
            messages.append(AIMessage(content=bot_msg))
    return messages


def add_user_message(user_message: str, history: History) -> tuple[str, History]:
    return "", history + [[user_message, None]]


def get_agent_response(history: History) -> History:
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(gradio_to_langchain_messages(history))
    
    result = agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": "default"}}
    )
    
    history[-1][1] = result["messages"][-1].content
    return history


def create_chat_interface():
    with gr.Blocks(title="D&D 5e RAG Agent") as demo:
        gr.Markdown("# D&D 5e Rules Expert")
        
        chatbot = gr.Chatbot(height=400)
        text_input = gr.Textbox(
            placeholder="Спроси о правилах D&D 5e...", 
            lines=2, 
            max_lines=2
        )
        send_button = gr.Button("Отправить", variant="primary")
        clear_button = gr.Button("Очистить")
        
        text_input.submit(
            add_user_message, 
            [text_input, chatbot], 
            [text_input, chatbot], 
            queue=False
        ).then(
            get_agent_response, 
            chatbot, 
            chatbot
        )
        
        send_button.click(
            add_user_message,
            [text_input, chatbot],
            [text_input, chatbot],
            queue=False
        ).then(
            get_agent_response,
            chatbot,
            chatbot
        )
        
        clear_button.click(lambda: None, None, chatbot, queue=False)
    
    return demo


if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
