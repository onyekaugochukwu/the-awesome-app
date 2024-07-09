import os
import logging
import shutil
import gradio as gr
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer

# Set up logging
logging.basicConfig(level=logging.ERROR)

os.environ['OPENAI_API_KEY'] = ""
query_engine = None
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

def process_pdf(pdf_file):
    if pdf_file is None:
        return "No PDF file uploaded."
    try:
        file_name = os.path.basename(pdf_file.name)
        shutil.copy(pdf_file.name, file_name)
        result = activate_engine(file_name)
        return result
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return f"Error processing PDF: {str(e)}"

def activate_engine(file_name):
    global query_engine
    try:
        documents = SimpleDirectoryReader(input_files=[file_name]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(memory=chat_memory)
        return f"PDF file '{file_name}' processed successfully."
    except Exception as e:
        logging.error(f"Error loading document: {str(e)}")
        return f"Error loading document: {str(e)}"

def chat_with_document(message, history):
    global query_engine
    if query_engine is None:
        return history + [(message, "Please upload and process a PDF file first.")], ""
    try:
        response = query_engine.query(message)
        response_str = str(response).strip()
        history.append((message, response_str))
        return history, ""
    except Exception as e:
        logging.error(f"Error during query: {str(e)}")
        return history + [(message, f"Error during query: {str(e)}")], ""

with gr.Blocks(theme=gr.themes.Soft()) as app_ui:
    gr.Markdown("# Chat with Your PDF")
    
    with gr.Row():
        with gr.Column(scale=2):
            pdf_input = gr.File(label="Upload PDF File")
        with gr.Column(scale=1):
            pdf_process_button = gr.Button("Process PDF", variant="primary")
    
    pdf_output = gr.Markdown()
    
    chatbot = gr.Chatbot(height=400)
    message_input = gr.Textbox(label="Enter your message", placeholder="Type your question here...")
    submit_button = gr.Button("Submit", variant="primary")

    pdf_process_button.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=pdf_output,
        show_progress=True
    )

    submit_button.click(
        fn=chat_with_document,
        inputs=[message_input, chatbot],
        outputs=[chatbot, message_input]
    ).then(
        lambda: gr.update(value=""), outputs=[message_input]
    )

    message_input.submit(
        fn=chat_with_document,
        inputs=[message_input, chatbot],
        outputs=[chatbot, message_input]
    ).then(
        lambda: gr.update(value=""), outputs=[message_input]
    )

app_ui.launch(debug=True)