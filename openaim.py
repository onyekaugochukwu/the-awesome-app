import os
import logging
import base64
import tempfile
from typing import List, Tuple
import gradio as gr
from openai import OpenAI
from PIL import Image
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ''  # Replace with your actual API key

# Initialize OpenAI client
client = OpenAI()

# Global variables
processed_file_path = None
pdf_text = None

def process_file(file_path: str) -> str:
    global processed_file_path, pdf_text
    if file_path is None:
        logging.error("No file uploaded.")
        return "No file uploaded."
    try:
        logging.info(f"Processing file: {file_path}")
        processed_file_path = file_path
        
        if file_path.lower().endswith('.pdf'):
            pdf_text = extract_text_from_pdf(file_path)
            return f"PDF file '{os.path.basename(file_path)}' processed successfully."
        else:
            pdf_text = None
            return f"File '{os.path.basename(file_path)}' processed successfully."
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return f"Error processing file: {str(e)}"

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat_with_document(message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    global processed_file_path, pdf_text
    if processed_file_path is None:
        return history + [(message, "Please upload and process a file first.")], ""
    
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant that can understand both text and images."}]
        
        # Add conversation history
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        # Prepare the new message
        content = []
        
        # If it's a PDF, add the extracted text to the message
        if pdf_text:
            content.append({"type": "text", "text": f"Document content:\n{pdf_text}\n\nUser question: {message}"})
        else:
            content.append({"type": "text", "text": message})
        
        # If it's an image, add it to the message
        if processed_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            base64_image = encode_image(processed_file_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        messages.append({"role": "user", "content": content})
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300
        )
        
        assistant_response = response.choices[0].message.content
        history.append((message, assistant_response))
        return history, ""
    except Exception as e:
        logging.error(f"Error during query: {str(e)}")
        return history + [(message, f"Error during query: {str(e)}")], ""

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as app_ui:
    gr.Markdown("# Multimodal Chat with Your Documents")
    
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload File (PDF or Image)", type="filepath")
        with gr.Column(scale=1):
            file_process_button = gr.Button("Process File", variant="primary")
    
    file_output = gr.Markdown()
    
    chatbot = gr.Chatbot(height=400)
    message_input = gr.Textbox(label="Enter your message", placeholder="Type your question here...")
    submit_button = gr.Button("Submit", variant="primary")

    file_process_button.click(
        fn=process_file,
        inputs=file_input,
        outputs=file_output,
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

if __name__ == "__main__":
    app_ui.launch(debug=True)