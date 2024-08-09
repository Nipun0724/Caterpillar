from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import torch
from transformers import BitsAndBytesConfig, pipeline
import whisper
from PIL import Image
from gtts import gTTS
import re
import datetime
import base64
CORS(app)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize the LLM pipeline
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline(
    "image-to-text",
    model=model_id,
    model_kwargs={"quantization_config": quant_config}
)

# Initialize the Whisper model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=DEVICE)

# Logger function
def writehistory(text):
    tstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"log_{tstamp}.txt"
    with open(os.path.join(app.config['PROCESSED_FOLDER'], logfile), "a", encoding='utf-8') as f:
        f.write(text + "\n")

# Function to encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Image-to-text function
def img2txt(input_text, input_image):
    image = Image.open(input_image)
    writehistory(f"Input text: {input_text}")

    if isinstance(input_text, tuple):
        prompt_instructions = """
        Describe the image using as much detail as possible.
        You are a helpful AI assistant who is able to answer questions about the image.
        What is the image all about?
        Now generate the helpful answer.
        """
    else:
        prompt_instructions = f"""
        Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt: {input_text}
        """

    prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    # Extract response text
    if outputs and len(outputs[0]["generated_text"]) > 0:
        match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
        reply = match.group(1) if match else "No response found."
    else:
        reply = "No response generated."

    return reply

# Audio transcription function
def transcribe(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    result = whisper.decode(whisper_model, mel)
    return result.text

# Text-to-speech function
def text_to_speech(text, file_path):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(file_path)
    return file_path

# Flask route for the main page
@app.route("/api/process", methods=["POST"])
def process_data():
    # Handle the POST request and return JSON
    data = request.json
    # Process the data...
    return jsonify({"result": "processed data"})


# Flask route to process uploaded files
@app.route('/process', methods=['POST'])
def process_files():
    if 'audio' not in request.files or 'image' not in request.files:
        return jsonify({"error": "Audio or image file missing"}), 400

    audio_file = request.files['audio']
    image_file = request.files['image']

    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)

    audio_file.save(audio_path)
    image_file.save(image_path)

    # Process audio and image
    speech_to_text_output = transcribe(audio_path)
    ai_output = img2txt(speech_to_text_output, image_path)
    processed_audio_path = text_to_speech(ai_output, os.path.join(app.config['PROCESSED_FOLDER'], "response.mp3"))

    return jsonify({
        "speech_to_text": speech_to_text_output,
        "ai_output": ai_output,
        "audio_url": processed_audio_path
    })

# Flask route to serve the processed audio file
@app.route('/processed/<filename>')
def serve_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
