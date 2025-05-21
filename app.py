import os
import json
import whisper
import time
import uuid
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import re
import traceback
from dotenv import load_dotenv
import torch
import threading
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
HISTORY_FILE = "history.json"

# Ensure uploads folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global placeholders for lazy loading
model = None
hf_client = None

# Processing jobs dictionary
processing_jobs = {}

# Initialize Hugging Face API
huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
if huggingface_api_key:
    # Set up the InferenceClient
    hf_client = InferenceClient(token=huggingface_api_key)
    print("Hugging Face API key configured successfully")
else:
    print("WARNING: No Hugging Face API key found in environment variables")

# Load history
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"History load error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error loading history: {e}")
        return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as file:
            json.dump(history, file)
        return True
    except Exception as e:
        print(f"History save error: {e}")
        return False

history = load_history()

# Lazy loading for Whisper model
def get_whisper_model():
    global model
    if model is None:
        print("Loading Whisper model...")
        model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
    return model

# Convert non-wav files with optimized settings
def convert_audio(input_file):
    try:
        # Get file extension
        file_ext = os.path.splitext(input_file)[1].lower()
        
        # Skip conversion if already WAV
        if file_ext == '.wav':
            return input_file
            
        # Convert with optimal settings
        audio = AudioSegment.from_file(input_file)
        
        # Downsample to 16kHz for Whisper
        if audio.frame_rate > 16000:
            audio = audio.set_frame_rate(16000)
            
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        wav_file = os.path.splitext(input_file)[0] + ".wav"
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return input_file

# Optimized transcription
def transcribe_audio(audio_path):
    try:
        if os.path.getsize(audio_path) == 0:
            return "Audio file is empty."
        
        audio = AudioSegment.from_file(audio_path)
        if len(audio) < 1000:
            return "Audio too short to transcribe."
        
        # Get model only when needed
        model = get_whisper_model()
        
        # Enable GPU optimizations if available
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            result = model.transcribe(audio_path)
            
        return result["text"]
    except Exception as e:
        return f"Transcription failed: {str(e)}"
    
# Chunking for long texts
def chunk_text(text, max_chars=4000):
    if len(text) <= max_chars:
        return [text]
    sentences = text.split('. ')
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 2 <= max_chars:
            current += s + ". "
        else:
            chunks.append(current)
            current = s + ". "
    if current:
        chunks.append(current)
    return chunks

# Call Hugging Face API with retry - with improved error handling
def call_huggingface_with_retry(client, model_id, payload, max_attempts=3):
    last_error = None
    for attempt in range(max_attempts):
        try:
            print(f"API call attempt {attempt+1}/{max_attempts} for model {model_id}")
            
            # Add timeout to prevent hanging
            response = client.post(
                model=model_id, 
                json=payload,
                headers={"Authorization": f"Bearer {huggingface_api_key}"}
            )
            
            return response
        except Exception as e:
            last_error = e
            print(f"API call failed: {type(e).__name__}: {str(e)}")
            if attempt < max_attempts - 1:
                sleep_time = 1 * (attempt + 1)  # Linear backoff (shorter)
                print(f"Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

# Summarization using Hugging Face
def summarize_text(text):
    """Generate a summary using Hugging Face API."""
    if not text or len(text.strip()) < 50:
        return "Text too short to summarize."
    
    if not huggingface_api_key:
        return "Hugging Face API key not configured. Please set up the API key to enable summarization."
    
    return summarize_with_huggingface(text)

def summarize_with_huggingface(text):
    if not huggingface_api_key:
        print("No Hugging Face API key configured")
        return "Hugging Face API key not configured. Please add a valid API key to enable summarization."
    
    # Add explicit validation
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "Text too short or invalid format for summarization."
    
    # Define multiple models to try in case of failures
    summarization_models = [
        "google/pegasus-xsum",   # Fast and reliable
        "sshleifer/distilbart-cnn-12-6",  # Smaller version of BART
        "facebook/bart-large-cnn",  # Only try if others fail
        "t5-small"  # Very small, should work even with limitations
    ]
    
    consolidation_models = [
        "gpt2-medium",  # Generally reliable
        "google/flan-t5-base",  # Good for instructions
        "meta-llama/Llama-2-7b-chat-hf"  # Larger model, try last
    ]
    
    try:
        global hf_client
        if hf_client is None:
            hf_client = InferenceClient(token=huggingface_api_key)
        
        # Handle different text lengths appropriately
        if len(text) < 8000:  # Process smaller texts directly
            print(f"Processing single chunk of {len(text)} characters")
            
            # Try each model until one works
            summary = None
            last_error = None
            
            for model_id in summarization_models:
                try:
                    print(f"Attempting summarization with model: {model_id}")
                    # Prepare payload - adjust based on model
                    if model_id.startswith("t5"):
                        payload = {
                            "inputs": "summarize: " + text,
                            "parameters": {"max_length": 800}
                        }
                    else:
                        payload = {
                            "inputs": text,
                            "parameters": {"max_length": 800, "min_length": 40}
                        }
                    
                    response = call_huggingface_with_retry(hf_client, model_id, payload, max_attempts=2)
                    
                    # Handle different response formats
                    if isinstance(response, list) and len(response) > 0:
                        if isinstance(response[0], dict) and "summary_text" in response[0]:
                            summary = response[0]["summary_text"]
                        elif isinstance(response[0], dict) and "generated_text" in response[0]:
                            summary = response[0]["generated_text"]
                        else:
                            summary = str(response[0])
                    elif isinstance(response, dict):
                        if "summary_text" in response:
                            summary = response["summary_text"]
                        elif "generated_text" in response:
                            summary = response["generated_text"]
                        else:
                            summary = str(response)
                    elif isinstance(response, str):
                        summary = response
                    
                    if summary:
                        print(f"Successfully generated summary with {model_id}")
                        break
                        
                except Exception as e:
                    last_error = e
                    print(f"Error with model {model_id}: {str(e)}")
                    time.sleep(1)  # Small delay before trying next model
            
            if summary:
                return summary
            else:
                # Fallback to local summarization if all API calls fail
                print("All API models failed, using fallback local summarization")
                return fallback_local_summarization(text)
        else:
            # Handle longer transcripts with smaller chunks
            print(f"Processing text of {len(text)} characters in chunks")
            chunks = chunk_text(text, max_chars=4000)  # Smaller chunks for better reliability
            print(f"Split into {len(chunks)} chunks")
            
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} characters")
                
                # Try each model for this chunk
                chunk_summary = None
                for model_id in summarization_models:
                    try:
                        print(f"Attempting chunk {i+1} with model: {model_id}")
                        
                        # Prepare payload - adjust based on model
                        if model_id.startswith("t5"):
                            payload = {
                                "inputs": "summarize: " + chunk,
                                "parameters": {"max_length": 300}
                            }
                        else:  
                            payload = {
                                "inputs": chunk,
                                "parameters": {"max_length": 300, "min_length": 30}
                            }
                        
                        response = call_huggingface_with_retry(hf_client, model_id, payload, max_attempts=2)
                        
                        # Handle different response formats
                        if isinstance(response, list) and len(response) > 0:
                            if isinstance(response[0], dict) and "summary_text" in response[0]:
                                chunk_summary = response[0]["summary_text"]
                            elif isinstance(response[0], dict) and "generated_text" in response[0]:
                                chunk_summary = response[0]["generated_text"]
                            else:
                                chunk_summary = str(response[0])
                        elif isinstance(response, dict):
                            if "summary_text" in response:
                                chunk_summary = response["summary_text"]
                            elif "generated_text" in response:
                                chunk_summary = response["generated_text"]
                            else:
                                chunk_summary = str(response)
                        elif isinstance(response, str):
                            chunk_summary = response
                        
                        if chunk_summary:
                            print(f"Successfully generated summary for chunk {i+1} with {model_id}")
                            chunk_summaries.append(f"**Part {i+1} Summary:**\n{chunk_summary}")
                            break
                    except Exception as e:
                        print(f"Error with model {model_id} for chunk {i+1}: {str(e)}")
                        time.sleep(1)  # Small delay before trying next model
                
                # If all models failed for this chunk, use fallback
                if not chunk_summary:
                    print(f"All models failed for chunk {i+1}, using fallback")
                    local_summary = fallback_local_summarization(chunk)
                    chunk_summaries.append(f"**Part {i+1} Summary:**\n{local_summary}")
            
            # If we have multiple chunks, combine them
            combined_summary = "\n\n".join(chunk_summaries)
            
            # Check if we need to create a consolidated summary (if all chunks were successful)
            if len(chunks) > 1 and not any("Failed to summarize" in s for s in chunk_summaries):
                try:
                    # Try each consolidation model
                    consolidated = None
                    for model_id in consolidation_models:
                        try:
                            print(f"Attempting consolidation with model: {model_id}")
                            
                            # Prepare prompt for consolidation
                            consolidation_prompt = f"""Synthesize these meeting summary sections into one cohesive summary:

                            {combined_summary}

                            Create a final summary with:
                            1. Meeting overview
                            2. Key discussions
                            3. Decisions made
                            4. Action items
                            """
                            
                            # Prepare payload - adjust based on model
                            if model_id.startswith("google/flan"):
                                payload = {
                                    "inputs": "Summarize: " + consolidation_prompt,
                                    "parameters": {"max_length": 1000, "temperature": 0.4}
                                }
                            else:
                                payload = {
                                    "inputs": consolidation_prompt,
                                    "parameters": {"max_length": 1000, "temperature": 0.4}
                                }
                            
                            response = call_huggingface_with_retry(hf_client, model_id, payload, max_attempts=2)
                            
                            if isinstance(response, list) and len(response) > 0:
                                if isinstance(response[0], dict) and "generated_text" in response[0]:
                                    consolidated = response[0]["generated_text"]
                                else:
                                    consolidated = str(response[0])
                            elif isinstance(response, dict) and "generated_text" in response:
                                consolidated = response["generated_text"]
                            elif isinstance(response, str):
                                consolidated = response
                            
                            if consolidated:
                                print(f"Successfully consolidated with {model_id}")
                                break
                                
                        except Exception as e:
                            print(f"Consolidation error with {model_id}: {str(e)}")
                            time.sleep(1)  # Small delay before trying next model
                    
                    if consolidated:
                        return f"**Consolidated Meeting Summary:**\n\n{consolidated}"
                    else:
                        # If consolidation failed, just return individual summaries
                        return combined_summary
                        
                except Exception as consolidation_error:
                    print(f"All consolidation attempts failed: {str(consolidation_error)}")
                    return f"{combined_summary}\n\n**Note:** Failed to consolidate summaries due to an error."
            
            return combined_summary
            
    except Exception as e:
        error_detail = f"Type: {type(e).__name__}, Message: {str(e)}"
        print(f"Hugging Face API error: {error_detail}")
        traceback.print_exc()
        return f"Hugging Face summarization failed. Error: {error_detail}"

# Fallback summarization that works without API
def fallback_local_summarization(text):
    """Create a basic summary using local processing when API calls fail"""
    try:
        # Very basic extractive summarization
        sentences = text.split('. ')
        
        # If text is too short, return it as is
        if len(sentences) <= 5:
            return text
            
        # Get the first sentence as it often contains key information
        summary = [sentences[0]]
        
        # Add a sentence from the middle
        middle_idx = len(sentences) // 2
        if middle_idx < len(sentences):
            summary.append(sentences[middle_idx])
            
        # Add a sentence from near the end (but not the very end as it might be a sign-off)
        end_idx = max(2, len(sentences) - 3)
        if end_idx < len(sentences):
            summary.append(sentences[end_idx])
        
        return '. '.join(summary)
    except Exception as e:
        print(f"Fallback summarization error: {str(e)}")
        return "Summary generation failed. Please check the full transcript."
    
# Background processing function
def process_in_background(job_id, file_path):
    try:
        # Convert audio if needed
        print(f"[Job {job_id}] Converting audio file")
        processing_jobs[job_id]["status"] = "converting_audio"
        wav_file = convert_audio(file_path)
        
        # Transcribe audio
        print(f"[Job {job_id}] Transcribing audio")
        processing_jobs[job_id]["status"] = "transcribing"
        transcript = transcribe_audio(wav_file)
        
        if transcript.startswith("Transcription failed"):
            processing_jobs[job_id] = {
                "status": "completed",
                "transcript": transcript,
                "summary": "Could not generate summary due to transcription failure."
            }
            return
            
        # Summarize transcript
        print(f"[Job {job_id}] Summarizing transcript")
        processing_jobs[job_id]["status"] = "summarizing"
        summary = summarize_text(transcript)
        
        # Store results
        processing_jobs[job_id] = {
            "status": "completed",
            "transcript": transcript,
            "summary": summary
        }
        
        # Add to history
        new_entry = {
            "name": os.path.basename(file_path),
            "transcript": transcript,
            "summary": summary
        }
        history.append(new_entry)
        save_history(history)
        
        print(f"[Job {job_id}] Processing completed")
    except Exception as e:
        print(f"[Job {job_id}] Error: {str(e)}")
        traceback.print_exc()
        processing_jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }

# Routes
@app.route('/', methods=["GET"])
def index():
    history = load_history()
    return render_template("index.html", transcript="", summary="", history=history, error="")

@app.route('/upload', methods=["POST"])
def upload_file():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded or selected."}), 400

    try:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        processing_jobs[job_id] = {"status": "processing"}
        
        # Start background processing
        thread = threading.Thread(target=process_in_background, args=(job_id, file_path))
        thread.daemon = True  # Allow the thread to exit when the main process exits
        thread.start()
        
        return jsonify({"job_id": job_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/job_status/<job_id>')
def get_job_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({"status": "not_found"})
    return jsonify(processing_jobs[job_id])

@app.route("/history")
def get_history():
    return jsonify(load_history())

@app.route("/clear_history", methods=["POST"])
def clear_history():
    global history
    history = []
    save_history(history)
    return redirect(url_for("index"))

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        transcript = data.get('transcript', '')
        summary = data.get('summary', '')
        
        if not transcript and not summary:
            return jsonify({"error": "No content provided for PDF"}), 400
        
        # Clean up the summary by removing part headers and consolidating
        cleaned_summary = clean_summary_for_pdf(summary)
        
        # Create PDF with reportlab
        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Add custom style for bullet points
        styles.add(ParagraphStyle(name='BulletPoint', 
                                parent=styles['Normal'],
                                leftIndent=20,
                                firstLineIndent=-15))
        
        # Prepare content
        elements = []
        
        # Add title
        elements.append(Paragraph("<b>Meeting Summary Report</b>", styles['Title']))
        elements.append(Spacer(1, 24))
        
        # Add summary section first (more important)
        elements.append(Paragraph("<b>Summary</b>", styles['Heading1']))
        elements.append(Spacer(1, 12))
        
        # Process cleaned summary text with better formatting
        safe_summary = cleaned_summary.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Convert markdown bold to HTML
        safe_summary = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_summary)
        
        # Process summary by paragraphs for better formatting
        for paragraph in safe_summary.split('\n'):
            if not paragraph.strip():
                continue
                
            # Handle bullet points
            if paragraph.strip().startswith('• ') or paragraph.strip().startswith('- '):
                bullet_text = paragraph.strip()[2:]
                elements.append(Paragraph(f"• {bullet_text}", styles['BulletPoint']))
            elif paragraph.strip().startswith('* '):
                bullet_text = paragraph.strip()[2:]
                elements.append(Paragraph(f"• {bullet_text}", styles['BulletPoint']))
            else:
                elements.append(Paragraph(paragraph, styles['Normal']))
            elements.append(Spacer(1, 6))
        
        elements.append(Spacer(1, 24))
        
        # Add transcript section
        elements.append(Paragraph("<b>Full Transcript</b>", styles['Heading1']))
        elements.append(Spacer(1, 12))
        
        # Process transcript text
        safe_transcript = transcript.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Split into paragraphs for better handling
        for paragraph in safe_transcript.split('\n'):
            if paragraph.strip():
                elements.append(Paragraph(paragraph, styles['Normal']))
                elements.append(Spacer(1, 6))
        
        # Build the PDF
        doc.build(elements)
        output.seek(0)
        
        # Return the PDF file
        return send_file(
            output, 
            mimetype='application/pdf',
            as_attachment=True,
            download_name='meeting_summary.pdf'
        )
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        traceback.print_exc()  # Print full stack trace for debugging
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

def clean_summary_for_pdf(raw_summary):
    """Clean up the summary by removing part headers and consolidating text"""
    if not raw_summary:
        return "No summary available."
        
    # Check if this is a consolidated summary already
    if "**Consolidated Meeting Summary:**" in raw_summary:
        # Remove just the header and keep the rest
        cleaned = re.sub(r'\*\*Consolidated Meeting Summary:\*\*\s*', '', raw_summary)
        return cleaned
        
    # Check if we have part summaries
    if "**Part" in raw_summary and "Summary:**" in raw_summary:
        # Remove part headers
        parts = re.split(r'\*\*Part \d+ Summary:\*\*\s*', raw_summary)
        # Remove empty parts
        parts = [p for p in parts if p.strip()]
        
        # Check if there's any "Failed to summarize" messages and remove them
        clean_parts = []
        for part in parts:
            if not part.strip().startswith("Failed to summarize"):
                clean_parts.append(part.strip())
                
        # Join the parts with paragraph breaks
        if clean_parts:
            return "\n\n".join(clean_parts)
        else:
            return "No valid summary parts available."
    
    # If it's a single part or already formatted properly, just return it
    return raw_summary

@app.route("/test_api", methods=["GET"])
def test_api():
    if not huggingface_api_key:
        return jsonify({"status": "error", "message": "Hugging Face API key not configured"})
    
    try:
        # Simple test query
        global hf_client
        if hf_client is None:
            hf_client = InferenceClient(token=huggingface_api_key)
        
        # Model for a simple text generation
        model_id = "gpt2"  # Simple model for quick testing
        
        payload = {
            "inputs": "Hello, this is a test query. Please respond with 'API is working'."
        }
        
        response = call_huggingface_with_retry(hf_client, model_id, payload)
        
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and "generated_text" in response[0]:
                return jsonify({"status": "success", "message": response[0]["generated_text"]})
            else:
                return jsonify({"status": "success", "message": str(response[0])})
        elif isinstance(response, dict) and "generated_text" in response:
            return jsonify({"status": "success", "message": response["generated_text"]})
        else:
            return jsonify({"status": "success", "message": str(response)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "error_type": type(e).__name__,
            "message": str(e)
        })

@app.route("/check_huggingface_key", methods=["GET"])
def check_huggingface_key():
    """Simple endpoint to check if the Hugging Face API key is configured"""
    return jsonify({
        "key_configured": bool(huggingface_api_key),
        "key_masked": f"{huggingface_api_key[:4]}...{huggingface_api_key[-4:]}" if huggingface_api_key and len(huggingface_api_key) > 8 else None
    })

if __name__ == "__main__":
    app.run(debug=True)