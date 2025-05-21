# MINDS

MINDS is a web application that allows users to upload audio recordings (e.g., meetings, interviews), automatically transcribe them, and generate AI-powered summaries. The app also enables users to download structured PDF reports containing the transcript and summary.

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage) 

---

## Description

**MINDS** is a **Flask-based** application designed to streamline the process of converting spoken content into written text and concise summaries. It supports a wide range of audio formats, leverages **OpenAI Whisper** and **Hugging Face APIs** for transcription and summarization, and generates downloadable PDF reports with structured meeting content.

---

## Features

- ✅ Upload and transcribe audio files  
- 🧠 Automatic summarization using Hugging Face models  
- 🔁 Supports non-WAV audio conversion and preprocessing  
- 🔄 Background processing with job status updates  
- 🕘 View history of previous transcriptions and summaries  
- 📄 Download clean, structured PDF reports (with bullet points and formatted text)

---

## Installation

To run **MINDS** locally, follow these steps:

### 1. Clone the repository

git clone https://github.com/your-username/MINDS.git

### 2. Navigate to the project directory

cd audio-summarizer
### 3. Create a virtual environment and activate it

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

### 4. Install the dependencies

pip install -r requirements.txt

### 5. Set your Hugging Face API key

Create a .env file in the root directory:
HUGGINGFACE_API_KEY=your_huggingface_api_key

### 6. Run the app

python app.py

### 7. Access the application

Open your browser and go to:
http://localhost:5000

---

## Usage
Once the application is running, users can:

🎤 Click Choose File to upload an audio file.

⏳ Wait for the file to be transcribed and summarized.

👀 View the transcript and summary on the homepage.

📥 Click Download PDF to get a structured report.

🧹 Click Clear History to remove saved summaries and transcripts.

