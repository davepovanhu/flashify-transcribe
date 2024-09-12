from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow CORS for all origins (you can specify certain origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configure Google Generative AI API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file, transcribe it using Google Generative AI, 
    and generate a summarized version of the transcription.
    """
    try:
        # Save the uploaded file locally
        file_location = f"./{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Upload the audio file using Generative AI
        audio_file = genai.upload_file(path=file_location)

        # Generate the transcription
        model = genai.GenerativeModel("gemini-1.5-flash")
        transcription_result = model.generate_content(
            [audio_file, "Generate a transcript of the speech."]
        )

        # Summarize the transcription
        summary_result = model.generate_content(
            [transcription_result.text, "Summarize the speech, focusing on important information for students."]
        )

        # Clean up local file
        os.remove(file_location)

        # Return the results
        return {
            "transcription": transcription_result.text,
            "summary": summary_result.text,
        }

    except Exception as e:
        return {"error": str(e)}

