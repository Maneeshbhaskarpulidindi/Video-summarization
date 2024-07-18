from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer  # Ensure this module is available in your path

# Define constants
YOUTUBE_URL = "https://www.youtube.com/watch?v=h5id4erwD4s"
AUDIO_BITRATE = '160kbps'
FULL_MODEL_PATH = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
SUMMARY_PROMPT = "deepset/summarization"

def download_youtube_audio(url: str, bitrate: str = AUDIO_BITRATE) -> str:
    """
    Download audio from a YouTube video at the specified bitrate.
    
    Args:
        url (str): The URL of the YouTube video.
        bitrate (str): The desired audio bitrate.
    
    Returns:
        str: The file path to the downloaded audio.
    """
    yt = YouTube(url)
    audio_stream = yt.streams.filter(abr=bitrate).last()
    return audio_stream.download()

# Initialize WhisperTranscriber for audio transcription
transcriber = WhisperTranscriber()

# Initialize the model for summarization
model = PromptModel(
    model_name_or_path=FULL_MODEL_PATH,
    invocation_layer_class=LlamaCPPInvocationLayer,
    use_gpu=False,
    max_length=512
)

# Initialize the prompt node for summarization tasks
prompt_node = PromptNode(
    model_name_or_path=model,
    default_prompt_template=SUMMARY_PROMPT,
    use_gpu=False
)

# Download the audio from the YouTube video
audio_file_path = download_youtube_audio(YOUTUBE_URL)

# Create and configure the pipeline
pipeline = Pipeline()
pipeline.add_node(component=transcriber, name="WhisperTranscriber", inputs=["File"])
pipeline.add_node(component=prompt_node, name="Summarizer", inputs=["WhisperTranscriber"])

# Run the pipeline with the downloaded audio file
results = pipeline.run(file_paths=[audio_file_path])

# Output the results
summary = results["results"][0].split("\n\n[INST]")[0]
print("Summary of the video:")
print(summary)

