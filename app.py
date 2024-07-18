import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time

# Set Streamlit page configuration
st.set_page_config(layout="wide")

def download_youtube_audio(url: str, bitrate: str = '160kbps') -> str:
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

def initialize_model(full_path: str) -> PromptModel:
    """
    Initialize the PromptModel with the given model path.
    
    Args:
        full_path (str): The path to the model file.
    
    Returns:
        PromptModel: The initialized model.
    """
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model: PromptModel) -> PromptNode:
    """
    Initialize the PromptNode with the given model.
    
    Args:
        model (PromptModel): The model to use for the PromptNode.
    
    Returns:
        PromptNode: The initialized PromptNode.
    """
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_and_summarize_audio(file_path: str, prompt_node: PromptNode) -> dict:
    """
    Transcribe and summarize audio using WhisperTranscriber and PromptNode.
    
    Args:
        file_path (str): The path to the audio file.
        prompt_node (PromptNode): The PromptNode for summarization.
    
    Returns:
        dict: The transcription and summarization results.
    """
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="WhisperTranscriber", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="Summarizer", inputs=["WhisperTranscriber"])
    return pipeline.run(file_paths=[file_path])

def main():
    # Set the title and background color
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙, Haystack, Streamlit and ❤️')
    st.markdown('<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by AI Anytime.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        start_time = time.time()  # Start the timer

        # Download video
        file_path = download_youtube_audio(youtube_url)

        # Initialize model and prompt node
        model_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(model_path)
        prompt_node = initialize_prompt_node(model)

        # Transcribe and summarize audio
        output = transcribe_and_summarize_audio(file_path, prompt_node)

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time

        # Display layout with 2 columns
        col1, col2 = st.columns([1, 1])

        # Column 1: Video view
        with col1:
            st.video(youtube_url)

        # Column 2: Summary view
        with col2:
            st.header("Summarization of YouTube Video")
            summary = output["results"][0].split("\n\n[INST]")[0]
            st.write(summary)
            st.write(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

