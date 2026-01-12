import streamlit as st
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    GitProcessor, GitForCausalLM
)
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="AI Image Captioner", page_icon="🤖", layout="wide")

# --- Model Loading Logic ---
@st.cache_resource
def load_ai_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_name == 'BLIP':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    elif model_name == 'ViT-GPT2':
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        # For ViT, we return a dict because it uses a separate tokenizer
        model.to(device)
        return {"model": model, "processor": processor, "tokenizer": tokenizer, "type": "vit"}
    elif model_name == 'GIT':
        processor = GitProcessor.from_pretrained("microsoft/git-base-coco")
        model = GitForCausalLM.from_pretrained("microsoft/git-base-coco")
    
    model.to(device)
    return {"model": model, "processor": processor, "type": "standard"}

# --- Generation Logic ---
def generate_caption(image, model_data, detail_level):
    model = model_data["model"]
    processor = model_data["processor"]
    device = next(model.parameters()).device
    
    max_length = 100 if detail_level == "Detailed" else 50

    if model_data["type"] == "vit":
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=max_length, num_beams=4)
        caption = model_data["tokenizer"].decode(output_ids[0], skip_special_tokens=True)
    else:
        # BLIP and GIT logic
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=max_length)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return caption.capitalize()

# --- UI Layout ---
st.title("🤖 AI Image Caption Generator")
st.markdown("Upload an image and let deep learning models describe it.")

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Choose AI Model", ["BLIP", "ViT-GPT2", "GIT"])
    detail_level = st.radio("Detail Level", ["Short", "Detailed"])
    st.info(f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

uploaded_file = st.file_upload("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if st.button("✨ Generate Caption"):
            with st.spinner(f"Loading {selected_model} and analyzing..."):
                model_data = load_ai_model(selected_model)
                caption = generate_caption(image, model_data, detail_level)
                
                st.subheader("Result:")
                st.success(caption)
                st.button("📋 Copy to Clipboard", on_click=lambda: st.write(f"Copied: {caption}"))
