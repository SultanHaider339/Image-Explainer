import streamlit as st
import torch
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Page Configuration
st.set_page_config(page_title="AI Image Captioner", layout="centered")

# --- Model Loading (Cached to RAM) ---
@st.cache_resource
def load_model():
    # Loading Salesforce BLIP - a versatile, free pre-trained model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def generate_caption(image):
    processor, model, device = load_model()
    # Convert image to RGB (removes alpha channel from PNGs)
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    
    return processor.decode(out[0], skip_special_tokens=True)

# --- UI Layout ---
st.title("📸 AI Image Captioner")
st.markdown("Describe any image using a local Deep Learning model.")

# Input selection
option = st.radio("Choose Input Method:", ["Upload from PC", "Real-time Camera", "Image URL"])

img_input = None

if option == "Upload from PC":
    uploaded_file = st.file_uploader("Select Image", type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_file:
        img_input = Image.open(uploaded_file)

elif option == "Real-time Camera":
    cam_file = st.camera_input("Snap a photo")
    if cam_file:
        img_input = Image.open(cam_file)

elif option == "Image URL":
    url = st.text_input("Paste URL (e.g., https://example.com/photo.jpg)")
    if url:
        try:
            response = requests.get(url, timeout=10)
            img_input = Image.open(io.BytesIO(response.content))
        except Exception as e:
            st.error(f"Error loading URL: {e}")

# --- Processing ---
if img_input:
    st.image(img_input, caption="Preview", use_container_width=True)
    
    if st.button("✨ Generate Caption"):
        with st.spinner("AI is analyzing the image..."):
            try:
                caption = generate_caption(img_input)
                st.success(f"**Description:** {caption.capitalize()}")
            except Exception as e:
                st.error(f"Processing error: {e}")

st.divider()
st.caption("Running locally with Transformers & PyTorch.")
