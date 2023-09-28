"""Main file for Streamlit app"""
import streamlit as st
import numpy as np

from PIL import Image
from wasabi import msg

from clip_encoder import CLIPEncoder
from index import BaseIndex
from utils import load_config

config = load_config("config.yaml")

## Resources
@st.cache_resource
def load_model():
    """Load CLIP model"""
    msg.info("Loading CLIP model ...")
    return CLIPEncoder()

@st.cache_resource
def load_index():
    """Load index"""
    msg.info("Loading index ...")
    return BaseIndex(config).load()

@st.cache_resource
def load_idx2img():
    """Load index to image mapping"""
    msg.info("Loading index to image mapping ...")
    return index.idx2image

# Config
st.set_page_config(
        page_title="Patent Search",
        page_icon="🔍",
        layout="wide"
    )

C = 5 # No. of columns

# Load model and index
model = load_model()
index = load_index()
idx2img = load_idx2img()

## Helper Functions
def get_patent_id(filename: str) -> str:
    """
        Get patent id from filename
    """
    if filename.startswith("EP"):
        if len(filename.split("-")) > 4:
            return "-".join(filename.split("-")[:-2])
        else:
            return "EP"+filename[7:14]
    elif filename.startswith("US"):
        return "US"+filename.split("-")[0].split("US")[1]
    elif filename.startswith("WO"):
        return "WO"+filename.split("img")[0][-10:]

def search(query):
    """Search for query"""
    if isinstance(query, str):
        msg.info(f"Searching ... {query}")
        query = model.tokenizer([query])
        query = model.encode(query, text=True)
    else:
        msg.info("Searching for given image ...")
        query = model.preprocess(Image.open(query)).unsqueeze(0)
        query = model.encode(query)

    query = index.normalize(query)
    query = np.array(query, dtype="float32")
    scores, indices = index.faiss_index.search(query, 20)
    results = []
    for idx, sim in zip(indices[0], scores[0]):
        results.append((idx, sim, idx2img[str(idx)]))
    return results

def display(results):
    """Display results"""
    with st.container():

        groups = [results[i:i+C] for i in range(0, len(results), C)]

        cols = st.columns(C)

        for group in groups:
            for i, each in enumerate(group):
                cols[i].image(Image.open(
                    f"{config['dir']}/{each[2]}.png"
                ).resize((2400, 3200)))
                cols[i].text(f"Patent: {get_patent_id(each[2])}")
                cols[i].text(f" Score: {round(each[1], 2)}")

def retrieve(_user_input):
    """Retrieve and display results"""
    with st.columns(3)[1]:
        with st.spinner(text='Retrieving ...'):
            results = search(_user_input)
    display(results)


# Layout
with st.container():

    with st.columns(3)[1]:
        st.header("🔍 Patent Search")

    left_column, right_column = st.columns([3,1])

    # Text input
    IMAGE = None
    with left_column:
        user_input = st.text_input("Enter your query here...", key="query")
        if user_input:
            user_input = f"a patent figure of {user_input}"

    # Image input
    with left_column:

        st.write("OR")
        image_btn = st.checkbox(" Upload an image ")

        if image_btn:
            IMAGE = st.file_uploader("Upload an image...", key="image",
                                        type=['jpg','png','jpeg'])
    with right_column:
        if IMAGE:
            st.image(Image.open(IMAGE).resize((2400, 3200)))
            user_input = IMAGE

    # Retrieve and display results
    if user_input:
        retrieve(user_input)
