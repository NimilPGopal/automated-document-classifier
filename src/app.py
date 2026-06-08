import os
import json
import torch
import streamlit as st
import re
from PIL import Image
from torchvision import transforms
from utils import get_model, get_transforms, GradCAM, generate_gradcam_overlay

# Set page config
st.set_page_config(
    page_title="AI Document Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .header-banner {
        padding: 30px;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.08) 50%, rgba(236, 72, 153, 0.08) 100%);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.02);
    }
    
    .title-text {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #9333ea, #ec4899);
        -webkit-background-clip: text;
        margin: 0;
        padding-bottom: 6px;
        letter-spacing: -0.5px;
    }
    
    .subtitle-text {
        font-size: 1.15rem;
        color: #718096;
        margin-top: 6px;
        margin-bottom: 0px;
        font-weight: 400;
    }
    
    .interactive-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.02);
        backdrop-filter: blur(8px);
        margin-bottom: 25px;
    }
    
    .sidebar-stats-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .sidebar-stat-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #a0aec0;
        margin-bottom: 4px;
        font-weight: 600;
    }
    
    .sidebar-stat-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .pred-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .pred-class {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 6px 0;
        letter-spacing: -0.5px;
    }
    
    .pred-confidence {
        font-size: 1.25rem;
        opacity: 0.95;
        font-weight: 500;
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        transition: background-color 0.2s ease;
    }
    .progress-container:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .progress-bar-bg {
        background-color: rgba(226, 232, 240, 0.15);
        border-radius: 10px;
        height: 10px;
        width: 100%;
        overflow: hidden;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease-in-out;
    }
            
    [data-testid="stFileUploader"] {
        border-radius: 18px;
        padding: 20px;
        background: rgba(99,102,241,0.04);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background: rgba(99,102,241,0.08);
    }

    [data-testid="stFileUploader"] section {
        padding: 20px;
    }

    [data-testid="stFileUploader"] small {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if model and mapping files exist
def check_setup():
    model_exists = os.path.exists('models/document_classifier.pth')
    mapping_exists = os.path.exists('models/class_mapping.json')
    return model_exists and mapping_exists

# Load model and class mapping
@st.cache_resource
def load_classifier():
    with open('models/class_mapping.json', 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=len(class_names), pretrained=False)
    checkpoint = torch.load(
        'models/document_classifier.pth',
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)    
    model = model.to(device)
    model.eval()
    
    return model, class_names, device

# Helper to load sample test images
def get_sample_images():
    if os.path.exists('data/splits.json'):
        with open('data/splits.json', 'r') as f:
            splits = json.load(f)
        test_split = splits.get('test', [])
        
        # Group test set samples by class name
        samples_by_class = {}
        for path, label in test_split:
            parts = os.path.normpath(path).split(os.sep)
            if len(parts) >= 2:
                cat_name = parts[-2]
                if cat_name not in samples_by_class:
                    samples_by_class[cat_name] = []
                if len(samples_by_class[cat_name]) < 4:  # up to 4 samples per category
                    samples_by_class[cat_name].append(path)
        return samples_by_class
    return {}

# Render top header banner
st.markdown("""
<div class='header-banner'>
    <h1 class='title-text'>Automated Document Classifier</h1>
    <p class='subtitle-text'>An Intelligent Document Analysis System Leveraging Transfer Learning and Grad-CAM Visualization.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "85.1%")

with col2:
    st.metric("Classes", "10")

with col3:
    st.metric("Dataset", "3482")

with col4:
    st.metric("Architecture", "MobileNetV2")

if not check_setup():
    st.error("⚠️ Model checkpoints or configurations not found!")

else:
    # Load resources
    model, class_names, device = load_classifier()
    _, val_transform = get_transforms()
    
    # Sidebar Setup
    st.sidebar.markdown("### ⚙️System Configuration")
    
    st.sidebar.markdown("""
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>
            MODEL PERFORMANCE
        </div>
        <div class='sidebar-stat-val'>
            Accuracy: 85.1%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>
            DATASET SIZE
        </div>
        <div class='sidebar-stat-val'>
            3482 Images
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    st.sidebar.markdown(f"""
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>Network Architecture</div>
        <div class='sidebar-stat-val'>Convolutional Neural Network</div>
        <div class='sidebar-stat-val'># MobileNetV2</div>
    </div>
    <div class='sidebar-stats-card'>
        <div class='sidebar-stat-label'>Outputs / Categories</div>
        <div class='sidebar-stat-val'>{len(class_names)} Classes</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    ### Model Details:

    - Framework: PyTorch
    - Explainability: Grad-CAM
    - Transfer Learning
    - GPU Accelerated
    - Image Size: 384×384
    """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([" ♻️Document Analysis", "📊 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.markdown("")
        image = None
        st.markdown("### 📤 Upload Document")

        uploaded_file = st.file_uploader(
            "Upload Document Image",
            type=["png", "jpg", "jpeg", "tiff", "tif"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

                
        if image is not None:
            col1, col2 = st.columns([1.1, 0.9])
            
            with col1:
                st.markdown("")
                st.markdown("### Document Analysis & Highlights")
                
                if image.mode != 'RGB':
                    img_input = image.convert('RGB')
                else:
                    img_input = image
                
                # Transform image to tensor
                tensor_img = val_transform(img_input).unsqueeze(0).to(device)
                tensor_img.requires_grad = True
                
                # Predict and compute Grad-CAM activations
                with st.spinner("Executing forward-backward passes to calculate spatial activation maps..."):
                    with torch.set_grad_enabled(True):
                        outputs = model(tensor_img)
                        probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
                        
                        top_idx = probs.argmax()
                        top_class = class_names[top_idx]
                        top_conf = probs[top_idx] * 100
                        
                        # Generate Grad-CAM heatmap for the top class
                        target_layer = model.features[-1]
                        cam = GradCAM(model, target_layer)
                        try:
                            heatmap, _ = cam(tensor_img, target_category=int(top_idx))
                            # Resize original image to standard 450x450 for aligned layout
                            original_resized = img_input.resize((450, 450), Image.Resampling.BILINEAR)
                            overlaid_image, heatmap_colored = generate_gradcam_overlay(original_resized, heatmap, alpha=0.45)
                        except Exception as e:
                            st.error(f"Failed to generate Grad-CAM explanation: {e}")
                            overlaid_image = None
                            original_resized = img_input
                        finally:
                            cam.remove_hooks()
                
                # Visual view selection
                view_mode = st.segmented_control(
                    "Select Visualization View",
                    options=["Original", "Grad-CAM", "Compare"],
                    default="Compare"
                )
                
                if view_mode is None:
                    view_mode = "Grad-CAM"
                    
                st.markdown("<div style='margin-top:15px; text-align:center;'>", unsafe_allow_html=True)
                if view_mode == "Original":
                    st.image(original_resized, caption="Original Document Layout", width="stretch")
                elif view_mode == "Grad-CAM" and overlaid_image is not None:
                    st.image(overlaid_image, caption="Neural Activation Heatmap (Grad-CAM)", width="stretch")
                else:
                    col_side1, col_side2 = st.columns(2)
                    with col_side1:
                        st.image(original_resized, caption="Original", width="stretch")
                    with col_side2:
                        if overlaid_image is not None:
                            st.image(overlaid_image, caption="Grad-CAM", width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
                
                if overlaid_image is not None:
                    st.markdown("""
                    <div style='font-size: 0.9rem; color: #718096; margin-top: 15px; border-left: 3px solid #6366f1; padding-left: 10px;'>
                        <strong>Layout Focus</strong>: The warm/red spots represent page layout segments (e.g., mail headers, signature blocks, paragraphs, logos) that the model focused on when classifying this document.
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("")
                st.markdown("### Class Prediction:")
                
                # Render prediction box
                st.markdown(f"""
                <div class='pred-box'>

                <div style='font-size:0.85rem;letter-spacing:1.5px;text-transform:uppercase;'>
                Prediction
                </div>

                <div class='pred-class'>
                {top_class}
                </div>

                <div class='pred-confidence'>
                Confidence: {top_conf:.2f}%
                </div>

                <hr style="margin:15px 0;opacity:.3;">

                <div style="
                display:flex;
                justify-content:space-around;
                font-size:.9rem;
                ">

                <div>
                <b>Model</b><br>
                MobileNetV2
                </div>

                <div>
                <b>Classes</b><br>
                10
                </div>

                </div>

                </div>
                """, unsafe_allow_html=True)
                
                # Confidence Breakdown list
                st.markdown("#### Confidence Breakdown:")
                sorted_indices = probs.argsort()[::-1][:2]

                for idx in sorted_indices:
                    c_name = class_names[idx]
                    c_prob = probs[idx]
                    
                    st.markdown(f"""
                    <div class='progress-container'>
                        <div class='progress-label'>
                            <span>{c_name}</span>
                            <span>{c_prob*100:.2f}%</span>
                        </div>
                        <div class='progress-bar-bg'>
                            <div class='progress-bar-fill' style='width: {c_prob*100}%;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
            text-align:center;
            padding:60px;
            opacity:0.7;
            ">
                <h3>Ready for Analysis</h3>
                <p>
                Upload a document image to get started.
                </p>
            </div>
            """, unsafe_allow_html=True)   

    with tab2:
        st.markdown("### Model Evaluation Summary")
        
        # Check if evaluation report exists
        if os.path.exists('reports/evaluation_report.md'):
            with open('reports/evaluation_report.md', 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Show the generated evaluation images
            col_a, col_b = st.columns(2)
            with col_a:
                if os.path.exists('reports/confusion_matrix.png'):
                    st.image('reports/confusion_matrix.png', caption='Confusion Matrix Heatmap', width="stretch")
            with col_b:
                if os.path.exists('reports/precision_recall.png'):
                    st.image('reports/precision_recall.png', caption='Precision-Recall Curves', width="stretch")
            
            st.markdown("---")
            if os.path.exists('reports/training_curves.png'):
                st.image('reports/training_curves.png', caption='Training & Validation History', width="stretch")
                
            st.markdown("---")
            clean_report = re.sub(r'!\[.*?\]\(.*?\)', '', report_content)
            st.markdown(clean_report)
        else:
            st.warning("Metric charts not found.")
            
    with tab3:
        st.markdown("### Tobacco3482 Dataset Details")
        st.markdown("""
        The **Tobacco3482** dataset is a standard dataset of scanned document images from the *Truth Tobacco Industry Documents* archive. 
        It consists of **3,482 document images** categorized into **10 classes** representing common types of files in commercial or legal environments:
        
        - **ADVE** (Advertisement): Visual advertisements, brochures, flyers.
        - **Email**: Scanned emails, structured headers (From, To, Date, Subject).
        - **Form**: Structured questionnaires, sheets with grids, tables, and rows.
        - **Letter**: Signed business correspondence on organization letterheads.
        - **Memo**: Scanned internal memo templates.
        - **News**: Press releases, news logs, columns of text.
        - **Note**: Scribbles, handwritten notes, message pads.
        - **Report**: Formal reports, cover pages, indexes.
        - **Resume**: Professional resumes and curriculum vitae templates.
        - **Scientific**: Academic, medical, and scientific papers with columns, figures, and reference list formats.
        
        #### Model Implementation Details:
        - **Base Model:** Pre-trained MobileNetV2 (via ImageNet weights), facilitating fast converging transfer learning.
        - **Loss function:** CrossEntropyLoss.
        - **Image Dimensions:** Normalized to `224x224` resolution.
        - **Train / Val / Test Split:** 80% / 10% / 10% stratified split.
        """)
