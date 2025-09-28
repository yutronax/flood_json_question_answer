st.title("FloodNet Image Analysis and Question Answering")

# Initialize models (load once)
@st.cache_resource
def load_models():
    obj_detection_model_instance = floodnet_colors()
    obj_detection_model = obj_detection_model_instance.load_model()
    t5_model_instance = T5Model()
    t5_model, t5_tokenizer, t5_device = t5_model_instance.load_trained_model()
    return obj_detection_model_instance, obj_detection_model, t5_model_instance, t5_model, t5_tokenizer, t5_device

obj_detection_model_instance, obj_detection_model, t5_model_instance, t5_model, t5_tokenizer, t5_device = load_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    image_path = f"/content/{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Perform object detection
    st.subheader("Object Detection Results")
    save_path = "/content/output"
    os.makedirs(save_path, exist_ok=True)

    # Pass the loaded model and the temporary image path
    detection_result = obj_detection_model_instance.process_single_image(obj_detection_model, image_path, save_path)

    if detection_result and detection_result['objects']:
        st.write("Detected Objects:")
        for obj_type, count in detection_result['objects'].items():
            st.write(f"- {obj_type.replace('_', ' ')}: {count}")

        # Display the generated mask image
        mask_filename = os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        mask_path = os.path.join(save_path, mask_filename)
        if os.path.exists(mask_path):
            st.image(mask_path, caption="Segmentation Mask", use_container_width=True)
    else:
        st.write("No significant objects detected.")

    # Question Answering
    st.subheader("Ask a Question about the Image")

    default_questions = ["burda sel var mı?", "kaç bina var", "binalar selden etkilenmiş mi?", "none"]
    question = st.selectbox("Select a question or type your own:", default_questions)

    if question == "none":
        custom_question = st.text_input("Enter your question:")
        if custom_question:
            question_to_ask = custom_question
        else:
            question_to_ask = None
            st.info("Please enter a question.")
    else:
        question_to_ask = question

    if question_to_ask and detection_result and detection_result['objects']:
        st.subheader("Answer")
        # Pass the loaded T5 model components
        answer = t5_model_instance.test_model(question_to_ask, detection_result['objects'], t5_model, t5_tokenizer, t5_device)
        st.write(answer)
    elif question_to_ask and (not detection_result or not detection_result['objects']):
         st.warning("Cannot answer the question as no objects were detected in the image.")
