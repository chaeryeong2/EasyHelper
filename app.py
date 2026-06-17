import re
import streamlit as st
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

st.set_page_config(page_title="Easy Helper", page_icon="🧓")
st.title("Easy Helper")
st.write("AI Digital Assistant for Seniors - OCR + VLM Prototype")


@st.cache_resource
def load_vlm():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    processor = AutoProcessor.from_pretrained(model_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # MPS or CPU 자동 선택
    )
    return processor, model


def generate_guidance(image: Image.Image, user_goal: str) -> tuple[str, str]:
    """
    이미지와 사용자 목표를 받아 (감지된 텍스트, 안내) 를 반환합니다.
    VLM이 이미지를 직접 읽고 안내를 생성합니다.
    """
    processor, model = load_vlm()

    # --- 1단계: 화면 텍스트 추출 ---
    ocr_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "List all the text you can read in this kiosk screen image. "
                    "Include menu item names, prices, buttons, and category labels. "
                    "Just list them separated by commas, nothing else."
                )},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        ocr_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(ocr_messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    screen_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    # --- 2단계: 안내 생성 ---
    guidance_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"This is a kiosk screen. The user wants to: {user_goal}\n\n"
                    "STRICT RULES:\n"
                    "1. Only use menu items and buttons visible in this image.\n"
                    "2. Write exactly 5 short numbered steps.\n"
                    "3. Each step must be ONE sentence, maximum 12 words.\n"
                    "4. Use simple language for elderly users.\n"
                    "5. Do NOT make up items or buttons not visible in the image.\n"
                    "6. Write all steps in Korean.\n\n"
                    "Write the 5 steps now:"
                )},
            ],
        }
    ]

    text_input2 = processor.apply_chat_template(
        guidance_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs2, _ = process_vision_info(guidance_messages)
    inputs2 = processor(
        text=[text_input2],
        images=image_inputs2,
        padding=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids2 = model.generate(**inputs2, max_new_tokens=200, do_sample=False,
                                      repetition_penalty=1.3)
    new_tokens2 = output_ids2[0][inputs2["input_ids"].shape[-1]:]
    guidance = processor.decode(new_tokens2, skip_special_tokens=True).strip()

    # 포맷 정규화 "1/ ..." → "1. ..."
    guidance = re.sub(r"(\d)\s*/\s*", r"\1. ", guidance)
    guidance = re.sub(r" (\d\.)", r"\n\1", guidance)

    return screen_text, guidance.strip()


# ── UI ──────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload kiosk screen image",
    type=["png", "jpg", "jpeg"]
)

user_goal = st.text_input(
    "What do you want to do?",
    placeholder="Example: I want to buy a Big Whopper"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Screen")
    st.image(image, use_container_width=True)

if st.button("Generate Guidance"):
    if not uploaded_file:
        st.error("Please upload an image first.")
    elif not user_goal:
        st.error("Please enter your goal.")
    else:
        with st.spinner("Analyzing screen with VLM..."):
            screen_text, guidance = generate_guidance(image, user_goal)

        st.subheader("Detected Text")
        st.write(screen_text)

        st.subheader("Step-by-Step Guidance")
        st.write(guidance)