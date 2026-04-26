import re

import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(page_title="Easy Helper", page_icon="🧓")

st.title("Easy Helper")
st.write("AI Digital Assistant for Seniors - OCR + LLM Prototype")


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)


@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    return tokenizer, model


def preprocess_image(image):
    return image.convert("RGB")


def extract_text_from_image(image):
    reader = load_ocr()
    image_array = np.array(image)
    results = reader.readtext(image_array)
    texts = []
    for bbox, text, confidence in results:
        if confidence >= 0.25:
            texts.append(text)
    return " ".join(texts).strip()


def find_best_menu(screen_text, user_goal):
    from difflib import SequenceMatcher

    # set 대신 list — 입력 순서 유지 ("big king" → "big king", not "king big")
    all_words = re.findall(r"[a-zA-Z]+", user_goal.lower())
    stopwords = {"i", "want", "to", "buy", "order", "get", "a", "an", "the",
                 "please", "would", "like", "have", "some", "one", "meal", "menu", "food"}
    goal_keywords = [w for w in all_words if w not in stopwords]

    words = re.findall(r"[A-Za-z0-9]+", screen_text)
    ignore_words = {"back", "restart", "total", "menu", "now", "order",
                    "explore", "our", "the", "and", "with", "price",
                    "home", "all", "s", "no", "yes"}
    clean_words = [w for w in words if w.lower() not in ignore_words
                   and not w.replace(".", "").isdigit() and len(w) >= 2]

    candidates = []
    for n in [3, 2, 1]:
        for i in range(len(clean_words) - n + 1):
            candidates.append(" ".join(clean_words[i:i+n]))

    best, best_score = None, 0.0
    goal_phrase = " ".join(goal_keywords)

    for candidate in candidates:
        c_lower = candidate.lower()
        score = SequenceMatcher(None, goal_phrase.lower(), c_lower).ratio()
        matched_kw = sum(1 for kw in goal_keywords if kw in c_lower)
        score += matched_kw * 0.3
        if goal_phrase.lower() in c_lower or c_lower in goal_phrase.lower():
            score += 0.5
        if score > best_score:
            best_score = score
            best = candidate

    return best if best_score >= 0.5 else None


def fallback_guidance(screen_text, user_goal):
    best_menu = find_best_menu(screen_text, user_goal)
    step2 = (f'Tap the "{best_menu}" button or menu.' if best_menu
             else "Choose the menu category that looks closest to your goal.")
    return "\n".join([
        "1. Look at the main menu area on the screen.",
        f"2. {step2}",
        "3. Choose the specific item you want.",
        "4. Check the item name, quantity, and price carefully.",
        "5. Tap the order, cart, or next button.",
        "6. Before payment, check the total price carefully."
    ])


def generate_guidance(screen_text, user_goal):
    best_menu = find_best_menu(screen_text, user_goal)
    menu_hint = best_menu if best_menu else "the most relevant menu item"

    system_prompt = (
        "You are a helpful assistant guiding older adults through kiosk ordering. "
        "STRICT RULES:\n"
        "1. Only use information visible in the screen text. Do NOT invent buttons, prices, or steps.\n"
        "2. Always respond with exactly 5 clear, simple numbered steps.\n"
        "3. Each step must be ONE short sentence. Maximum 12 words per step.\n"
        "4. Use plain, simple language suitable for elderly users.\n"
        "5. Never mention credit card details.\n"
        "6. If unsure, say 'Look for [item] on the screen' instead of making something up."
    )

    user_prompt = (
        f"Here is the exact text visible on the kiosk screen:\n{screen_text}\n\n"
        f"The user's goal: {user_goal}\n"
        f"The most relevant menu item found on screen: {menu_hint}\n\n"
        f"Write exactly 5 SHORT numbered steps to help this elderly person order '{menu_hint}'.\n"
        f"Each step: max 12 words. Only use buttons and items from the screen text above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        tokenizer, model = load_llm()
        device = next(model.parameters()).device

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if result:
            # "1/ ... 2/ ..." 형식을 "1. ...\n2. ..." 형식으로 정규화
            result = re.sub(r"(\d)\s*/\s*", r"\1. ", result)
            # 숫자 앞에 줄바꿈 추가
            result = re.sub(r" (\d\.)", r"\n\1", result)
            return result.strip()

    except Exception as e:
        st.warning(f"LLM error: {e}")

    return fallback_guidance(screen_text, user_goal)


# ── UI ──────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload kiosk screen image",
    type=["png", "jpg", "jpeg"]
)

user_goal = st.text_input(
    "What do you want to do?",
    placeholder="Example: I want to buy Big Whopper"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    st.subheader("Uploaded Screen")
    st.image(processed_image, use_container_width=True)

if st.button("Generate Guidance"):
    if not uploaded_file:
        st.error("Please upload an image first.")
    elif not user_goal:
        st.error("Please enter your goal.")
    else:
        with st.spinner("Analyzing screen..."):
            screen_text = extract_text_from_image(processed_image)
            if not screen_text:
                screen_text = "No readable text found."
            guidance = generate_guidance(screen_text, user_goal)

        st.subheader("Detected Text")
        st.write(screen_text)

        st.subheader("Step-by-Step Guidance")
        st.write(guidance)