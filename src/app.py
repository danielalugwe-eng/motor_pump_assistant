import os

import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Motor Pump Assistant", page_icon="⚙️")
st.title("Motor Pump Assistant")
st.write("Ask the equipment manual anything, or enter vibration sensor feature values to get a prediction.")

if "history" not in st.session_state:
    st.session_state.history = []

if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = "deep_pdf"

if "query" not in st.session_state:
    st.session_state.query = ""

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

mode = st.radio("Interaction mode:", ["Manual Chat", "Sensor Prediction"])

if mode == "Manual Chat":
    st.subheader("Ask the manual")
    st.write("This assistant is rooted in the PDF manual. It will only answer from the provided excerpts and will tell you when the manual does not contain a direct answer.")
    answer_style = st.selectbox("Answer style", ["detailed", "short"], index=0)

    if st.session_state.pending_query:
        st.session_state.query = st.session_state.pending_query
        st.session_state.pending_query = ""

    with st.form("manual_chat_form", clear_on_submit=True):
        query = st.text_input("Ask a question:", key="query")
        submitted = st.form_submit_button("Send")

    cols = st.columns([0.2, 0.8])
    if submitted and query:
        q = query
        with st.spinner("Looking up the manual..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"text": q, "answer_style": answer_style},
                    timeout=30,
                )
                data = resp.json()
            except Exception as exc:
                data = {"route": "error", "answer": f"Request failed: {exc}"}

        st.session_state.history.append({
            "question": q,
            "answer": data.get("answer", ""),
            "route": data.get("route", ""),
            "sources": data.get("sources", []),
            "preview": data.get("context_preview", ""),
            "source_quality": data.get("source_quality", "none"),
            "source_count": data.get("source_count", 0),
            "answer_style": data.get("answer_style", answer_style),
        })

    if cols[0].button("Clear History"):
        st.session_state.history = []

    if cols[1].button("Ask a follow-up") and st.session_state.history:
        last_question = st.session_state.history[-1]["question"]
        st.session_state.pending_query = f"Following up on: {last_question} — can you explain more about this?"
        st.rerun()

    st.markdown("---")

    if not st.session_state.history:
        st.info("No conversation yet — ask the manual a question to get started.")

    for i, item in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}  ")

        quality = item.get("source_quality", "none")
        count = item.get("source_count", 0)
        style = item.get("answer_style", "detailed")
        if quality == "high":
            st.success(f"Source quality: HIGH ({count} sources) | Style: {style}")
        elif quality == "medium":
            st.info(f"Source quality: MEDIUM ({count} sources) | Style: {style}")
        elif quality == "low":
            st.warning(f"Source quality: LOW ({count} source) | Style: {style}")
        else:
            st.error(f"Source quality: NONE (0 sources) | Style: {style}")

        if item.get("sources"):
            st.markdown("**Sources:** " + ", ".join(item["sources"]))

        if item.get("preview"):
            with st.expander("Context preview"):
                st.write(item["preview"])

        st.write("---")

else:
    st.subheader("Vibration Sensor Prediction")
    st.write("Use actual raw CWRU vibration files or enter the extracted feature values manually.")

    sensor_mode = st.radio("Prediction input:", ["Upload raw CWRU .npz", "Enter extracted features"])

    if sensor_mode == "Upload raw CWRU .npz":
        uploaded = st.file_uploader("Upload a CWRU vibration file", type=["npz"])
        signal_key = st.text_input("Signal key inside the .npz file", value="DE")

        if st.button("Predict from raw file") and uploaded is not None:
            with st.spinner("Reading vibration signal and scoring it..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
                    resp = requests.post(
                        f"{API_BASE_URL}/predict_raw",
                        files=files,
                        data={"signal_key": signal_key},
                        timeout=60,
                    )
                    data = resp.json()
                except Exception as exc:
                    data = {"route": "error", "answer": f"Request failed: {exc}"}

            st.markdown(f"**Prediction result:** {data.get('answer', '')}")
            if data.get("source_file"):
                st.caption(f"File: {data.get('source_file')}, signal: {data.get('signal_key')}, windows: {data.get('window_count')}")
            if data.get("features"):
                with st.expander("Extracted feature values"):
                    st.json(data["features"])
            if data.get("prediction"):
                prediction = str(data["prediction"])
                if prediction.lower() == "normal":
                    st.success("Healthy reading: the vibration looks normal.")
                else:
                    st.error(f"Bad reading: the vibration looks like {prediction} fault behavior.")
            confidence = data.get("confidence")
            if confidence is not None:
                if confidence >= 0.8:
                    st.success(f"Confidence badge: HIGH ({confidence:.2f})")
                elif confidence >= 0.6:
                    st.info(f"Confidence badge: MEDIUM ({confidence:.2f})")
                else:
                    st.warning(f"Confidence badge: LOW ({confidence:.2f})")
            with st.expander("How to read this result"):
                st.write("Healthy reading usually means the model predicts `Normal` or the fallback risk score stays low.")
                st.write("Bad reading usually means the model predicts `InnerRace`, `Ball`, or `OuterRace`, or the risk score is high.")
                st.write("Higher RMS, higher Kurtosis, and bigger Crest Factor often mean a rougher vibration signal.")

    else:
        st.write("These are the model-ready features extracted from the raw vibration signal.")
        rms = st.number_input("RMS (root mean square)", value=0.5, format="%.4f")
        kurtosis = st.number_input("Kurtosis", value=1.0, format="%.4f")
        crest_factor = st.number_input("Crest Factor", value=1.0, format="%.4f")
        dominant_freq = st.number_input("Dominant Frequency (Hz)", value=1000.0, format="%.1f")
        spectral_energy = st.number_input("Spectral Energy", value=100.0, format="%.4f")
        spectral_entropy = st.number_input("Spectral Entropy", value=1.0, format="%.4f")

        st.write("These values come from raw vibration windows using the same feature extraction pipeline used in the project.")

        if st.button("Predict condition"):
            payload = {
                "rms": rms,
                "kurtosis": kurtosis,
                "crest_factor": crest_factor,
                "dominant_freq": dominant_freq,
                "spectral_energy": spectral_energy,
                "spectral_entropy": spectral_entropy,
            }
            with st.spinner("Evaluating sensor data..."):
                try:
                    resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
                    data = resp.json()
                except Exception as exc:
                    data = {"route": "error", "answer": f"Request failed: {exc}"}

            st.markdown(f"**Prediction result:** {data.get('answer', '')}")
            if data.get("confidence") is not None:
                st.markdown(f"**Confidence:** {data.get('confidence'):.2f}")
                confidence = float(data.get("confidence"))
                if confidence >= 0.8:
                    st.success(f"Confidence badge: HIGH ({confidence:.2f})")
                elif confidence >= 0.6:
                    st.info(f"Confidence badge: MEDIUM ({confidence:.2f})")
                else:
                    st.warning(f"Confidence badge: LOW ({confidence:.2f})")
            if data.get("prediction"):
                prediction = str(data["prediction"])
                if prediction.lower() == "normal":
                    st.success("Healthy reading: the vibration looks normal.")
                else:
                    st.error(f"Bad reading: the vibration looks like {prediction} fault behavior.")
            with st.expander("How to read this result"):
                st.write("Healthy reading usually means the model predicts `Normal` or the fallback risk score stays low.")
                st.write("Bad reading usually means the model predicts `InnerRace`, `Ball`, or `OuterRace`, or the risk score is high.")
                st.write("Higher RMS, higher Kurtosis, and bigger Crest Factor often mean a rougher vibration signal.")

    st.markdown("---")
    st.info("Upload a raw `.npz` from `data/raw/CWRU_Bearing_NumPy/Data` or enter the extracted vibration features manually.")
