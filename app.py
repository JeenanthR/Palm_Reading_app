"""AI-Powered Palm Reading & Chatbot - Streamlit app (Task 06: whole workflow in one UI).
Run:  streamlit run app.py
Uploaded images are kept in memory only (st.session_state) - nothing is written to disk."""
import cv2
import numpy as np
import streamlit as st

from core import config as C
from core.palm_features import analyse_palm, locate_hand
from core.validation import validate_upload
from core.report import generate_report
from core.chatbot import PalmChatbot
from core.rag import PalmistryKB

st.set_page_config(page_title="AI Palm Reading (Entertainment)", page_icon="🖐️", layout="wide")
st.title("🖐️ AI-Powered Palm Reading & Chatbot")
st.warning(C.DISCLAIMER)


@st.cache_resource
def get_kb():
    return PalmistryKB()


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


ss = st.session_state
ss.setdefault("analysis", {}); ss.setdefault("report", None); ss.setdefault("bot", None); ss.setdefault("chat", [])

# ------------------------------------------------------------------ Task 01: upload + validation
st.header("1 · Upload both palms")
cols = st.columns(2)
valid = {}
for col, hand in zip(cols, ["left", "right"]):
    with col:
        up = st.file_uploader(f"{hand.title()} palm (JPG/JPEG/PNG)", type=None, key=f"up_{hand}",
                              help="Open palm facing the camera, plain background, bright even light, whole hand + wrist in frame.")
        if up is not None:
            r = validate_upload(up.getvalue(), up.name, locate_hand, hand)
            for e in r.errors: st.error(f"❌ {e}")
            for w in r.warnings: st.warning(f"⚠️ {w}")
            if r.ok:
                st.success(f"✅ {hand.title()} palm accepted")
                st.image(rgb(r.image), caption=f"{hand.title()} palm preview", use_container_width=True)
                valid[hand] = r.image
            with st.expander("Quality metrics"): st.json(r.metrics)

ready = len(valid) == 2
if st.button("🔍 Analyse palms", type="primary", disabled=not ready):
    ss.analysis, ss.report, ss.bot, ss.chat = {}, None, None, []
    with st.spinner("Processing images and detecting features..."):
        for hand, img in valid.items():
            ss.analysis[hand] = analyse_palm(img, hand)
    feats = {h: a["features"] for h, a in ss.analysis.items()}
    with st.spinner("Generating palmistry reading..."):
        ss.report = generate_report(feats, get_kb())
        ss.bot = PalmChatbot(feats, ss.report["markdown"], get_kb())
if not ready:
    st.info("Upload a valid image for BOTH palms to enable analysis.")

# ------------------------------------------------------------------ results
if ss.analysis:
    t1, t2, t3, t4, t5 = st.tabs(["🖼️ Processing (Task 02)", "📍 Features (Task 03)", "📜 Report (Task 04)",
                                  "💬 Chatbot (Task 05)", "🧾 Evidence"])
    with t1:
        for hand, a in ss.analysis.items():
            st.subheader(f"{hand.title()} palm - original vs processed")
            order = [("original", "Aligned original"), ("grayscale", "Grayscale"), ("gaussian", "Gaussian blur"),
                     ("hist_eq", "Histogram equalisation"), ("clahe", "CLAHE"), ("canny", "Canny edges"),
                     ("sobel", "Sobel magnitude"), ("ridge", "Ridge (Sato) filter"), ("threshold", "Threshold"),
                     ("morphology", "Morphology (open+close)")]
            for i in range(0, len(order), 5):
                cs = st.columns(5)
                for c, (k, cap) in zip(cs, order[i:i + 5]):
                    c.image(rgb(a["stages"][k]), caption=cap, use_container_width=True)
    with t2:
        for hand, a in ss.analysis.items():
            st.subheader(f"{hand.title()} palm - detected features (CV only, not interpretation)")
            c1, c2 = st.columns(2)
            c1.image(rgb(a["annotated_original"]), caption="Lines projected on the original photo", use_container_width=True)
            c2.image(rgb(a["annotated_canon"]), caption="Aligned palm: solid = detected, dashed = uncertain", use_container_width=True)
            st.caption(f"Hand localisation: **{a['features']['localisation_method']}**")
            for n in a["features"]["notes"]: st.warning(n)
            st.dataframe([{k: v for k, v in l.items() if k != "source"} for l in a["features"]["lines"]], use_container_width=True)
            st.write("Palm shape (estimate):", a["features"]["palm_shape"])
    with t3:
        if ss.report:
            st.caption(f"Generated with: **{ss.report['mode']}** · specialisation: RAG over curated palmistry knowledge base")
            for w in ss.report["warnings"]: st.warning(w)
            st.markdown(ss.report["markdown"])
            with st.expander("Retrieved knowledge excerpts (traceability)"): st.json(ss.report["retrieved"])
            st.download_button("Download report (.md)", ss.report["markdown"], "palm_report.md")
    with t4:
        st.caption("Ask follow-up questions. Try: *What does my heart line traditionally represent?* / *What differences are visible between my left and right palms?*")
        for role, msg in ss.chat:
            st.chat_message(role).markdown(msg)
        q = st.chat_input("Ask about your reading...")
        if q and ss.bot:
            ss.chat.append(("user", q)); st.chat_message("user").markdown(q)
            with st.spinner("Thinking..."):
                ans = ss.bot.ask(q)
            ss.chat.append(("assistant", ans)); st.chat_message("assistant").markdown(ans)
    with t5:
        st.write("Use this tab for screenshots: both uploads (section 1), processing comparison, annotated lines, report and chat.")
        st.write("Feature JSON passed to the LLM:")
        st.json({h: a["features"] for h, a in ss.analysis.items()})
