"""
XAI-Powered DoS Detection & Mitigation Dashboard
==================================================
Entry point. Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="XAI DoS Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Top navigation — flat list so all links are directly visible (no dropdowns)
pages = [
    st.Page("pages/dashboard.py", title="Dashboard", default=True),
    st.Page("pages/analyze.py", title="Analyze"),
    st.Page("pages/about.py", title="About"),
]

pg = st.navigation(pages, position="top")
pg.run()
