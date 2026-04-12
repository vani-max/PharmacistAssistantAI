"""
app.py  --  Pharmacist Assistant AI  |  Professional Dashboard
Powered by: FDA NDC Database (71,000+ drugs) + Hybrid AI Engine

Run:  streamlit run app.py
"""

import streamlit as st
import sys
import os
import time
import json
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from core.drug_search import (
    search_drugs,
    get_alternatives,
    check_interactions,
    get_drug_stats,
    expand_abbreviation,
)
from core.prescription_parser import parse_prescription

# ---- Page Config ----
st.set_page_config(
    page_title="PharmAssist AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Custom CSS for professional look ----
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ---- Global Reset ---- */
        * { font-family: 'Inter', sans-serif !important; }
        .stApp {
            background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
        }
        .block-container { max-width: 1400px; padding-top: 2rem; }

        /* ---- Hide defaults ---- */
        #MainMenu, footer, header { visibility: hidden; }

        /* ---- Hero Header ---- */
        .hero-header {
            background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(147,51,234,0.08) 100%);
            border: 1px solid rgba(59,130,246,0.15);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            font-weight: 400;
            margin-top: 0.5rem;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(34,197,94,0.15);
            color: #4ade80;
            border: 1px solid rgba(34,197,94,0.3);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        /* ---- Metric Cards ---- */
        .metric-card {
            background: rgba(30,41,59,0.6);
            border: 1px solid rgba(71,85,105,0.3);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border-color: rgba(59,130,246,0.5);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(59,130,246,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #f1f5f9;
            margin: 0;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.3rem;
        }

        /* ---- Section Headers ---- */
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #e2e8f0;
            margin: 1.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(59,130,246,0.3);
        }

        /* ---- Result Cards ---- */
        .drug-card {
            background: rgba(30,41,59,0.5);
            border: 1px solid rgba(71,85,105,0.3);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.8rem;
            transition: all 0.2s ease;
        }
        .drug-card:hover {
            border-color: rgba(59,130,246,0.4);
            background: rgba(30,41,59,0.7);
        }
        .drug-name {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 0;
        }
        .drug-generic {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 2px;
        }
        .drug-meta {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }
        .drug-tag {
            display: inline-block;
            background: rgba(59,130,246,0.1);
            color: #93c5fd;
            border: 1px solid rgba(59,130,246,0.2);
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .drug-tag-alt {
            background: rgba(168,85,247,0.1);
            color: #c4b5fd;
            border-color: rgba(168,85,247,0.2);
        }
        .score-bar {
            height: 4px;
            border-radius: 2px;
            background: rgba(71,85,105,0.3);
            margin-top: 0.6rem;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            transition: width 0.5s ease;
        }

        /* ---- Alert Cards ---- */
        .alert-critical {
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.3);
            border-left: 4px solid #ef4444;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }
        .alert-high {
            background: rgba(249,115,22,0.08);
            border: 1px solid rgba(249,115,22,0.3);
            border-left: 4px solid #f97316;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }
        .alert-moderate {
            background: rgba(234,179,8,0.08);
            border: 1px solid rgba(234,179,8,0.3);
            border-left: 4px solid #eab308;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }
        .alert-title {
            font-weight: 700;
            font-size: 0.9rem;
            margin: 0;
        }
        .alert-desc {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        .alert-critical .alert-title { color: #fca5a5; }
        .alert-high .alert-title { color: #fdba74; }
        .alert-moderate .alert-title { color: #fde047; }

        /* ---- Confidence Indicator ---- */
        .confidence-high { color: #4ade80; }
        .confidence-medium { color: #fbbf24; }
        .confidence-low { color: #f87171; }

        /* ---- Processed Rx Card ---- */
        .rx-card {
            background: rgba(30,41,59,0.5);
            border: 1px solid rgba(71,85,105,0.3);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.8rem;
        }
        .rx-medicine {
            font-size: 1rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .rx-detail {
            display: inline-block;
            background: rgba(34,197,94,0.1);
            color: #86efac;
            border: 1px solid rgba(34,197,94,0.2);
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-right: 6px;
            margin-top: 6px;
        }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background: rgba(15,23,42,0.95);
            border-right: 1px solid rgba(71,85,105,0.3);
        }
        [data-testid="stSidebar"] .stMarkdown p {
            color: #94a3b8;
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: rgba(30,41,59,0.5);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid rgba(71,85,105,0.3);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
            padding: 8px 20px;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(59,130,246,0.15) !important;
            color: #93c5fd !important;
        }

        /* ---- Input ---- */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(30,41,59,0.6) !important;
            border: 1px solid rgba(71,85,105,0.4) !important;
            color: #e2e8f0 !important;
            border-radius: 10px !important;
            font-size: 0.95rem !important;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: rgba(59,130,246,0.6) !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(59,130,246,0.3) !important;
        }

        /* ---- Audit Log ---- */
        .audit-entry {
            background: rgba(30,41,59,0.3);
            border-left: 3px solid #3b82f6;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: #cbd5e1;
        }
        .audit-time {
            font-size: 0.75rem;
            color: #64748b;
        }

        /* ---- Animate on load ---- */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .hero-header, .metric-card, .drug-card, .rx-card {
            animation: fadeIn 0.5s ease forwards;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- Session state init ----
def init_state():
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    if "interaction_drugs" not in st.session_state:
        st.session_state.interaction_drugs = []
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []


def add_audit(action: str, detail: str):
    st.session_state.audit_log.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "detail": detail,
        },
    )
    # Keep last 50
    st.session_state.audit_log = st.session_state.audit_log[:50]


# ---- Render Functions ----

def render_header():
    st.markdown(
        """
        <div class="hero-header">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
                <div>
                    <p class="hero-title">PharmAssist AI</p>
                    <p class="hero-subtitle">
                        Intelligent Prescription Analysis &bull; Drug Matching &bull; Interaction Safety
                    </p>
                </div>
                <div>
                    <span class="hero-badge">POWERED BY FDA NDC DATABASE</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics():
    stats = get_drug_stats()
    cols = st.columns(5)
    metrics = [
        (f"{stats['total_drugs']:,}", "DRUGS IN DATABASE"),
        (f"{stats['unique_generics']:,}", "UNIQUE GENERICS"),
        (f"{stats['unique_brands']:,}", "UNIQUE BRANDS"),
        (f"{stats['total_alternatives']:,}", "ALTERNATIVE LINKS"),
        (f"{stats['interaction_rules']}", "SAFETY RULES"),
    ]
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value">{value}</p>
                    <p class="metric-label">{label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_drug_card(drug: dict, show_score: bool = True):
    score = drug.get("score", 0)
    score_pct = int(score * 100)

    confidence_class = "confidence-high" if score >= 0.8 else (
        "confidence-medium" if score >= 0.6 else "confidence-low"
    )
    confidence_label = "High" if score >= 0.8 else ("Medium" if score >= 0.6 else "Low")

    tags_html = ""
    if drug.get("dosage_form"):
        tags_html += f'<span class="drug-tag">{drug["dosage_form"]}</span>'
    if drug.get("route"):
        tags_html += f'<span class="drug-tag">{drug["route"]}</span>'
    if drug.get("strength"):
        tags_html += f'<span class="drug-tag drug-tag-alt">{drug["strength"]}</span>'

    score_html = ""
    if show_score:
        score_html = f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
                <span style="font-size:0.75rem; color:#64748b;">Match confidence</span>
                <span class="{confidence_class}" style="font-size:0.8rem; font-weight:600;">{confidence_label} ({score_pct}%)</span>
            </div>
            <div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>
        """

    st.markdown(
        f"""
        <div class="drug-card">
            <p class="drug-name">{drug.get('brand_name') or 'N/A'}</p>
            <p class="drug-generic">{drug.get('generic_name') or ''} &mdash; {drug.get('labeler') or ''}</p>
            <div class="drug-meta">{tags_html}</div>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_interaction_alert(alert: dict):
    severity = alert.get("severity", "MODERATE").upper()
    css_class = {
        "CRITICAL": "alert-critical",
        "HIGH": "alert-high",
    }.get(severity, "alert-moderate")

    st.markdown(
        f"""
        <div class="{css_class}">
            <p class="alert-title">{severity} -- {alert['drug_a']} + {alert['drug_b']}</p>
            <p class="alert-desc">{alert['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---- Main Tabs ----

def tab_prescription():
    st.markdown('<p class="section-header">Prescription Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.9rem; font-weight:500; margin-bottom:0.5rem;">Enter prescription text</p>',
            unsafe_allow_html=True,
        )
        prescription_text = st.text_area(
            "prescription_input",
            placeholder="e.g.  Tab PCM 500mg TDS x 5 days\n       Cap Amoxicillin 250mg BD x 7 days",
            height=180,
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            use_llm = st.checkbox("Use AI extraction (requires API key)", value=False)
        with c2:
            auto_match = st.checkbox("Auto-match with database", value=True)

        process_btn = st.button("Analyze Prescription", use_container_width=True)

    with col2:
        if process_btn and prescription_text.strip():
            add_audit("PRESCRIPTION_ANALYZE", f"Input: {prescription_text[:80]}...")
            with st.spinner("Analyzing..."):
                extracted = parse_prescription(prescription_text, use_llm=use_llm)

            if not extracted:
                st.warning("No medicines could be extracted from the input text.")
                return

            st.markdown(
                f'<p style="color:#94a3b8; font-size:0.9rem; font-weight:500;">'
                f'Extracted {len(extracted)} medicine(s)</p>',
                unsafe_allow_html=True,
            )

            matched_ids = []

            for item in extracted:
                med_name = item.get("medicine", "")
                details = []
                if item.get("dosage"):
                    details.append(f'<span class="rx-detail">{item["dosage"]}</span>')
                if item.get("frequency"):
                    details.append(f'<span class="rx-detail">{item["frequency"]}</span>')
                if item.get("duration"):
                    details.append(f'<span class="rx-detail">{item["duration"]}</span>')

                method_tag = (
                    '<span class="rx-detail" style="background:rgba(168,85,247,0.1); color:#c4b5fd; border-color:rgba(168,85,247,0.2);">AI Extracted</span>'
                    if item.get("method") == "llm"
                    else '<span class="rx-detail">Regex Extracted</span>'
                )

                match_html = ""
                if auto_match and med_name:
                    results = search_drugs(med_name, limit=1)
                    if results:
                        top = results[0]
                        matched_ids.append(top["id"])
                        score = top.get("score", 0)
                        conf = "HIGH" if score >= 0.8 else ("MED" if score >= 0.6 else "LOW")
                        match_html = (
                            f'<div style="margin-top:0.5rem; padding:0.5rem 0.7rem; '
                            f'background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.2); '
                            f'border-radius:8px;">'
                            f'<span style="color:#86efac; font-size:0.8rem; font-weight:600;">'
                            f'Matched: {top["brand_name"]} ({top["generic_name"]})</span>'
                            f'<span style="float:right; color:#64748b; font-size:0.75rem;">Confidence: {conf}</span>'
                            f'</div>'
                        )

                st.markdown(
                    f"""
                    <div class="rx-card">
                        <p class="rx-medicine">{med_name}</p>
                        <div>{''.join(details)} {method_tag}</div>
                        {match_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interaction check
            if len(matched_ids) >= 2:
                interactions = check_interactions(matched_ids)
                if interactions:
                    st.markdown(
                        '<p class="section-header" style="color:#fca5a5;">Safety Alerts</p>',
                        unsafe_allow_html=True,
                    )
                    for alert in interactions:
                        render_interaction_alert(alert)
                    add_audit(
                        "INTERACTION_DETECTED",
                        f"{len(interactions)} potential interaction(s) found",
                    )
                else:
                    st.markdown(
                        '<div style="background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.2); '
                        'border-radius:8px; padding:0.8rem 1rem; margin-top:1rem;">'
                        '<span style="color:#86efac; font-weight:600;">No known interactions detected among prescribed medicines.</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

        elif process_btn:
            st.warning("Please enter prescription text to analyze.")
        else:
            st.markdown(
                '<div style="display:flex; align-items:center; justify-content:center; height:300px; '
                'color:#475569; font-size:0.95rem;">'
                'Enter a prescription and click Analyze to see results here.'
                '</div>',
                unsafe_allow_html=True,
            )


def tab_search():
    st.markdown('<p class="section-header">Drug Database Search</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input(
            "search_query",
            placeholder="Search by brand name, generic name, or abbreviation (e.g. PCM, Dolo, Aspirin)...",
            label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("Search Database", use_container_width=True)

    if (search_btn or query) and query.strip():
        add_audit("DRUG_SEARCH", f"Query: {query}")
        expanded = expand_abbreviation(query)
        if expanded != query.lower():
            st.markdown(
                f'<p style="color:#94a3b8; font-size:0.85rem;">Abbreviation expanded: '
                f'<strong style="color:#93c5fd;">{query}</strong> &rarr; '
                f'<strong style="color:#a78bfa;">{expanded}</strong></p>',
                unsafe_allow_html=True,
            )

        with st.spinner("Searching..."):
            results = search_drugs(query, limit=15)

        if results:
            st.markdown(
                f'<p style="color:#94a3b8; font-size:0.85rem;">{len(results)} result(s) found</p>',
                unsafe_allow_html=True,
            )
            for drug in results:
                render_drug_card(drug)

                # Expandable alternatives
                with st.expander(f"View alternatives for {drug.get('brand_name', 'this drug')}"):
                    alts = get_alternatives(drug["id"], limit=5)
                    if alts:
                        for alt in alts:
                            st.markdown(
                                f"""
                                <div class="drug-card" style="margin-left:1rem;">
                                    <p class="drug-name" style="font-size:0.9rem;">{alt.get('brand_name', 'N/A')}</p>
                                    <p class="drug-generic">{alt.get('generic_name', '')} | {alt.get('strength', '')} | {alt.get('labeler', '')}</p>
                                    <span class="drug-tag drug-tag-alt">{alt.get('reason', 'Alternative')}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            '<p style="color:#64748b; font-size:0.85rem;">No alternatives found in database.</p>',
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                '<p style="color:#f87171; font-size:0.9rem;">No matching drugs found. Try a different term or check spelling.</p>',
                unsafe_allow_html=True,
            )


def tab_interactions():
    st.markdown('<p class="section-header">Drug Interaction Checker</p>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#94a3b8; font-size:0.9rem;">Add two or more drugs below to check for known interactions.</p>',
        unsafe_allow_html=True,
    )

    # Drug input slots
    drug_inputs = []
    for i in range(4):
        col1, col2 = st.columns([3, 1])
        with col1:
            val = st.text_input(
                f"Drug {i+1}",
                key=f"interaction_drug_{i}",
                placeholder=f"Enter drug name #{i+1}",
                label_visibility="collapsed" if i > 0 else "visible",
            )
            if val:
                drug_inputs.append(val)

    check_btn = st.button("Check Interactions", use_container_width=True)

    if check_btn and len(drug_inputs) >= 2:
        add_audit("INTERACTION_CHECK", f"Drugs: {', '.join(drug_inputs)}")
        with st.spinner("Checking interactions..."):
            # Resolve each drug to an ID
            resolved = []
            for name in drug_inputs:
                results = search_drugs(name, limit=1)
                if results:
                    resolved.append(results[0])

            if len(resolved) >= 2:
                st.markdown(
                    '<p style="color:#94a3b8; font-size:0.85rem; margin-top:1rem;">Resolved drugs:</p>',
                    unsafe_allow_html=True,
                )
                for r in resolved:
                    st.markdown(
                        f'<div class="drug-card" style="padding:0.8rem 1rem;">'
                        f'<span class="drug-name" style="font-size:0.9rem;">{r["brand_name"]}</span> '
                        f'<span class="drug-generic">{r["generic_name"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                drug_ids = [r["id"] for r in resolved]
                alerts = check_interactions(drug_ids)

                if alerts:
                    st.markdown(
                        '<p class="section-header" style="color:#fca5a5; margin-top:1.5rem;">Interactions Found</p>',
                        unsafe_allow_html=True,
                    )
                    for a in alerts:
                        render_interaction_alert(a)
                else:
                    st.markdown(
                        '<div style="background:rgba(34,197,94,0.06); border:1px solid rgba(34,197,94,0.2); '
                        'border-radius:8px; padding:1rem; margin-top:1.5rem; text-align:center;">'
                        '<span style="color:#86efac; font-weight:600; font-size:1rem;">'
                        'No known interactions detected between these drugs.'
                        '</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Could not resolve all drug names. Please check your input.")
    elif check_btn:
        st.warning("Please enter at least 2 drug names to check interactions.")


def tab_batch():
    st.markdown('<p class="section-header">Batch Prescription Processing</p>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#94a3b8; font-size:0.9rem;">'
        'Process multiple prescriptions at once. Enter one prescription per line, separated by blank lines.'
        '</p>',
        unsafe_allow_html=True,
    )

    batch_text = st.text_area(
        "batch_input",
        placeholder=(
            "Prescription 1:\n"
            "Tab PCM 500mg TDS\n"
            "Cap Amoxicillin 250mg BD\n\n"
            "Prescription 2:\n"
            "Tab Atorvastatin 10mg OD\n"
            "Tab Metformin 500mg BD"
        ),
        height=250,
        label_visibility="collapsed",
    )

    batch_btn = st.button("Process Batch", use_container_width=True)

    if batch_btn and batch_text.strip():
        add_audit("BATCH_PROCESS", f"{batch_text.count(chr(10))+1} lines submitted")
        prescriptions = [
            block.strip()
            for block in batch_text.split("\n\n")
            if block.strip()
        ]

        st.markdown(
            f'<p style="color:#94a3b8; font-size:0.85rem;">'
            f'Processing {len(prescriptions)} prescription block(s)...</p>',
            unsafe_allow_html=True,
        )

        progress = st.progress(0)

        for idx, rx in enumerate(prescriptions):
            progress.progress((idx + 1) / len(prescriptions))
            extracted = parse_prescription(rx)

            st.markdown(
                f'<p class="section-header" style="font-size:1rem;">Prescription #{idx+1}</p>',
                unsafe_allow_html=True,
            )

            for item in extracted:
                med_name = item.get("medicine", "")
                results = search_drugs(med_name, limit=1)
                match_info = ""
                if results:
                    top = results[0]
                    match_info = (
                        f'<span style="color:#86efac; font-size:0.8rem;">'
                        f' &rarr; {top["brand_name"]} ({top["generic_name"]})</span>'
                    )

                st.markdown(
                    f"""
                    <div class="rx-card" style="padding:0.8rem 1rem;">
                        <span class="rx-medicine" style="font-size:0.9rem;">{med_name}</span>
                        {match_info}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        progress.empty()


def tab_audit():
    st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#94a3b8; font-size:0.9rem;">'
        'Complete log of all actions performed during this session. '
        'Audit trails are essential for pharmacy regulatory compliance.'
        '</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.audit_log:
        st.markdown(
            '<p style="color:#475569; font-size:0.9rem; text-align:center; margin-top:3rem;">'
            'No actions recorded yet. Use the other tabs to generate activity.</p>',
            unsafe_allow_html=True,
        )
        return

    for entry in st.session_state.audit_log:
        st.markdown(
            f"""
            <div class="audit-entry">
                <span class="audit-time">{entry['time']}</span>
                &nbsp;&bull;&nbsp;
                <strong style="color:#93c5fd;">{entry['action']}</strong>
                &nbsp;&mdash;&nbsp;
                {entry['detail']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Export Audit Log (JSON)"):
        log_json = json.dumps(st.session_state.audit_log, indent=2)
        st.download_button(
            "Download JSON",
            data=log_json,
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ---- Sidebar ----
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:1.5rem 0;">
                <p style="font-size:1.3rem; font-weight:800;
                   background: linear-gradient(135deg, #60a5fa, #a78bfa);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   margin:0;">PharmAssist AI</p>
                <p style="color:#64748b; font-size:0.75rem; margin-top:4px;">v2.0 | Production Build</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            """
            <div style="padding: 0 0.5rem;">
                <p style="color:#94a3b8; font-size:0.8rem; font-weight:600; text-transform:uppercase;
                   letter-spacing:1px; margin-bottom:0.8rem;">System Information</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick stats in sidebar
        try:
            stats = get_drug_stats()
            st.markdown(
                f"""
                <div style="padding:0 0.5rem; color:#cbd5e1; font-size:0.85rem; line-height:2;">
                    Database: <strong>{stats['total_drugs']:,}</strong> drugs<br>
                    Generics: <strong>{stats['unique_generics']:,}</strong><br>
                    Brands: <strong>{stats['unique_brands']:,}</strong><br>
                    Alternatives: <strong>{stats['total_alternatives']:,}</strong><br>
                    Safety Rules: <strong>{stats['interaction_rules']}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                '<p style="color:#f87171; font-size:0.85rem;">Database not loaded.</p>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown(
            """
            <div style="padding: 0 0.5rem;">
                <p style="color:#94a3b8; font-size:0.8rem; font-weight:600; text-transform:uppercase;
                   letter-spacing:1px; margin-bottom:0.8rem;">Capabilities</p>
                <div style="color:#cbd5e1; font-size:0.82rem; line-height:2.2;">
                    Hybrid AI Extraction (Regex + LLM)<br>
                    Fuzzy Drug Search (71K+ drugs)<br>
                    Abbreviation Expansion (Indian brands)<br>
                    Alternative Medicine Finder<br>
                    Drug Interaction Safety Checker<br>
                    Multi-Prescription Batch Processing<br>
                    Regulatory Compliance Audit Trail<br>
                    Confidence Scoring Engine
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            """
            <div style="padding: 0 0.5rem;">
                <p style="color:#94a3b8; font-size:0.8rem; font-weight:600; text-transform:uppercase;
                   letter-spacing:1px; margin-bottom:0.8rem;">Data Source</p>
                <p style="color:#cbd5e1; font-size:0.82rem; line-height:1.8;">
                    U.S. Food & Drug Administration<br>
                    National Drug Code Directory<br>
                    <span style="color:#64748b;">Last updated: 2026</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---- Main ----
def main():
    inject_css()
    init_state()
    render_sidebar()
    render_header()
    render_metrics()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Prescription Analysis",
            "Drug Search",
            "Interaction Checker",
            "Batch Processing",
            "Audit Trail",
        ]
    )

    with tab1:
        tab_prescription()
    with tab2:
        tab_search()
    with tab3:
        tab_interactions()
    with tab4:
        tab_batch()
    with tab5:
        tab_audit()


if __name__ == "__main__":
    main()
