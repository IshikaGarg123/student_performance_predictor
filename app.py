import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg":        "#0f1117",
    "card":      "#1a1d27",
    "card2":     "#20243a",
    "accent":    "#6c63ff",
    "accent2":   "#00d4aa",
    "accent3":   "#ff6b6b",
    "accent4":   "#ffd93d",
    "text":      "#e8eaf6",
    "muted":     "#8892b0",
    "border":    "#2d3154",
    "excellent": "#00d4aa",
    "outstanding":"#6c63ff",
    "good":      "#ffd93d",
    "average":   "#ff9f43",
    "needs":     "#ff6b6b",
}

CAT_COLOR = {
    "Outstanding":        PALETTE["outstanding"],
    "Excellent":          PALETTE["excellent"],
    "Good":               PALETTE["good"],
    "Average":            PALETTE["average"],
    "Needs Improvement":  PALETTE["needs"],
}

FEATURES = [
    'Attendance_%', 'Study_Hours_Per_Day', 'Sleep_Hours',
    'Class_Test_Avg', 'Midterm_Score', 'Assignment_Score',
    'Previous_Exam_Score', 'Homework_Completion_%', 'Participation_Level',
    'Extra_Classes', 'Internet_Access', 'Stress_Level',
    'Health_Status', 'Sports_Participation', 'CoCurricular_Activities'
]

FEATURE_LABELS = {
    'Attendance_%':         'Attendance %',
    'Study_Hours_Per_Day':  'Study Hours/Day',
    'Sleep_Hours':          'Sleep Hours',
    'Class_Test_Avg':       'Class Test Avg',
    'Midterm_Score':        'Midterm Score',
    'Assignment_Score':     'Assignment Score',
    'Previous_Exam_Score':  'Previous Exam Score',
    'Homework_Completion_%':'Homework Completion %',
    'Participation_Level':  'Participation Level',
    'Extra_Classes':        'Extra Classes',
    'Internet_Access':      'Internet Access',
    'Stress_Level':         'Stress Level',
    'Health_Status':        'Health Status',
    'Sports_Participation': 'Sports Participation',
    'CoCurricular_Activities':'Co-Curricular Activities',
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {{
    font-family: 'DM Sans', sans-serif;
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
}}
h1,h2,h3,h4,h5,h6 {{ font-family: 'Syne', sans-serif; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(160deg, #12152b 0%, #1a1d2e 100%);
    border-right: 1px solid {PALETTE['border']};
}}
section[data-testid="stSidebar"] * {{ color: {PALETTE['text']} !important; }}

/* Cards */
.stat-card {{
    background: linear-gradient(135deg, {PALETTE['card']} 0%, {PALETTE['card2']} 100%);
    border: 1px solid {PALETTE['border']};
    border-radius: 16px;
    padding: 22px 24px;
    text-align: center;
    transition: transform .25s, box-shadow .25s;
    height: 140px;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
}}
.stat-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(108,99,255,.25);
}}
.stat-icon  {{ font-size: 2rem; margin-bottom: 4px; }}
.stat-val   {{ font-family:'Syne',sans-serif; font-size:1.75rem; font-weight:800; }}
.stat-label {{ font-size:.78rem; color:{PALETTE['muted']}; margin-top:2px; letter-spacing:.5px; text-transform:uppercase; }}

/* Section headers */
.section-header {{
    display: flex; align-items: center; gap: 12px;
    margin: 32px 0 18px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid {PALETTE['border']};
}}
.section-header .icon {{ font-size:1.6rem; }}
.section-header h2 {{
    font-family:'Syne',sans-serif; font-size:1.4rem;
    font-weight:700; margin:0; color:{PALETTE['text']};
}}

/* Badge */
.badge {{
    display: inline-block;
    padding: 4px 12px; border-radius: 20px;
    font-size: .78rem; font-weight: 600; letter-spacing:.4px;
}}

/* Student card */
.student-card {{
    background: linear-gradient(135deg,{PALETTE['card']} 0%,{PALETTE['card2']} 100%);
    border: 1px solid {PALETTE['border']};
    border-radius: 16px; padding: 24px;
    margin-bottom: 16px;
}}

/* Risk alert */
.risk-high  {{ background:#3d1515; border-left: 4px solid {PALETTE['accent3']}; border-radius:8px; padding:14px 18px; margin:12px 0; }}
.risk-mid   {{ background:#3d3015; border-left: 4px solid {PALETTE['accent4']}; border-radius:8px; padding:14px 18px; margin:12px 0; }}
.risk-low   {{ background:#0d3028; border-left: 4px solid {PALETTE['accent2']}; border-radius:8px; padding:14px 18px; margin:12px 0; }}

/* Table */
.stDataFrame {{ background:{PALETTE['card']}; border-radius:12px; overflow:hidden; }}
thead th {{ background:{PALETTE['accent']} !important; color:white !important; }}

/* Inputs */
.stSlider > div > div > div > div {{ background:{PALETTE['accent']} !important; }}
.stSelectbox > div {{ background:{PALETTE['card']} !important; border-color:{PALETTE['border']} !important; }}
.stTextInput > div > div > input {{ background:{PALETTE['card']} !important; color:{PALETTE['text']} !important; border-color:{PALETTE['border']} !important; }}
div[data-baseweb="select"] > div {{ background:{PALETTE['card']} !important; border-color:{PALETTE['border']} !important; color:{PALETTE['text']} !important; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg,{PALETTE['accent']},{PALETTE['accent2']});
    color:white; border:none; border-radius:10px;
    padding:10px 24px; font-weight:600; font-family:'DM Sans',sans-serif;
    transition: opacity .2s, transform .2s;
}}
.stButton > button:hover {{ opacity:.9; transform:scale(1.02); }}

/* Divider */
hr {{ border-color:{PALETTE['border']}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    orig    = pd.read_csv('student_performance_dataset.csv')
    updated = pd.read_csv('student_performance_dataset_updated.csv')
    # Merge: updated takes priority (has Performance_Category)
    merged = pd.concat([orig, updated], ignore_index=True)
    merged = merged.drop_duplicates(subset='UID', keep='last').reset_index(drop=True)
    if 'Performance_Category' not in merged.columns:
        merged['Performance_Category'] = merged['Final_Score'].apply(score_to_cat)
    return merged

@st.cache_resource
def load_model():
    return joblib.load('student_performance_model.pkl')

def score_to_cat(score):
    if score >= 85:   return "Outstanding"
    elif score >= 75: return "Excellent"
    elif score >= 60: return "Good"
    elif score >= 50: return "Average"
    else:             return "Needs Improvement"

def cat_color(cat):
    return CAT_COLOR.get(cat, PALETTE['muted'])

def cat_badge(cat):
    c = cat_color(cat)
    return f'<span class="badge" style="background:{c}22;color:{c};border:1px solid {c}55">{cat}</span>'

def risk_level(score, attendance, study_hours):
    if score < 50 or attendance < 60 or study_hours < 1:
        return "High Risk", PALETTE['accent3']
    elif score < 65 or attendance < 75 or study_hours < 2:
        return "Medium Risk", PALETTE['accent4']
    else:
        return "Low Risk", PALETTE['accent2']

def predict_score(model, row_dict):
    X = pd.DataFrame([row_dict])[FEATURES]
    pred = model.predict(X)[0]
    return np.clip(pred, 0, 100)

def ai_suggestions(score, attendance, study_hours, stress, health, participation, homework, sleep):
    tips = []
    if attendance < 75:
        tips.append("📅 **Attendance is critically low.** Target ≥85% to stay on track. Missing classes has a direct negative impact on final scores.")
    if study_hours < 2:
        tips.append("📚 **Study hours are insufficient.** Aim for at least 3–4 focused hours/day. Use the Pomodoro technique for better retention.")
    if sleep < 6:
        tips.append("😴 **Sleep deprivation detected.** Students sleeping <6 hrs score 15–20% lower on average. Aim for 7–8 hours nightly.")
    if stress >= 4:
        tips.append("🧘 **High stress level detected.** Practice mindfulness, journaling, or yoga. Consider talking to a counselor.")
    if health <= 2:
        tips.append("🏥 **Health status is poor.** Prioritize medical check-ups and a balanced diet — physical wellness strongly correlates with academic performance.")
    if participation <= 2:
        tips.append("🙋 **Low classroom participation.** Active participation improves retention by up to 60%. Ask questions and join discussions.")
    if homework < 70:
        tips.append("📝 **Homework completion is low.** Set a daily homework schedule. Consistent completion builds exam readiness.")
    if score >= 85:
        tips.append("🌟 **Outstanding performance!** Keep the momentum — explore advanced topics, competitions, or mentorship roles.")
    elif score >= 70:
        tips.append("✅ **Good progress!** Focus on weak subjects and aim for consistent improvement each week.")
    if not tips:
        tips.append("🎯 All indicators look healthy! Keep maintaining this excellent routine.")
    return tips

def make_radar_chart(student_row, title="Student Radar"):
    cats = ['Attendance', 'Study Hours', 'Class Test', 'Midterm', 'Assignment', 'Homework', 'Participation', 'Health']
    # Normalize to 0-100
    vals = [
        student_row.get('Attendance_%', 0),
        min(student_row.get('Study_Hours_Per_Day', 0) * 16.67, 100),
        student_row.get('Class_Test_Avg', 0),
        student_row.get('Midterm_Score', 0),
        student_row.get('Assignment_Score', 0),
        student_row.get('Homework_Completion_%', 0),
        student_row.get('Participation_Level', 1) * 20,
        student_row.get('Health_Status', 1) * 20,
    ]
    vals_plot = vals + [vals[0]]
    cats_plot = cats + [cats[0]]
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor='#1a1d27')
    ax.set_facecolor('#1a1d27')
    ax.plot(angles, vals_plot, color='#6c63ff', linewidth=2.5)
    ax.fill(angles, vals_plot, color='#6c63ff', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color='#e8eaf6', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='#8892b0', fontsize=7)
    ax.grid(color='#2d3154', linewidth=0.7)
    ax.spines['polar'].set_color('#2d3154')
    ax.set_title(title, color='#e8eaf6', fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#1a1d27', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <h2>{title}</h2>
    </div>""", unsafe_allow_html=True)

def plotly_theme():
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=PALETTE['text'], family='DM Sans'),
        xaxis=dict(gridcolor=PALETTE['border'], zeroline=False),
        yaxis=dict(gridcolor=PALETTE['border'], zeroline=False),
        margin=dict(l=40, r=20, t=40, b=40),
    )


# ─────────────────────────────────────────────
#  LOAD DATA & MODEL
# ─────────────────────────────────────────────
df    = load_data()
model = load_model()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 10px">
        <div style="font-size:3rem">🎓</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
                    background:linear-gradient(135deg,{PALETTE['accent']},{PALETTE['accent2']});
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            EduPredict Pro
        </div>
        <div style="font-size:.75rem;color:{PALETTE['muted']};margin-top:4px;">
            Student Performance Intelligence
        </div>
    </div>
    <hr style="border-color:{PALETTE['border']};margin:8px 0 20px">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🔍 Smart Student Search",
            "📊 Student Prediction",
            "📈 Class Analytics",
            "🏆 Leaderboard",
            "⚖️ Student Comparison",
            "🧠 Feature Importance",
            "📤 CSV Bulk Prediction",
            "➕ New Student Prediction",
        ],
        label_visibility="collapsed"
    )

    st.markdown(f"""
    <hr style="border-color:{PALETTE['border']};margin:20px 0 12px">
    <div style="font-size:.72rem;color:{PALETTE['muted']};text-align:center;">
        📦 Dataset: <b style="color:{PALETTE['accent2']}">{len(df)}</b> students &nbsp;|&nbsp;
        Model: <b style="color:{PALETTE['accent']}">Linear Regression</b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE 1 — HOME DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Home Dashboard":
    st.markdown(f"""
    <div style="margin-bottom:28px">
        <h1 style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;margin-bottom:4px;
                   background:linear-gradient(135deg,{PALETTE['accent']},{PALETTE['accent2']});
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🎓 Student Performance Intelligence
        </h1>
        <p style="color:{PALETTE['muted']};font-size:1rem;margin:0;">
            AI-powered insights for every student in your class — at a glance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    total       = len(df)
    avg_score   = df['Final_Score'].mean()
    top_student = df.loc[df['Final_Score'].idxmax()]
    at_risk     = len(df[df['Final_Score'] < 50])
    avg_attend  = df['Attendance_%'].mean()
    outstanding = len(df[df['Performance_Category'].isin(['Outstanding','Excellent'])])

    cards = [
        ("👥", f"{total}", "Total Students",       PALETTE['accent']),
        ("📊", f"{avg_score:.1f}",  "Avg Final Score",    PALETTE['accent2']),
        ("🌟", top_student['Student_Name'].split()[0], "Top Performer", PALETTE['accent4']),
        ("⚠️", f"{at_risk}",        "Students At Risk",   PALETTE['accent3']),
        ("📅", f"{avg_attend:.1f}%","Avg Attendance",      "#a29bfe"),
        ("🏅", f"{outstanding}",    "High Achievers",     "#55efc4"),
    ]

    cols = st.columns(6)
    for col, (icon, val, label, color) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="border-top:3px solid {color};">
                <div class="stat-icon">{icon}</div>
                <div class="stat-val" style="color:{color};">{val}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance distribution donut
    section_header("📊", "Class Performance Overview")
    col1, col2 = st.columns([1.4, 1])

    with col1:
        cat_counts = df['Performance_Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig_pie = px.pie(
            cat_counts, values='Count', names='Category',
            color='Category',
            color_discrete_map=CAT_COLOR,
            hole=0.55,
        )
        fig_pie.update_layout(**plotly_theme(), showlegend=True,
                              legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=12)))
        fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                              marker=dict(line=dict(color='#0f1117', width=2)))
        fig_pie.update_layout(title=dict(text="Performance Category Breakdown",
                                         font=dict(color=PALETTE['text'], size=14)))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        section_header("⚠️", "At-Risk Students")
        at_risk_df = df[df['Final_Score'] < 60].sort_values('Final_Score')[['Student_Name','Final_Score','Attendance_%','Performance_Category']].head(8)
        for _, row in at_risk_df.iterrows():
            rl, rc = risk_level(row['Final_Score'], row['Attendance_%'], 0)
            st.markdown(f"""
            <div style="background:{PALETTE['card']};border:1px solid {rc}44;border-left:3px solid {rc};
                        border-radius:10px;padding:10px 14px;margin-bottom:8px;
                        display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <b style="font-size:.9rem;">{row['Student_Name']}</b><br>
                    <span style="color:{PALETTE['muted']};font-size:.78rem;">Attend: {row['Attendance_%']}%</span>
                </div>
                <div style="text-align:right;">
                    <b style="color:{rc};font-size:1.1rem;">{row['Final_Score']:.1f}</b><br>
                    <span style="font-size:.72rem;color:{rc};">{rl}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Score distribution histogram
    section_header("📉", "Score Distribution")
    fig_hist = px.histogram(df, x='Final_Score', nbins=30,
                            color_discrete_sequence=[PALETTE['accent']],
                            labels={'Final_Score': 'Final Score'})
    fig_hist.update_layout(**plotly_theme())
    fig_hist.update_traces(marker_line_color='rgba(0,0,0,.3)', marker_line_width=1)
    st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE 2 — SMART STUDENT SEARCH
# ─────────────────────────────────────────────
elif page == "🔍 Smart Student Search":
    section_header("🔍", "Smart Student Search")
    st.markdown(f"<p style='color:{PALETTE['muted']};margin-top:-12px;'>Search by name or UID for instant AI-powered insights.</p>", unsafe_allow_html=True)

    search_query = st.text_input("🔎 Type student name or UID...", placeholder="e.g. Navjot Kaur or 24108001")

    if search_query:
        q = search_query.strip().lower()
        results = df[
            df['Student_Name'].str.lower().str.contains(q) |
            df['UID'].astype(str).str.contains(q)
        ]

        if results.empty:
            st.warning("No students found. Try a different name or UID.")
        else:
            st.markdown(f"<p style='color:{PALETTE['accent2']};'>Found <b>{len(results)}</b> result(s):</p>", unsafe_allow_html=True)

            for _, row in results.iterrows():
                row_dict = row[FEATURES].to_dict()
                pred     = predict_score(model, row_dict)
                cat      = score_to_cat(pred)
                rl, rc   = risk_level(pred, row['Attendance_%'], row['Study_Hours_Per_Day'])
                cc       = cat_color(cat)
                tips     = ai_suggestions(pred, row['Attendance_%'], row['Study_Hours_Per_Day'],
                                          row['Stress_Level'], row['Health_Status'],
                                          row['Participation_Level'], row['Homework_Completion_%'],
                                          row['Sleep_Hours'])

                with st.expander(f"👤 {row['Student_Name']}  —  UID: {row['UID']}  |  Score: {pred:.1f}", expanded=len(results)==1):
                    c1, c2, c3 = st.columns([1.2, 1, 1])
                    with c1:
                        st.markdown(f"""
                        <div class="student-card">
                            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;">{row['Student_Name']}</div>
                            <div style="color:{PALETTE['muted']};font-size:.82rem;margin-bottom:14px;">UID: {row['UID']}</div>
                            <div style="font-size:2rem;font-weight:800;color:{cc};">{pred:.1f}<span style="font-size:1rem;color:{PALETTE['muted']};">/100</span></div>
                            <div style="margin:8px 0;">{cat_badge(cat)}</div>
                            <div style="margin-top:10px;padding:6px 12px;border-radius:8px;
                                        background:{rc}22;color:{rc};font-size:.8rem;font-weight:600;">
                                {rl}
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        metrics = {
                            "📅 Attendance":     f"{row['Attendance_%']}%",
                            "📚 Study Hrs/Day":  f"{row['Study_Hours_Per_Day']}",
                            "😴 Sleep Hrs":      f"{row['Sleep_Hours']}",
                            "📝 Class Test Avg": f"{row['Class_Test_Avg']}",
                            "📄 Midterm Score":  f"{row['Midterm_Score']}",
                            "✅ Homework":       f"{row['Homework_Completion_%']}%",
                        }
                        for k, v in metrics.items():
                            st.markdown(f"""
                            <div style="display:flex;justify-content:space-between;
                                        padding:7px 0;border-bottom:1px solid {PALETTE['border']};">
                                <span style="color:{PALETTE['muted']};font-size:.85rem;">{k}</span>
                                <b style="font-size:.9rem;">{v}</b>
                            </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"**🤖 AI Recommendations**")
                        for tip in tips[:4]:
                            st.markdown(f"<div style='font-size:.84rem;margin-bottom:8px;padding:8px 12px;background:{PALETTE['card2']};border-radius:8px;'>{tip}</div>", unsafe_allow_html=True)

                    # Radar chart
                    fig_r = make_radar_chart(row.to_dict(), title=f"{row['Student_Name']} — Profile Radar")
                    buf = io.BytesIO()
                    fig_r.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1d27')
                    buf.seek(0)
                    st.image(buf, width=320)
                    plt.close(fig_r)


# ─────────────────────────────────────────────
#  PAGE 3 — STUDENT PREDICTION
# ─────────────────────────────────────────────
elif page == "📊 Student Prediction":
    section_header("📊", "Student Performance Prediction")
    st.markdown(f"<p style='color:{PALETTE['muted']};margin-top:-12px;'>Select an existing student and adjust factors to see predicted performance.</p>", unsafe_allow_html=True)

    student_names = df['Student_Name'].tolist()
    sel_student   = st.selectbox("Select Student", student_names)
    srow          = df[df['Student_Name'] == sel_student].iloc[0]

    st.markdown(f"<div style='background:{PALETTE['card']};border-radius:12px;padding:16px;margin:8px 0 20px;'>Editing profile for <b style='color:{PALETTE['accent2']};'>{sel_student}</b> (UID: {srow['UID']})</div>", unsafe_allow_html=True)

    # Input sliders in two columns
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        st.markdown("**📚 Academic Factors**")
        inputs['Attendance_%']         = st.slider("Attendance %",           0, 100, int(srow['Attendance_%']))
        inputs['Study_Hours_Per_Day']   = st.slider("Study Hours/Day",        0.0, 12.0, float(srow['Study_Hours_Per_Day']), 0.1)
        inputs['Class_Test_Avg']        = st.slider("Class Test Avg",         0, 100, int(srow['Class_Test_Avg']))
        inputs['Midterm_Score']         = st.slider("Midterm Score",           0, 100, int(srow['Midterm_Score']))
        inputs['Assignment_Score']      = st.slider("Assignment Score",        0, 100, int(srow['Assignment_Score']))
        inputs['Previous_Exam_Score']   = st.slider("Previous Exam Score",    0, 100, int(srow['Previous_Exam_Score']))
        inputs['Homework_Completion_%'] = st.slider("Homework Completion %",  0, 100, int(srow['Homework_Completion_%']))

    with col2:
        st.markdown("**🧬 Behavioral & Lifestyle Factors**")
        inputs['Sleep_Hours']           = st.slider("Sleep Hours",            0.0, 12.0, float(srow['Sleep_Hours']), 0.5)
        inputs['Participation_Level']   = st.slider("Participation Level",    1, 5, int(srow['Participation_Level']))
        inputs['Extra_Classes']         = st.selectbox("Extra Classes",        [0, 1], index=int(srow['Extra_Classes']))
        inputs['Internet_Access']       = st.selectbox("Internet Access",      [0, 1], index=int(srow['Internet_Access']))
        inputs['Stress_Level']          = st.slider("Stress Level",           1, 5, int(srow['Stress_Level']))
        inputs['Health_Status']         = st.slider("Health Status",          1, 5, int(srow['Health_Status']))
        inputs['Sports_Participation']  = st.selectbox("Sports Participation", [0, 1], index=int(srow['Sports_Participation']))
        inputs['CoCurricular_Activities']= st.selectbox("Co-Curricular Activities", [0, 1], index=int(srow['CoCurricular_Activities']))

    if st.button("🚀 Predict Performance"):
        pred   = predict_score(model, inputs)
        cat    = score_to_cat(pred)
        rl, rc = risk_level(pred, inputs['Attendance_%'], inputs['Study_Hours_Per_Day'])
        cc     = cat_color(cat)
        tips   = ai_suggestions(pred, inputs['Attendance_%'], inputs['Study_Hours_Per_Day'],
                                 inputs['Stress_Level'], inputs['Health_Status'],
                                 inputs['Participation_Level'], inputs['Homework_Completion_%'],
                                 inputs['Sleep_Hours'])

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="stat-card" style="border-top:3px solid {cc};height:160px;">
                <div class="stat-icon">🎯</div>
                <div class="stat-val" style="color:{cc};font-size:2.5rem;">{pred:.1f}</div>
                <div class="stat-label">Predicted Final Score</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="stat-card" style="border-top:3px solid {cc};height:160px;">
                <div class="stat-icon">📌</div>
                <div class="stat-val" style="color:{cc};font-size:1.4rem;">{cat}</div>
                <div class="stat-label">Performance Category</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="stat-card" style="border-top:3px solid {rc};height:160px;">
                <div class="stat-icon">⚡</div>
                <div class="stat-val" style="color:{rc};font-size:1.4rem;">{rl}</div>
                <div class="stat-label">Risk Level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI Recommendations")
        for tip in tips:
            st.markdown(f"<div style='padding:12px 16px;background:{PALETTE['card2']};border-radius:10px;margin-bottom:8px;font-size:.9rem;'>{tip}</div>", unsafe_allow_html=True)

        # PDF Report Card
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📄", "Generate PDF Report Card")

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            HAS_RL = True
        except ImportError:
            HAS_RL = False

            if st.button("📥 Generate PDF Report Card"):
    try:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles  = getSampleStyleSheet()
        story   = []
        title_s = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=22, spaceAfter=6, textColor=colors.HexColor('#6c63ff'))
        sub_s   = ParagraphStyle('Sub', parent=styles['Normal'],
                                  fontSize=11, textColor=colors.grey)
        h2_s    = ParagraphStyle('H2', parent=styles['Heading2'],
                                  fontSize=14, spaceAfter=4, textColor=colors.HexColor('#00d4aa'))

        story.append(Paragraph("EduPredict Pro — Report Card", title_s))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", sub_s))
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(f"Student: <b>{sel_student}</b>  |  UID: {srow['UID']}", styles['Normal']))
        story.append(Paragraph(f"Predicted Score: <b>{pred:.1f}/100</b>  |  Category: <b>{cat}</b>  |  Risk: <b>{rl}</b>", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("Performance Metrics", h2_s))
        data = [["Metric", "Value"]] + [[FEATURE_LABELS.get(k, k), str(round(v, 2))] for k, v in inputs.items()]
        tbl = Table(data, colWidths=[8*cm, 6*cm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',     (0, 0), (-1, 0), colors.HexColor('#6c63ff')),
            ('TEXTCOLOR',      (0, 0), (-1, 0), colors.white),
            ('FONTNAME',       (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
            ('TEXTCOLOR',      (0, 1), (-1, -1), colors.black),
            ('GRID',           (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5*cm))

        # Radar chart
        fig_r   = make_radar_chart(inputs, title=f"{sel_student} — Profile Radar")
        img_buf = io.BytesIO()
        fig_r.savefig(img_buf, format='png', dpi=120,
                      bbox_inches='tight', facecolor='#1a1d27')
        img_buf.seek(0)
        plt.close(fig_r)
        story.append(Paragraph("Performance Radar Chart", h2_s))
        story.append(RLImage(img_buf, width=10*cm, height=10*cm))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("AI Recommendations", h2_s))
        for tip in tips:
            clean = tip.replace('**', '').replace('*', '')
            story.append(Paragraph(f"• {clean}", styles['Normal']))
            story.append(Spacer(1, 0.15*cm))

        doc.build(story)
        pdf_buf.seek(0)

        st.download_button(
            label="📥 Download Report Card PDF",
            data=pdf_buf,
            file_name=f"{sel_student.replace(' ', '_')}_report.pdf",
            mime="application/pdf"
        )
        st.success("✅ PDF ready! Click the button above to download.")

    except Exception as e:
        st.error(f"PDF generation failed: {e}")
# ─────────────────────────────────────────────
#  PAGE 4 — CLASS ANALYTICS
# ─────────────────────────────────────────────
elif page == "📈 Class Analytics":
    section_header("📈", "Class Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Distribution", "📅 Attendance Analysis", "📚 Study Hours", "📉 Trends"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Final_Score', nbins=25, color_discrete_sequence=[PALETTE['accent']],
                               title='Final Score Distribution')
            fig.update_layout(**plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cat_df = df['Performance_Category'].value_counts().reset_index()
            cat_df.columns = ['Category', 'Count']
            fig2 = px.bar(cat_df, x='Category', y='Count',
                          color='Category', color_discrete_map=CAT_COLOR,
                          title='Students per Category')
            fig2.update_layout(**plotly_theme(), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Box plot by category
        fig3 = px.box(df, x='Performance_Category', y='Final_Score',
                      color='Performance_Category', color_discrete_map=CAT_COLOR,
                      title='Score Range per Category')
        fig3.update_layout(**plotly_theme(), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x='Attendance_%', y='Final_Score',
                             color='Performance_Category', color_discrete_map=CAT_COLOR,
                             hover_data=['Student_Name'],
                             title='Attendance % vs Final Score',
                             trendline='ols')
            fig.update_layout(**plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            attend_bins = pd.cut(df['Attendance_%'], bins=[0,60,75,85,100],
                                  labels=['<60%','60-75%','75-85%','>85%'])
            attend_group = df.groupby(attend_bins, observed=True)['Final_Score'].mean().reset_index()
            attend_group.columns = ['Attendance Range', 'Avg Score']
            fig2 = px.bar(attend_group, x='Attendance Range', y='Avg Score',
                          color='Avg Score', color_continuous_scale=['#ff6b6b','#ffd93d','#00d4aa'],
                          title='Avg Score by Attendance Range')
            fig2.update_layout(**plotly_theme())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x='Study_Hours_Per_Day', y='Final_Score',
                             color='Performance_Category', color_discrete_map=CAT_COLOR,
                             hover_data=['Student_Name'],
                             title='Study Hours vs Final Score',
                             trendline='ols')
            fig.update_layout(**plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(df, x='Sleep_Hours', y='Final_Score',
                              color='Performance_Category', color_discrete_map=CAT_COLOR,
                              hover_data=['Student_Name'],
                              title='Sleep Hours vs Final Score',
                              trendline='ols')
            fig2.update_layout(**plotly_theme())
            st.plotly_chart(fig2, use_container_width=True)

        # Stress Level analysis
        stress_group = df.groupby('Stress_Level')['Final_Score'].mean().reset_index()
        fig3 = px.bar(stress_group, x='Stress_Level', y='Final_Score',
                      color='Final_Score', color_continuous_scale=['#00d4aa','#ffd93d','#ff6b6b'],
                      title='Avg Score by Stress Level (1=Low, 5=High)')
        fig3.update_layout(**plotly_theme())
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        # Performance trend based on Previous vs Final score
        df_trend = df.copy()
        df_trend['Score_Delta'] = df_trend['Final_Score'] - df_trend['Previous_Exam_Score']
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df_trend, x='Previous_Exam_Score', y='Final_Score',
                             color='Score_Delta',
                             color_continuous_scale=['#ff6b6b','#ffd93d','#00d4aa'],
                             hover_data=['Student_Name'],
                             title='Previous Score vs Current Final Score')
            fig.add_shape(type='line', x0=0, y0=0, x1=100, y1=100,
                          line=dict(color='white', dash='dash', width=1))
            fig.update_layout(**plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df_trend, x='Score_Delta', nbins=25,
                                color_discrete_sequence=[PALETTE['accent2']],
                                title='Score Change: Previous → Final')
            fig2.update_layout(**plotly_theme())
            st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        section_header("🔗", "Feature Correlation with Final Score")
        corr_cols = ['Attendance_%','Study_Hours_Per_Day','Sleep_Hours','Class_Test_Avg',
                     'Midterm_Score','Assignment_Score','Homework_Completion_%',
                     'Stress_Level','Health_Status','Final_Score']
        corr_df = df[corr_cols].corr()
        fig_hm = px.imshow(corr_df, text_auto='.2f', aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Correlation Heatmap')
        fig_hm.update_layout(**plotly_theme(), height=500)
        st.plotly_chart(fig_hm, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE 5 — LEADERBOARD
# ─────────────────────────────────────────────
elif page == "🏆 Leaderboard":
    section_header("🏆", "Class Leaderboard")

    top_n = st.slider("Show Top N Students", 5, 50, 20)
    sort_col = st.selectbox("Sort By", ['Final_Score', 'Attendance_%', 'Study_Hours_Per_Day', 'Class_Test_Avg'])

    top_df = df.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)
    top_df.index += 1

    for rank, row in top_df.iterrows():
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        cc    = cat_color(row['Performance_Category'])
        st.markdown(f"""
        <div style="background:{PALETTE['card']};border:1px solid {PALETTE['border']};
                    border-radius:12px;padding:14px 20px;margin-bottom:8px;
                    display:flex;justify-content:space-between;align-items:center;">
            <div style="display:flex;align-items:center;gap:16px;">
                <span style="font-size:1.4rem;width:36px;text-align:center;">{medal}</span>
                <div>
                    <b style="font-size:1rem;">{row['Student_Name']}</b>
                    <span style="color:{PALETTE['muted']};font-size:.8rem;margin-left:8px;">UID: {row['UID']}</span><br>
                    <span style="font-size:.78rem;color:{PALETTE['muted']};">
                        Attend: {row['Attendance_%']}% &nbsp;|&nbsp; Study: {row['Study_Hours_Per_Day']}h/day
                    </span>
                </div>
            </div>
            <div style="text-align:right;">
                <b style="color:{cc};font-size:1.4rem;">{row['Final_Score']:.1f}</b>
                <div style="margin-top:4px;">{cat_badge(row['Performance_Category'])}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig_lb = px.bar(top_df, x='Student_Name', y='Final_Score',
                    color='Performance_Category', color_discrete_map=CAT_COLOR,
                    title=f'Top {top_n} Students by {FEATURE_LABELS.get(sort_col, sort_col)}')
    fig_lb.update_layout(**plotly_theme(), xaxis_tickangle=-35)
    st.plotly_chart(fig_lb, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE 6 — STUDENT COMPARISON
# ─────────────────────────────────────────────
elif page == "⚖️ Student Comparison":
    section_header("⚖️", "Student Comparison Tool")
    st.markdown(f"<p style='color:{PALETTE['muted']};margin-top:-12px;'>Compare two students side-by-side with radar charts and metrics.</p>", unsafe_allow_html=True)

    names   = df['Student_Name'].tolist()
    col1, col2 = st.columns(2)
    with col1:
        s1 = st.selectbox("Student 1", names, index=0, key='s1')
    with col2:
        s2 = st.selectbox("Student 2", names, index=1, key='s2')

    r1 = df[df['Student_Name'] == s1].iloc[0]
    r2 = df[df['Student_Name'] == s2].iloc[0]

    p1 = predict_score(model, r1[FEATURES].to_dict())
    p2 = predict_score(model, r2[FEATURES].to_dict())

    if st.button("⚖️ Compare Students"):
        # Side-by-side cards
        c1, c2 = st.columns(2)
        for col, student, pred in [(c1, r1, p1), (c2, r2, p2)]:
            cat = score_to_cat(pred)
            cc  = cat_color(cat)
            with col:
                st.markdown(f"""
                <div class="student-card" style="border-top:3px solid {cc};">
                    <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;">{student['Student_Name']}</div>
                    <div style="color:{PALETTE['muted']};font-size:.82rem;">UID: {student['UID']}</div>
                    <div style="font-size:2.5rem;font-weight:800;color:{cc};margin:10px 0;">{pred:.1f}</div>
                    <div>{cat_badge(cat)}</div>
                </div>""", unsafe_allow_html=True)

        # Metric comparison table
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📋", "Metric Comparison")
        compare_metrics = ['Attendance_%','Study_Hours_Per_Day','Sleep_Hours',
                           'Class_Test_Avg','Midterm_Score','Assignment_Score',
                           'Homework_Completion_%','Participation_Level','Stress_Level','Health_Status']
        for m in compare_metrics:
            v1, v2 = r1[m], r2[m]
            lbl    = FEATURE_LABELS.get(m, m)
            better = PALETTE['accent2'] if v1 >= v2 else PALETTE['muted']
            worse  = PALETTE['accent2'] if v2 > v1  else PALETTE['muted']
            if m == 'Stress_Level':  # Lower is better for stress
                better = PALETTE['accent2'] if v1 <= v2 else PALETTE['muted']
                worse  = PALETTE['accent2'] if v2 < v1  else PALETTE['muted']
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                        padding:9px 0;border-bottom:1px solid {PALETTE['border']};
                        align-items:center;font-size:.88rem;">
                <b style="color:{better};">{v1}</b>
                <span style="text-align:center;color:{PALETTE['muted']};">{lbl}</span>
                <b style="color:{worse};text-align:right;">{v2}</b>
            </div>""", unsafe_allow_html=True)

        # Dual radar charts
        st.markdown("<br>", unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        for col, row, name in [(rc1, r1, s1), (rc2, r2, s2)]:
            with col:
                fig_r = make_radar_chart(row.to_dict(), title=f"{name}")
                buf   = io.BytesIO()
                fig_r.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1d27')
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close(fig_r)

        # Plotly grouped bar comparison
        comp_data = pd.DataFrame({
            'Metric': [FEATURE_LABELS.get(m, m) for m in compare_metrics],
            s1:       [r1[m] for m in compare_metrics],
            s2:       [r2[m] for m in compare_metrics],
        })
        fig_comp = go.Figure(data=[
            go.Bar(name=s1, x=comp_data['Metric'], y=comp_data[s1], marker_color=PALETTE['accent']),
            go.Bar(name=s2, x=comp_data['Metric'], y=comp_data[s2], marker_color=PALETTE['accent2']),
        ])
        fig_comp.update_layout(**plotly_theme(), barmode='group',
                               title='Side-by-Side Metric Comparison', xaxis_tickangle=-30)
        st.plotly_chart(fig_comp, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE 7 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
elif page == "🧠 Feature Importance":
    section_header("🧠", "Feature Importance Analysis")
    st.markdown(f"<p style='color:{PALETTE['muted']};margin-top:-12px;'>Which factors matter most in predicting student performance?</p>", unsafe_allow_html=True)

    coefs   = model.coef_
    feat_df = pd.DataFrame({
        'Feature':    [FEATURE_LABELS.get(f, f) for f in FEATURES],
        'Importance': np.abs(coefs),
        'Coefficient':coefs,
        'Direction':  ['Positive 📈' if c > 0 else 'Negative 📉' for c in coefs],
    }).sort_values('Importance', ascending=False)

    colors_bar = [PALETTE['accent2'] if c > 0 else PALETTE['accent3'] for c in feat_df['Coefficient']]

    fig = go.Figure(go.Bar(
        x=feat_df['Importance'],
        y=feat_df['Feature'],
        orientation='h',
        marker_color=colors_bar,
        text=[f"{v:.3f}" for v in feat_df['Coefficient']],
        textposition='outside',
    ))
    theme = plotly_theme()
    theme['yaxis'] = dict(autorange='reversed', gridcolor=PALETTE['border'])
    fig.update_layout(**theme, height=550,
                  title="Feature Importance (Absolute Coefficient Value)",
                  xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### 📋 Detailed Feature Coefficients")
    display_df = feat_df[['Feature','Coefficient','Direction']].reset_index(drop=True)
    display_df['Coefficient'] = display_df['Coefficient'].round(4)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Positive vs Negative
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pos = feat_df[feat_df['Coefficient'] > 0]
        fig_pos = px.bar(pos, x='Coefficient', y='Feature', orientation='h',
                         color_discrete_sequence=[PALETTE['accent2']],
                         title='✅ Positive Factors (Help Score)')
        theme_pos = plotly_theme()
        theme_pos['yaxis'] = dict(autorange='reversed')
        fig_pos.update_layout(**theme_pos)
        st.plotly_chart(fig_pos, use_container_width=True)
    with col2:
        neg = feat_df[feat_df['Coefficient'] < 0]
        fig_neg = px.bar(neg, x='Coefficient', y='Feature', orientation='h',
                         color_discrete_sequence=[PALETTE['accent3']],
                         title='❌ Negative Factors (Hurt Score)')
        theme_neg = plotly_theme()
        theme_neg['yaxis'] = dict(autorange='reversed')
        fig_neg.update_layout(**theme_neg)
        st.plotly_chart(fig_neg, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE 8 — CSV BULK PREDICTION
# ─────────────────────────────────────────────
elif page == "📤 CSV Bulk Prediction":
    section_header("📤", "CSV Bulk Prediction")
    st.markdown(f"<p style='color:{PALETTE['muted']};margin-top:-12px;'>Upload a CSV with student data and get predictions for all students at once.</p>", unsafe_allow_html=True)

    # Show required columns
    st.markdown(f"""
    <div style="background:{PALETTE['card']};border:1px solid {PALETTE['border']};border-radius:12px;padding:16px 20px;margin-bottom:20px;">
        <b>📋 Required columns in your CSV:</b><br>
        <code style="color:{PALETTE['accent2']};font-size:.85rem;">{', '.join(FEATURES)}</code><br>
        <span style="color:{PALETTE['muted']};font-size:.82rem;">Optional: UID, Student_Name</span>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV File", type=['csv'])

    if uploaded:
        try:
            upload_df = pd.read_csv(uploaded)
            st.markdown(f"✅ Loaded **{len(upload_df)}** rows, **{len(upload_df.columns)}** columns.")

            missing = [f for f in FEATURES if f not in upload_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                preds = model.predict(upload_df[FEATURES])
                upload_df['Predicted_Score']        = np.clip(preds, 0, 100).round(2)
                upload_df['Performance_Category']   = upload_df['Predicted_Score'].apply(score_to_cat)
                upload_df['Risk_Level']             = upload_df.apply(
                    lambda r: risk_level(r['Predicted_Score'],
                                         r.get('Attendance_%', 75),
                                         r.get('Study_Hours_Per_Day', 3))[0], axis=1)

                st.markdown("### 📊 Prediction Results")
                st.dataframe(upload_df[['Student_Name','Predicted_Score','Performance_Category','Risk_Level']
                                       if 'Student_Name' in upload_df.columns
                                       else ['Predicted_Score','Performance_Category','Risk_Level']],
                             use_container_width=True)

                # Category breakdown
                cat_cnt = upload_df['Performance_Category'].value_counts().reset_index()
                cat_cnt.columns = ['Category','Count']
                fig = px.pie(cat_cnt, values='Count', names='Category',
                             color='Category', color_discrete_map=CAT_COLOR, hole=.5,
                             title='Predicted Category Distribution')
                fig.update_layout(**plotly_theme())
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv_out = upload_df.to_csv(index=False).encode()
                b64 = base64.b64encode(csv_out).decode()
                st.markdown(
                    f'<a href="data:file/csv;base64,{b64}" download="bulk_predictions.csv" '
                    f'style="background:linear-gradient(135deg,#6c63ff,#00d4aa);color:white;'
                    f'padding:10px 24px;border-radius:10px;text-decoration:none;font-weight:600;">'
                    f'📥 Download Predictions CSV</a>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ─────────────────────────────────────────────
#  PAGE 9 — NEW STUDENT PREDICTION
# ─────────────────────────────────────────────
elif page == "➕ New Student Prediction":
    section_header("➕", "New Student Prediction")
    st.markdown(f"<p style='color:{PALETTE['muted']};margin-top:-12px;'>Enter details for a student not in the dataset and predict their performance.</p>", unsafe_allow_html=True)

    with st.form("new_student_form"):
        col1, col2 = st.columns(2)
        with col1:
            name  = st.text_input("Student Name", placeholder="e.g. Arjun Sharma")
            uid   = st.text_input("UID (optional)", placeholder="e.g. 24109001")
            st.markdown("**📚 Academic Factors**")
            attend  = st.slider("Attendance %",           0, 100, 80)
            study   = st.slider("Study Hours/Day",        0.0, 12.0, 3.0, 0.1)
            ct_avg  = st.slider("Class Test Avg",         0, 100, 70)
            midterm = st.slider("Midterm Score",           0, 100, 65)
            assign  = st.slider("Assignment Score",        0, 100, 75)
            prev    = st.slider("Previous Exam Score",    0, 100, 70)
            hw      = st.slider("Homework Completion %",  0, 100, 80)

        with col2:
            st.markdown("**🧬 Behavioral & Lifestyle**")
            sleep   = st.slider("Sleep Hours",            0.0, 12.0, 7.0, 0.5)
            part    = st.slider("Participation Level",    1, 5, 3)
            extra   = st.selectbox("Extra Classes",        [0, 1], format_func=lambda x: "Yes" if x else "No")
            net     = st.selectbox("Internet Access",      [0, 1], format_func=lambda x: "Yes" if x else "No")
            stress  = st.slider("Stress Level",           1, 5, 3)
            health  = st.slider("Health Status",          1, 5, 3)
            sports  = st.selectbox("Sports Participation", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cocurr  = st.selectbox("Co-Curricular Activities", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🚀 Predict Performance")

    if submitted:
        inputs_new = {
            'Attendance_%':          attend,
            'Study_Hours_Per_Day':    study,
            'Sleep_Hours':            sleep,
            'Class_Test_Avg':         ct_avg,
            'Midterm_Score':          midterm,
            'Assignment_Score':       assign,
            'Previous_Exam_Score':    prev,
            'Homework_Completion_%':  hw,
            'Participation_Level':    part,
            'Extra_Classes':          extra,
            'Internet_Access':        net,
            'Stress_Level':           stress,
            'Health_Status':          health,
            'Sports_Participation':   sports,
            'CoCurricular_Activities':cocurr,
        }
        pred   = predict_score(model, inputs_new)
        cat    = score_to_cat(pred)
        rl, rc = risk_level(pred, attend, study)
        cc     = cat_color(cat)
        tips   = ai_suggestions(pred, attend, study, stress, health, part, hw, sleep)

        sname = name if name else "New Student"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{PALETTE['card']},{PALETTE['card2']});
                    border:1px solid {cc}55;border-radius:16px;padding:28px;margin-top:20px;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;">{sname}</div>
            <div style="font-size:3.5rem;font-weight:800;color:{cc};margin:14px 0;">{pred:.1f}</div>
            <div style="font-size:1rem;color:{PALETTE['muted']};margin-bottom:10px;">Predicted Final Score / 100</div>
            <div style="display:flex;gap:12px;justify-content:center;">
                {cat_badge(cat)}
                <span class="badge" style="background:{rc}22;color:{rc};border:1px solid {rc}55;">{rl}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5])
        with col1:
            fig_r = make_radar_chart(inputs_new, title=f"{sname} Profile")
            buf   = io.BytesIO()
            fig_r.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#1a1d27')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig_r)
        with col2:
            st.markdown("### 🤖 AI Recommendations")
            for tip in tips:
                st.markdown(f"<div style='padding:10px 14px;background:{PALETTE['card2']};border-radius:9px;margin-bottom:8px;font-size:.88rem;'>{tip}</div>", unsafe_allow_html=True)