# app_sociogram.py

import io

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from fpdf import FPDF
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

import audit
import auth

# ─── Auth & session ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sociogram Generator", layout="wide")
current_user = auth.require_auth()
audit.session_start()

# ─── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* NSW brand palette */
:root {
    --nsw-navy:   #002664;
    --nsw-blue:   #0066CC;
    --nsw-light:  #F5F7FA;
    --nsw-border: #D7DCE0;
}

/* Page background */
.stApp { background-color: var(--nsw-light); }

/* Main headings */
h1, h2, h3 { color: var(--nsw-navy) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #fff;
    border-right: 1px solid var(--nsw-border);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid var(--nsw-border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button {
    background-color: var(--nsw-blue);
    color: #fff;
    border: none;
    border-radius: 4px;
}
.stButton > button:hover {
    background-color: var(--nsw-navy);
    color: #fff;
}

/* Upload box */
div[data-testid="stFileUploader"] {
    background: #fff;
    border: 1px solid var(--nsw-border);
    border-radius: 8px;
    padding: 0.5rem;
}

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Tab active indicator */
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid var(--nsw-blue) !important;
    color: var(--nsw-navy) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Account zone ──────────────────────────────────────────────────────────
    initial = (current_user["display_name"] or "?")[0].upper()
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:0.5rem 0 0.25rem;">
        <div style="
            width:36px;height:36px;border-radius:50%;
            background:var(--nsw-blue);color:#fff;
            display:flex;align-items:center;justify-content:center;
            font-weight:700;font-size:1rem;flex-shrink:0;">
            {initial}
        </div>
        <div>
            <div style="font-weight:600;font-size:0.9rem;color:#002664;">
                {current_user["display_name"]}
            </div>
            <div style="font-size:0.75rem;color:#666;">{current_user["email"]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        auth.logout()

    st.markdown("---")

    # ── Graph settings zone ───────────────────────────────────────────────────
    st.markdown("**Graph Settings**")
    selected_categories = st.multiselect(
        "Nomination types",
        options=["Inclusive", "Helpful", "Collaborator"],
        default=["Inclusive", "Helpful", "Collaborator"],
    )
    cluster_coloring = st.checkbox(
        "Colour nodes by social group",
        value=False,
        help="Uses Louvain community detection to colour clusters instead of popularity gradient.",
    )
    show_mutual = st.checkbox(
        "Highlight mutual nominations",
        value=False,
        help="Draws a thick purple edge between any two students who nominated each other.",
    )

# ─── Page header ───────────────────────────────────────────────────────────────
st.title("Sociogram Generator")
st.info(
    "This tool visualises peer nomination data from your class survey as an interactive "
    "social network (sociogram). "
    "[Get the Google Form template](https://docs.google.com/forms/d/16ARyYjgnF0SN-5VO3ZNriftCPjHhI94ylKUk7t8jiFk/copy)",
    icon="ℹ️",
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# ─── Privacy Collection Notice ─────────────────────────────────────────────────
if not st.session_state.get("privacy_acknowledged"):
    st.markdown("""
    <div style="
        border-left: 4px solid #002664;
        background: #fff;
        border-radius: 0 8px 8px 0;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="font-size:1.1rem;font-weight:700;color:#002664;margin-bottom:0.5rem;">
            🔒 Privacy Collection Notice
        </div>
        <p style="margin:0 0 0.75rem;">
            This tool collects student names and peer nomination responses submitted via your
            class survey. Information is collected to generate a sociogram to help you understand
            social connections within your class.
        </p>
        <p style="margin:0 0 0.5rem;"><strong>How your data is handled:</strong></p>
        <ul style="margin:0 0 0.75rem;padding-left:1.25rem;">
            <li>Processed in your browser session only — <strong>not stored</strong> on any server after your session ends.</li>
            <li><strong>Not shared</strong> with any third party.</li>
            <li>You are responsible for ensuring students have been informed their responses will be used for this purpose.</li>
        </ul>
        <p style="margin:0;font-size:0.85rem;color:#555;">
            Operated in accordance with the
            <a href="https://legislation.nsw.gov.au/view/html/inforce/current/act-1998-133"
               target="_blank">NSW Privacy and Personal Information Protection Act 1998</a>
            and the NSW Department of Education Privacy Code of Practice.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("I understand — continue to the tool", type="primary"):
        st.session_state["privacy_acknowledged"] = True
        st.rerun()
    st.stop()

# ─── Upload section ────────────────────────────────────────────────────────────
st.subheader("Load data")
col_upload, col_mid, col_sheets = st.columns([4, 1, 4])

with col_upload:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], label_visibility="collapsed")

with col_mid:
    st.markdown(
        "<div style='text-align:center;color:#999;padding-top:1.5rem;font-size:0.9rem;'>or</div>",
        unsafe_allow_html=True,
    )

with col_sheets:
    sheets_url = st.text_input(
        "Paste Google Sheets URL",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        help="The sheet must be shared as 'Anyone with the link can view'.",
        label_visibility="collapsed",
    )
    if sheets_url:
        if st.button("Load from Google Sheets", use_container_width=True, type="primary"):
            st.session_state["sheets_url"] = sheets_url
            st.session_state.pop("sample_data", None)
            st.rerun()

if st.button("Load example data", key="load_example"):
    audit.sample_data_loaded()
    st.session_state["sample_data"] = {
        "Timestamp": ["2025-04-01"] * 30,
        "Your name": [
            "Alice", "Ben", "Charlotte", "Daniel", "Emma", "Finn", "Grace", "Harry",
            "Isla", "Jack", "Kayla", "Liam", "Mia", "Noah", "Olivia", "Patrick",
            "Quinn", "Ryan", "Sophie", "Thomas", "Uma", "Victor", "Willow", "Connor",
            "Yasmin", "Zoe", "Aiden", "Bella", "Xavier", "Daisy",
        ],
        "Inclusive - Choice 1": [
            "Emma", "Emma", "Emma", "Grace", "Alice", "Charlotte", "Emma", "Emma",
            "Liam", "Liam", "Liam", "Isla", "Liam", "Olivia", "Liam", "Liam",
            "Sophie", "Sophie", "Quinn", "Sophie", "Sophie", "Sophie", "Sophie", "Sophie",
            "Emma", "Alice", "Liam", "Emma", "Sophie", "Aiden",
        ],
        "Inclusive - Choice 2": [
            "Ben", "Yasmin", "Daniel", "Emma", "Bella", "Emma", "Harry", "Finn",
            "Yasmin", "Kayla", "Jack", "Mia", "Noah", "Liam", "Jack", "Mia",
            "Ryan", "Thomas", "Uma", "Ryan", "Willow", "Connor", "Victor", "Uma",
            "Liam", "Sophie", "Sophie", "Sophie", "Liam", "",
        ],
        "Helpful - Choice 1": [
            "Emma", "Alice", "Emma", "Ben", "Finn", "Emma", "Charlotte", "Grace",
            "Liam", "Olivia", "Liam", "Jack", "Kayla", "Liam", "Noah", "Kayla",
            "Sophie", "Uma", "Ryan", "Uma", "Thomas", "Sophie", "Sophie", "Ryan",
            "Alice", "Bella", "Noah", "Alice", "Quinn", "Sophie",
        ],
        "Helpful - Choice 2": [
            "Yasmin", "Charlotte", "Harry", "Alice", "Grace", "Harry", "Emma", "Ben",
            "Aiden", "Liam", "Mia", "Olivia", "Jack", "Kayla", "Patrick", "Noah",
            "Connor", "Sophie", "Thomas", "Quinn", "Connor", "Uma", "Aiden", "Victor",
            "Yasmin", "Grace", "Aiden", "Finn", "Thomas", "",
        ],
        "Collaborator - Choice 1": [
            "Emma", "Emma", "Ben", "Charlotte", "Daniel", "Ben", "Harry", "Grace",
            "Liam", "Liam", "Olivia", "Kayla", "Patrick", "Mia", "Liam", "Jack",
            "Sophie", "Quinn", "Thomas", "Sophie", "Victor", "Uma", "Connor", "Sophie",
            "Ben", "Emma", "Patrick", "Zoe", "Victor", "Liam",
        ],
        "Collaborator - Choice 2": [
            "Zoe", "Harry", "Alice", "Emma", "Bella", "Alice", "Ben", "Emma",
            "Patrick", "Isla", "Jack", "Noah", "Liam", "Jack", "Mia", "Aiden",
            "Willow", "Victor", "Uma", "Willow", "Sophie", "Connor", "Ryan", "Willow",
            "Isla", "Charlotte", "Mia", "Alice", "Sophie", "",
        ],
    }
    st.session_state.pop("sheets_url", None)
    st.rerun()

# ─── Nomination categories ─────────────────────────────────────────────────────
categories = {
    "Inclusive": "green",
    "Helpful": "blue",
    "Collaborator": "red",
}

# ─── Input validation ──────────────────────────────────────────────────────────

def _validate_csv(dataframe: pd.DataFrame) -> list[str]:
    errors = []
    if dataframe.empty:
        errors.append("The uploaded file contains no rows.")
        return errors
    if len(dataframe.columns) < 2:
        errors.append(
            f"Expected at least 2 columns (Timestamp, Your name) but found {len(dataframe.columns)}."
        )
        return errors

    name_column = dataframe.columns[1]
    if dataframe[name_column].dropna().eq("").all():
        errors.append(f'Column "{name_column}" (student names) appears to be empty.')

    nomination_cols = [c for c in dataframe.columns if any(cat in c for cat in categories)]
    if not nomination_cols:
        errors.append(
            'No nomination columns found. Expected columns containing '
            '"Inclusive", "Helpful", or "Collaborator" (e.g. "Inclusive - Choice 1").'
        )
    return errors


# ─── Google Sheets URL helper ─────────────────────────────────────────────────

def _sheets_url_to_csv_bytes(url: str) -> bytes:
    """
    Convert a Google Sheets share URL to CSV bytes by using the /export endpoint.

    Accepts URLs in the forms:
      https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=GID
      https://docs.google.com/spreadsheets/d/SHEET_ID/edit?usp=sharing
      https://docs.google.com/spreadsheets/d/SHEET_ID/
    """
    import re
    import urllib.request

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Could not find a spreadsheet ID in the URL. Please check the link.")

    sheet_id = match.group(1)

    # Preserve gid (sheet tab) if present
    gid_match = re.search(r"[#&?]gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"

    export_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )

    try:
        with urllib.request.urlopen(export_url, timeout=15) as resp:  # noqa: S310
            return resp.read()
    except Exception as exc:
        raise ValueError(
            f"Could not fetch the sheet ({exc}). "
            "Make sure the sheet is shared as 'Anyone with the link can view'."
        ) from exc


# ─── Cached data pipeline ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_and_build(file_bytes: bytes) -> list[tuple]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    name_candidates = [c for c in df.columns if "name" in c.lower()]
    name_col = name_candidates[0] if name_candidates else df.columns[1]
    df[name_col] = df[name_col].astype(str).str.strip().str.title()
    for col in df.columns:
        if any(cat in col for cat in categories):
            df[col] = df[col].astype(str).str.strip().str.title()
    df.replace("Nan", pd.NA, inplace=True)
    edges = []
    for _, row in df.iterrows():
        source = str(row[name_col]).strip()
        for cat in categories:
            for i in (1, 2):
                col_name = f"{cat} - Choice {i}"
                if col_name in df.columns:
                    target = row[col_name]
                    if pd.notna(target) and str(target).strip():
                        edges.append((source, str(target).strip(), cat))
    return edges


@st.cache_data(show_spinner=False)
def _spring_layout(edge_tuples: tuple) -> dict:
    G = nx.DiGraph()
    for u, v, cat in edge_tuples:
        G.add_edge(u, v, category=cat)
    return nx.spring_layout(G, seed=42)


@st.cache_data(show_spinner=False)
def _louvain_partition(filtered_edge_tuples: tuple) -> dict:
    G_f = nx.DiGraph()
    for u, v, cat in filtered_edge_tuples:
        G_f.add_edge(u, v, category=cat)
    return community_louvain.best_partition(G_f.to_undirected())


# ─── Load data ─────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error(
            f"File is too large ({uploaded_file.size / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )
        st.stop()
    audit.file_uploaded(uploaded_file.size)
    file_bytes = uploaded_file.getvalue()
    try:
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()
    validation_errors = _validate_csv(raw_df)
    if validation_errors:
        st.error("**The uploaded file has the following issues:**")
        for err in validation_errors:
            st.markdown(f"- {err}")
        st.markdown(
            "Please check that you exported from the correct Google Form response sheet "
            "and that the column headers haven't been renamed."
        )
        st.stop()
    try:
        edges = _load_and_build(file_bytes)
    except Exception as e:
        st.error(
            f"An unexpected error occurred while processing the file: {e}\n\n"
            "Try re-exporting the CSV from Google Sheets and uploading again."
        )
        st.stop()
elif st.session_state.get("sheets_url"):
    with st.spinner("Fetching data from Google Sheets..."):
        try:
            file_bytes = _sheets_url_to_csv_bytes(st.session_state["sheets_url"])
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("sheets_url", None)
            st.stop()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error(
            f"Sheet is too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )
        st.stop()
    audit.file_uploaded(len(file_bytes))
    try:
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Could not parse the sheet as CSV: {e}")
        st.stop()
    validation_errors = _validate_csv(raw_df)
    if validation_errors:
        st.error("**The sheet has the following issues:**")
        for err in validation_errors:
            st.markdown(f"- {err}")
        st.stop()
    try:
        edges = _load_and_build(file_bytes)
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the sheet: {e}")
        st.stop()
    st.success("Loaded from Google Sheets.")
elif st.session_state.get("sample_data") is not None:
    sample_bytes = pd.DataFrame(st.session_state["sample_data"]).to_csv(index=False).encode()
    edges = _load_and_build(sample_bytes)
else:
    st.caption("Upload a CSV, paste a Google Sheets URL, or load the example data to explore.")
    st.stop()

# ─── Build full graph ──────────────────────────────────────────────────────────

G = nx.DiGraph()
for u, v, cat in edges:
    G.add_edge(u, v, category=cat)

if G.number_of_edges() == 0:
    st.warning(
        "No nominations were found in the uploaded data. "
        "This usually means the nomination columns are empty, or their headers "
        "don't contain the words 'Inclusive', 'Helpful', or 'Collaborator'. "
        "Please check your CSV and try again."
    )
    st.stop()

# ─── Dataset metrics ───────────────────────────────────────────────────────────

st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Students", len(G.nodes()))
m2.metric("Nominations", len(G.edges()))
m3.metric("Types active", len(selected_categories))

# ─── Filtered graph ────────────────────────────────────────────────────────────

G_filtered = nx.DiGraph()
G_filtered.add_nodes_from(G.nodes())
for u, v, cat in edges:
    if cat in selected_categories:
        G_filtered.add_edge(u, v, category=cat)

in_degrees_filtered = dict(G_filtered.in_degree())

partition = None
if cluster_coloring:
    filtered_edges = tuple((u, v, cat) for u, v, cat in edges if cat in selected_categories)
    try:
        partition = _louvain_partition(filtered_edges)
        unique_groups = sorted(set(partition.values()))
        color_map = cm.get_cmap("tab10", len(unique_groups))
        node_colors = [color_map(partition.get(n, 0)) for n in G_filtered.nodes()]
    except Exception as e:
        st.warning(
            f"Cluster detection could not be completed ({e}). "
            "Falling back to popularity colouring."
        )
        cluster_coloring = False

if not cluster_coloring:
    max_deg = max(max(in_degrees_filtered.values()) if in_degrees_filtered else 0, 1)
    norm = Normalize(vmin=0, vmax=max_deg)
    node_colors = [cm.viridis(norm(in_degrees_filtered.get(n, 0))) for n in G_filtered.nodes()]

if not selected_categories:
    st.info("Select at least one nomination type in the sidebar to display the sociogram.")
    st.stop()

# ─── Class insights ───────────────────────────────────────────────────────────

with st.expander("Class insights", expanded=True):
    ins1, ins2, ins3 = st.columns(3)

    # Most nominated (by in-degree on filtered graph)
    sorted_by_degree = sorted(in_degrees_filtered.items(), key=lambda x: x[1], reverse=True)
    top_nominated = [(n, d) for n, d in sorted_by_degree if d > 0][:3]

    with ins1:
        st.markdown("**Most nominated**")
        if top_nominated:
            for name, deg in top_nominated:
                st.markdown(f"- **{name}** — {deg} nomination{'s' if deg != 1 else ''}")
        else:
            st.caption("No nominations in selected categories.")

    # Social bridges (betweenness centrality on undirected filtered graph)
    with ins2:
        st.markdown("**Social bridges**")
        st.caption("Students who connect different groups.")
        if G_filtered.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(G_filtered.to_undirected())
            top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            top_bridges = [(n, s) for n, s in top_bridges if s > 0][:3]
            if top_bridges:
                for name, score in top_bridges:
                    st.markdown(f"- **{name}** — score {score:.2f}")
            else:
                st.caption("No bridges detected.")
        else:
            st.caption("No edges in selected categories.")

    # Receiving no nominations (in-degree = 0)
    with ins3:
        st.markdown("**Receiving no nominations**")
        st.caption("Not nominated by anyone in selected categories.")
        no_nominations = [n for n in G_filtered.nodes() if in_degrees_filtered.get(n, 0) == 0]
        if no_nominations:
            for name in sorted(no_nominations):
                st.markdown(f"- {name}")
        else:
            st.success("Every student received at least one nomination.")


# ─── Graph drawing helper ──────────────────────────────────────────────────────

rads = {"Inclusive": -0.7, "Helpful": 0.0, "Collaborator": 0.7}
scale = 100


def _draw_sociogram(
    graph, layout, cats_to_draw, node_colors, title,
    highlight_student=None, mutual_pairs=None,
):
    in_deg = dict(graph.in_degree())
    nodes = list(graph.nodes())

    # ── Node sizes ────────────────────────────────────────────────────────────
    if highlight_student and highlight_student in graph:
        neighbours = (
            set(graph.successors(highlight_student))
            | set(graph.predecessors(highlight_student))
        )
        sizes = [
            (in_deg.get(n, 0) + 1) ** 2 * scale * (3 if n == highlight_student else 1.4 if n in neighbours else 0.8)
            for n in nodes
        ]
    else:
        sizes = [(in_deg.get(n, 0) + 1) ** 2 * scale for n in nodes]

    # ── Node colours ──────────────────────────────────────────────────────────
    if highlight_student and highlight_student in graph:
        neighbours = (
            set(graph.successors(highlight_student))
            | set(graph.predecessors(highlight_student))
        )
        draw_colors = [
            "#FF6B35" if n == highlight_student
            else "#93C5FD" if n in neighbours
            else "#E5E7EB"
            for n in nodes
        ]
    else:
        draw_colors = node_colors

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_facecolor("#F5F7FA")

    # ── Mutual nomination highlight (drawn first, behind category edges) ──────
    if mutual_pairs:
        mutual_edgelist = [
            (u, v) for u, v in graph.edges()
            if frozenset([u, v]) in mutual_pairs
        ]
        if mutual_edgelist:
            nx.draw_networkx_edges(
                graph, layout,
                edgelist=mutual_edgelist,
                edge_color="#7C3AED",
                arrowstyle="-|>",
                arrowsize=0,
                width=6,
                connectionstyle="arc3,rad=0.0",
                ax=ax,
                alpha=0.35,
            )

    nx.draw_networkx_nodes(
        graph, layout,
        node_size=sizes,
        node_color=draw_colors,
        edgecolors="black",
        linewidths=1,
        ax=ax,
    )
    for cat in cats_to_draw:
        color = categories[cat]
        edgelist = [(u, v) for u, v, d in graph.edges(data=True) if d.get("category") == cat]
        nx.draw_networkx_edges(
            graph, layout,
            edgelist=edgelist,
            edge_color=color,
            arrowstyle="-|>",
            arrowsize=20,
            width=2,
            connectionstyle=f"arc3,rad={rads[cat]}",
            ax=ax,
        )
    nx.draw_networkx_labels(graph, layout, font_size=10, ax=ax)
    legend_handles = [Patch(facecolor=categories[cat], label=cat) for cat in cats_to_draw]
    if mutual_pairs:
        legend_handles.append(Patch(facecolor="#7C3AED", label="Mutual", alpha=0.5))
    ax.legend(handles=legend_handles, title="Nomination Type", loc="lower left")
    ax.set_title(title, fontsize=16, color="#002664", fontweight="bold")
    ax.axis("off")
    return fig


# ─── Sociogram tabs ────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Sociogram")

in_degrees = dict(G.in_degree())
pos = _spring_layout(tuple(edges))
overview_colors = ["lightgray"] * len(G.nodes())

# Highlight student selector
all_students_sorted = sorted(G_filtered.nodes())
hl_col, _ = st.columns([3, 5])
with hl_col:
    highlight_student = st.selectbox(
        "Highlight student",
        options=["None"] + all_students_sorted,
        help="Enlarges the selected student, highlights their connections, and fades everyone else.",
    )
highlight_student = None if highlight_student == "None" else highlight_student

# Mutual pairs (computed once, used in both tabs)
if show_mutual and G_filtered.number_of_edges() > 0:
    outgoing = {(u, v) for u, v, cat in edges if cat in selected_categories}
    mutual_pairs = frozenset(
        frozenset([u, v]) for (u, v) in outgoing if (v, u) in outgoing
    )
else:
    mutual_pairs = None

tab_overview, tab_filtered = st.tabs(["Overview", "Filtered view"])

with tab_overview:
    fig_overview = _draw_sociogram(
        G, pos, list(categories.keys()), overview_colors, "All nominations",
        highlight_student=highlight_student,
        mutual_pairs=mutual_pairs,
    )
    st.pyplot(fig_overview)
    buf = io.BytesIO()
    fig_overview.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        "Download as PNG", data=buf,
        file_name="sociogram_overview.png", mime="image/png",
    )
    plt.close(fig_overview)

with tab_filtered:
    fig_filtered = _draw_sociogram(
        G_filtered, pos, selected_categories, node_colors, "Filtered by selected types",
        highlight_student=highlight_student,
        mutual_pairs=mutual_pairs,
    )
    st.pyplot(fig_filtered)
    buf = io.BytesIO()
    fig_filtered.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        "Download as PNG", data=buf,
        file_name="sociogram_filtered.png", mime="image/png",
    )
    plt.close(fig_filtered)

# ─── Student detail panel ─────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Student detail")

selected_student = st.selectbox(
    "Select a student to see their nomination profile",
    options=["— select a student —"] + all_students_sorted,
    label_visibility="collapsed",
)

if selected_student != "— select a student —":
    # Incoming: who nominated this student
    received = {}
    for src, tgt, cat in edges:
        if tgt == selected_student and cat in selected_categories:
            received.setdefault(cat, []).append(src)

    # Outgoing: who this student nominated
    gave = {}
    for src, tgt, cat in edges:
        if src == selected_student and cat in selected_categories:
            gave.setdefault(cat, []).append(tgt)

    # Betweenness score
    if G_filtered.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(G_filtered.to_undirected())
        b_score = betweenness.get(selected_student, 0.0)
    else:
        b_score = 0.0

    total_received = sum(len(v) for v in received.values())

    st.markdown(f"""
    <div style="
        background:#fff;border:1px solid #D7DCE0;border-radius:8px;
        padding:1.25rem 1.5rem;margin-bottom:1rem;
    ">
        <div style="font-size:1.2rem;font-weight:700;color:#002664;margin-bottom:0.25rem;">
            {selected_student}
        </div>
        <div style="color:#555;font-size:0.85rem;">
            {total_received} nomination{'s' if total_received != 1 else ''} received
            &nbsp;·&nbsp;
            Bridge score: {b_score:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_recv, col_gave = st.columns(2)

    with col_recv:
        st.markdown("**Nominated by**")
        if received:
            for cat in selected_categories:
                nominators = received.get(cat, [])
                if nominators:
                    color = categories[cat]
                    st.markdown(
                        f"<span style='color:{color};font-weight:600;'>{cat}</span>: "
                        + ", ".join(nominators),
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Not nominated by anyone in the selected categories.")

    with col_gave:
        st.markdown("**Nominated**")
        if gave:
            for cat in selected_categories:
                nominees = gave.get(cat, [])
                if nominees:
                    color = categories[cat]
                    st.markdown(
                        f"<span style='color:{color};font-weight:600;'>{cat}</span>: "
                        + ", ".join(nominees),
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("Did not nominate anyone in the selected categories.")

# ─── Nomination summary ────────────────────────────────────────────────────────

summary_counts = {
    student: {"Inclusive": 0, "Helpful": 0, "Collaborator": 0}
    for student in G_filtered.nodes()
}
for _, target, cat in edges:
    if cat in selected_categories and target in summary_counts:
        summary_counts[target][cat] += 1

st.markdown("---")
st.subheader("Nomination summary")

summary_table = pd.DataFrame([
    {"Student": student, "Total": sum(counts.values()), **counts}
    for student, counts in summary_counts.items()
]).sort_values(by="Total", ascending=False)

st.dataframe(
    summary_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Total": st.column_config.NumberColumn("Total", help="Total nominations received across all selected types"),
        "Inclusive": st.column_config.NumberColumn("Inclusive"),
        "Helpful": st.column_config.NumberColumn("Helpful"),
        "Collaborator": st.column_config.NumberColumn("Collaborator"),
    },
)
st.caption("Counts reflect the currently selected nomination types only.")

csv_export = summary_table.to_csv(index=False).encode("utf-8")
if st.download_button(
    label="Download summary table (CSV)",
    data=csv_export,
    file_name="sociogram_summary_table.csv",
    mime="text/csv",
):
    audit.csv_exported(
        student_count=len(summary_table),
        categories_shown=selected_categories,
    )

# ─── PDF export ────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Export report")

with st.expander("Generate PDF report"):
    st.markdown(
        "The PDF report includes top nominated students, per-category breakdowns, "
        "cluster groups (if enabled), and a list of socially isolated students."
    )
    if st.button("Generate PDF", type="primary"):
        audit.pdf_exported(
            student_count=len(G_filtered.nodes()),
            categories_shown=selected_categories,
        )
        with st.spinner("Generating PDF..."):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, txt="Sociogram Summary Report", ln=True, align="C")
                pdf.ln(10)
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 10, txt=f"Total Students: {len(G_filtered.nodes())}", ln=True)
                pdf.cell(0, 10, txt=f"Total Nominations: {len(G_filtered.edges())}", ln=True)
                if selected_categories != list(categories.keys()):
                    shown = ", ".join(selected_categories) if selected_categories else "none"
                    pdf.cell(0, 10, txt=f"Showing categories: {shown}", ln=True)

                def _row(text: str):
                    safe = text.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.multi_cell(0, 10, txt=safe)

                pdf.ln(10)
                pdf.cell(0, 10, txt="Top 5 Most Nominated Students:", ln=True)
                top5 = sorted(in_degrees_filtered.items(), key=lambda x: x[1], reverse=True)[:5]
                for name, deg in top5:
                    _row(f"- {name}: {deg} nominations")

                pdf.ln(10)
                pdf.cell(0, 10, txt="Top 3 Nominated Students in Each Category:", ln=True)
                for cat in categories:
                    cat_counts = {s: c[cat] for s, c in summary_counts.items() if c[cat] > 0}
                    top3 = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    pdf.cell(0, 10, txt=f"{cat}:", ln=True)
                    for name, count in top3:
                        _row(f"- {name}: {count} nominations")

                if cluster_coloring and partition is not None:
                    pdf.ln(10)
                    group_membership = {v: [] for v in set(partition.values())}
                    for name, group in partition.items():
                        group_membership[group].append(name)
                    pdf.cell(0, 10, txt="Cluster Groups:", ln=True)
                    for group, members in group_membership.items():
                        members_str = ", ".join(members[:5]) + ("..." if len(members) > 5 else "")
                        _row(f"- Group {group}: {members_str}")

                pdf.ln(10)
                pdf.cell(0, 10, txt="Socially Isolated Students:", ln=True)
                for n in G_filtered.nodes():
                    if in_degrees_filtered.get(n, 0) == 0:
                        _row(f"- {n}")

                pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button(
                    label="Download PDF",
                    data=pdf_output_bytes,
                    file_name="sociogram_summary.pdf",
                    mime="application/pdf",
                )
                st.success("PDF ready to download.")
            except Exception as e:
                st.error(
                    f"PDF generation failed: {e}\n\n"
                    "Please try again. If the problem persists, check that student names "
                    "do not contain unsupported special characters."
                )
