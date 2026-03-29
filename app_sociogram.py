# app_sociogram.py

import io
from collections import defaultdict

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

# ─── Authentication ────────────────────────────────────────────────
st.set_page_config(page_title="Sociogram Generator", layout="wide")
current_user = auth.require_auth()
audit.session_start()

# ─── Sidebar: user identity & logout ──────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(f"**Signed in as**")
    st.markdown(f"{current_user['display_name']}")
    st.caption(current_user['email'])
    if st.button("Sign out", use_container_width=True):
        auth.logout()

st.title("📊 Sociogram Generator")

st.markdown('''
> **ℹ️ This Sociogram Generator is based on student responses to a survey.**
> To access your own copy of the survey for use with your students, click here:
> [Google Form Template](https://docs.google.com/forms/d/16ARyYjgnF0SN-5VO3ZNriftCPjHhI94ylKUk7t8jiFk/copy)
''')

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# ─── Privacy Collection Notice ────────────────────────────────────────
# Required under the NSW Privacy and Personal Information Protection Act 1998
# and the NSW DoE Privacy Code of Practice before personal information is
# collected. Shown once per session; must be explicitly acknowledged.

if not st.session_state.get("privacy_acknowledged"):
    st.info(
        "**Privacy Collection Notice**\n\n"
        "This tool collects student names and peer nomination responses "
        "submitted via your class survey. This information is collected for "
        "the purpose of generating a sociogram to help you understand social "
        "connections within your class.\n\n"
        "**How your data is handled:**\n"
        "- Data is processed in your browser session only and is **not stored** "
        "on any server after your session ends.\n"
        "- Data is **not shared** with any third party.\n"
        "- You are responsible for ensuring students have been appropriately "
        "informed that their responses will be used for this purpose.\n\n"
        "This tool is operated in accordance with the "
        "[NSW Privacy and Personal Information Protection Act 1998]"
        "(https://legislation.nsw.gov.au/view/html/inforce/current/act-1998-133) "
        "and the NSW Department of Education Privacy Code of Practice.\n\n"
        "By continuing, you confirm you are authorised to collect and view "
        "this student information in your professional capacity as a NSW DoE staff member."
    )
    if st.button("I understand — continue to the tool"):
        st.session_state["privacy_acknowledged"] = True
        st.rerun()
    st.stop()

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if st.button("📥 Load Example Data"):
    audit.sample_data_loaded()
    sample_data = {
        'Timestamp': ['2025-04-01'] * 5,
        'Your name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'Inclusive - Choice 1': ['Bob', 'Charlie', 'David', 'Eva', 'Alice'],
        'Inclusive - Choice 2': ['Charlie', '', '', '', 'Bob'],
        'Helpful - Choice 1': ['Eva', 'David', '', 'Charlie', ''],
        'Helpful - Choice 2': ['', 'Alice', 'Bob', '', 'David'],
        'Collaborator - Choice 1': ['David', '', 'Eva', 'Bob', 'Charlie'],
        'Collaborator - Choice 2': ['', '', '', 'Alice', '']
    }
    st.session_state["sample_data"] = sample_data
    st.success("Loaded example data. You can explore the sociogram now!")
    st.session_state["example_loaded"] = True



# ─── Read CSV and Normalize ───────────────────────────────────────────

def _validate_csv(dataframe: pd.DataFrame) -> list[str]:
    """
    Returns a list of human-readable error strings.
    Empty list means the dataframe is valid.
    """
    errors = []
    if dataframe.empty:
        errors.append("The uploaded file contains no rows.")
        return errors
    if len(dataframe.columns) < 2:
        errors.append(
            "Expected at least 2 columns (Timestamp, Your name) but found "
            f"{len(dataframe.columns)}."
        )
        return errors

    # Check the name column is present and non-empty
    name_column = dataframe.columns[1]
    if dataframe[name_column].dropna().eq("").all():
        errors.append(f'Column "{name_column}" (student names) appears to be empty.')

    # Check at least one nomination column exists
    nomination_cols = [
        c for c in dataframe.columns
        if any(cat in c for cat in ["Inclusive", "Helpful", "Collaborator"])
    ]
    if not nomination_cols:
        errors.append(
            "No nomination columns found. Expected columns containing "
            '"Inclusive", "Helpful", or "Collaborator" (e.g. "Inclusive - Choice 1").'
        )

    return errors


if uploaded_file is not None:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error(
            f"File is too large ({uploaded_file.size / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )
        st.stop()
    audit.file_uploaded(uploaded_file.size)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()
    validation_errors = _validate_csv(df)
    if validation_errors:
        st.error("**The uploaded file has the following issues:**")
        for err in validation_errors:
            st.markdown(f"- {err}")
        st.markdown(
            "Please check that you exported from the correct Google Form response sheet "
            "and that the column headers haven't been renamed."
        )
        st.stop()
elif st.session_state.get("sample_data") is not None:
    df = pd.DataFrame(st.session_state["sample_data"])
else:
    st.info("Please upload a CSV exported from your Google Sheet.")
    st.stop()

# Identify name column by header name, falling back to position
_name_col_candidates = [c for c in df.columns if "name" in c.lower()]
name_col = _name_col_candidates[0] if _name_col_candidates else df.columns[1]

df[name_col] = df[name_col].astype(str).str.strip().str.title()
for col in df.columns:
    if any(cat in col for cat in ["Inclusive", "Helpful", "Collaborator"]):
        df[col] = df[col].astype(str).str.strip().str.title()

# Fix: after title casing, convert 'Nan' strings back to real NaNs
df.replace("Nan", pd.NA, inplace=True)

# ─── Build Nominations and Graph ─────────────────────────────────────

# Define categories and colors
categories = {
    "Inclusive": "green",
    "Helpful": "blue",
    "Collaborator": "red"
}

# Build edges from nominations
edges = []
for _, row in df.iterrows():
    source = str(row[name_col]).strip()
    for cat in categories:
        for i in (1, 2):
            col = f"{cat} - Choice {i}"
            if col in df.columns:
                target = row[col]
                if pd.notna(target) and str(target).strip():
                    target = str(target).strip()
                    edges.append((source, target, cat))

# Create the directed graph
G = nx.DiGraph()
for u, v, cat in edges:
    G.add_edge(u, v, category=cat)

# ─── Shared graph drawing helper ─────────────────────────────────────

# Edge curvature offsets keep the three category arrows visually separated.
rads = {"Inclusive": -0.7, "Helpful": 0.0, "Collaborator": 0.7}
scale = 100


def _draw_sociogram(
    graph: nx.DiGraph,
    layout: dict,
    cats_to_draw: list,
    node_colors,
    title: str,
) -> plt.Figure:
    """
    Render a sociogram and return the matplotlib Figure.

    Args:
        graph:        The directed graph to draw.
        layout:       Pre-computed node position dict (from nx.*_layout).
        cats_to_draw: Which nomination categories to include as edges.
        node_colors:  Per-node colour list (popularity gradient or cluster).
        title:        Figure title string.
    """
    in_deg = dict(graph.in_degree())
    sizes = [(in_deg.get(n, 0) + 1) ** 2 * scale for n in graph.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(
        graph, layout,
        node_size=sizes,
        node_color=node_colors,
        edgecolors='black',
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
            arrowstyle='-|>',
            arrowsize=20,
            width=2,
            connectionstyle=f'arc3,rad={rads[cat]}',
            ax=ax,
        )
    nx.draw_networkx_labels(graph, layout, font_size=10, ax=ax)
    legend_handles = [
        Patch(facecolor=categories[cat], label=cat) for cat in cats_to_draw
    ]
    ax.legend(handles=legend_handles, title='Nomination Type', loc='lower left')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    return fig


# ─── Compute Layout and draw overview graph ──────────────────────────

in_degrees = dict(G.in_degree())
pos = nx.spring_layout(G, seed=42)

overview_colors = ['lightgray'] * len(G.nodes())
st.pyplot(_draw_sociogram(G, pos, list(categories.keys()), overview_colors, 'Sociogram'))

# ─── Add Sidebar Filters and Cluster Coloring ───────────────────────

# Sidebar settings
st.sidebar.header("Settings")
selected_categories = st.sidebar.multiselect(
    "Select nomination types to display",
    options=list(categories.keys()),
    default=list(categories.keys())
)

cluster_coloring = st.sidebar.checkbox(
    "Color nodes by group (cluster) instead of popularity",
    value=False
)

# ─── Build Filtered Graph ─────────────────────────────────────────────
# Rebuild the graph using only the selected categories so that node
# sizes, in-degrees, clustering, and stats all reflect what is shown.

G_filtered = nx.DiGraph()
G_filtered.add_nodes_from(G.nodes())  # preserve all nodes (avoid layout jumps)
for u, v, cat in edges:
    if cat in selected_categories:
        G_filtered.add_edge(u, v, category=cat)

in_degrees_filtered = dict(G_filtered.in_degree())

# If clustering enabled, compute communities on the filtered graph
partition = None
if cluster_coloring:
    try:
        partition = community_louvain.best_partition(G_filtered.to_undirected())
        unique_groups = sorted(set(partition.values()))
        color_map = cm.get_cmap('tab10', len(unique_groups))
        node_colors = [color_map(partition[n]) for n in G_filtered.nodes()]
    except Exception as e:
        st.warning(
            f"Cluster detection could not be completed ({e}). "
            "This can happen when the graph has no edges. Falling back to popularity colouring."
        )
        cluster_coloring = False
if not cluster_coloring:
    max_deg = max(in_degrees_filtered.values()) if in_degrees_filtered else 1
    norm = Normalize(vmin=0, vmax=max_deg)
    node_colors = [cm.viridis(norm(in_degrees_filtered.get(n, 0))) for n in G_filtered.nodes()]

# ─── Draw filtered graph ─────────────────────────────────────────────

st.pyplot(_draw_sociogram(G_filtered, pos, selected_categories, node_colors, 'Sociogram (Filtered)'))

# ─── Generate PDF Report ─────────────────────────────────────────────

if st.button("📄 Generate PDF Report"):
    audit.pdf_exported(
        student_count=len(G_filtered.nodes()),
        categories_shown=selected_categories,
    )
    with st.spinner("Generating PDF Report..."):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Sociogram Summary Report", ln=True, align='C')

            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Total Students: {len(G_filtered.nodes())}", ln=True)
            pdf.cell(200, 10, txt=f"Total Nominations: {len(G_filtered.edges())}", ln=True)
            if selected_categories != list(categories.keys()):
                shown = ", ".join(selected_categories) if selected_categories else "none"
                pdf.cell(200, 10, txt=f"Showing categories: {shown}", ln=True)

            pdf.ln(10)
            pdf.cell(200, 10, txt="Top 5 Most Nominated Students:", ln=True)
            top5 = sorted(in_degrees_filtered.items(), key=lambda x: x[1], reverse=True)[:5]
            for name, deg in top5:
                safe_name = str(name).encode("latin-1", errors="replace").decode("latin-1")
                pdf.cell(200, 10, txt=f"- {safe_name}: {deg} nominations", ln=True)

            pdf.ln(10)
            pdf.cell(200, 10, txt="Top 3 Nominated Students in Each Category:", ln=True)
            for cat in categories:
                nomination_counts = {}
                for _, target, c in edges:
                    if c == cat:
                        nomination_counts[target] = nomination_counts.get(target, 0) + 1
                top3 = sorted(nomination_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                pdf.cell(200, 10, txt=f"{cat}:", ln=True)
                for name, count in top3:
                    safe_name = str(name).encode("latin-1", errors="replace").decode("latin-1")
                    pdf.cell(200, 10, txt=f"- {safe_name}: {count} nominations", ln=True)

            if cluster_coloring and partition is not None:
                pdf.ln(10)
                group_membership = {v: [] for v in set(partition.values())}
                for name, group in partition.items():
                    group_membership[group].append(name)
                pdf.cell(200, 10, txt="Cluster Groups:", ln=True)
                for group, members in group_membership.items():
                    members_str = ", ".join(members[:5]) + ("..." if len(members) > 5 else "")
                    safe_str = members_str.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.cell(200, 10, txt=f"- Group {group}: {safe_str}", ln=True)

            pdf.ln(10)
            pdf.cell(200, 10, txt="Socially Isolated Students:", ln=True)
            for n in G_filtered.nodes():
                if in_degrees_filtered.get(n, 0) == 0:
                    safe_name = str(n).encode("latin-1", errors="replace").decode("latin-1")
                    pdf.cell(200, 10, txt=f"- {safe_name}", ln=True)

            pdf_output_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="⬇️ Download PDF Summary",
                data=pdf_output_bytes,
                file_name="sociogram_summary.pdf",
                mime="application/pdf"
            )
            st.success("✅ PDF generated successfully! Ready to download.")
        except Exception as e:
            st.error(
                f"PDF generation failed: {e}\n\n"
                "Please try again. If the problem persists, check that student names "
                "do not contain unsupported special characters."
            )

# ─── Export Full Summary Table ──────────────────────────────────────────

summary_counts = {student: {"Inclusive": 0, "Helpful": 0, "Collaborator": 0} for student in G_filtered.nodes()}

for _, target, cat in edges:
    if cat in selected_categories and target in summary_counts:
        summary_counts[target][cat] += 1

summary_table = pd.DataFrame([
    {
        "Student": student,
        "Total": sum(counts.values()),
        **counts
    }
    for student, counts in summary_counts.items()
])

# Sort by Total nominations descending
summary_table = summary_table.sort_values(by="Total", ascending=False)

st.dataframe(summary_table)

csv_export = summary_table.to_csv(index=False).encode('utf-8')

if st.download_button(
    label="⬇️ Download Full Summary Table (CSV)",
    data=csv_export,
    file_name="sociogram_summary_table.csv",
    mime="text/csv"
):
    audit.csv_exported(
        student_count=len(summary_table),
        categories_shown=selected_categories,
    )
