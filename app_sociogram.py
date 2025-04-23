# app_sociogram.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.colors import Normalize
from fpdf import FPDF
import community as community_louvain
import io
from collections import defaultdict
import zipfile

# ─── Streamlit UI ────────────────────────────────────────────
st.set_page_config(page_title="Sociogram Generator", layout="wide")
st.title("\U0001F4CA Sociogram Generator")
st.markdown('''
> **ℹ️ This Sociogram Generator is based on student responses to a survey.**  
> To access your own copy of the survey for use with your students, click here:  
> [Google Form Template](https://docs.google.com/forms/d/16ARyYjgnF0SN-5VO3ZNriftCPjHhI94ylKUk7t8jiFk/copy)
''')
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load example button
if st.button("📥 Load Example Data"):
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
    uploaded_file = io.StringIO(pd.DataFrame(sample_data).to_csv(index=False))
    st.success("Loaded example data. You can explore the sociogram now!")
    st.session_state["example_loaded"] = True

if not uploaded_file:
    st.info("Please upload a CSV exported from your Google Sheet.")
    st.stop()

# ─── Read & Parse Data ────────────────────────────────────────
df = pd.read_csv(uploaded_file)
name_col = df.columns[1]  # column B: "Your name"

categories = {
    "Inclusive":    "green",
    "Helpful":      "blue",
    "Collaborator": "red"
}

students = sorted(set(df[name_col].dropna().astype(str).str.strip()).union(
    *[df[c].dropna().astype(str).str.strip() for c in df.columns if any(k in c for k in categories)]
))

with st.expander("⚙️ Advanced Settings", expanded=st.session_state.get("example_loaded", False)):
    category_toggle = {
        cat: st.checkbox(f"Show {cat} edges", value=True)
        for cat in categories
    }
    selected = [cat for cat, show in category_toggle.items() if show]
    is_directed = st.checkbox("Use directed graph", value=True)
    focus_student = st.selectbox("Focus on a specific student (optional)", options=["All"] + students)
    cluster_toggle = st.checkbox("Color nodes by group (cluster) instead of popularity", value=False)

# ─── Build Edge List ──────────────────────────────────────────
edges = []
edge_weights = {}
for _, row in df.iterrows():
    source = str(row[name_col]).strip()
    for cat in categories:
        for i in (1, 2):
            col = f"{cat} - Choice {i}"
            if col in df.columns:
                target = row[col]
                if pd.notna(target) and str(target).strip():
                    target = str(target).strip()
                    edge = (source, target, cat)
                    edges.append(edge)
                    key = (source, target, cat)
                    edge_weights[key] = edge_weights.get(key, 0) + 1

# ─── Graph Construction ───────────────────────────────────────
G = nx.DiGraph() if is_directed else nx.Graph()
for (u, v, cat), weight in edge_weights.items():
    if G.has_edge(u, v):
        G[u][v]['weight'] += weight
    else:
        G.add_edge(u, v, category=cat, weight=weight)

in_degrees = dict(G.in_degree()) if is_directed else dict(G.degree())
scale = 100
node_sizes_dict = {
    n: min((in_degrees.get(n, 0) + 1)**2 * scale, 2500) for n in G.nodes()
}
node_sizes = list(node_sizes_dict.values())

# Color nodes
if cluster_toggle:
    partition = community_louvain.best_partition(G.to_undirected())
    unique_groups = sorted(set(partition.values()))
    color_map = cm.get_cmap('tab10', len(unique_groups))
    node_colors = [color_map(partition[n]) for n in G.nodes()]
else:
    max_deg = max(in_degrees.values()) if in_degrees else 1
    norm = Normalize(vmin=0, vmax=max_deg)
    node_colors = [cm.viridis(norm(in_degrees.get(n, 0))) for n in G.nodes()]

pos = nx.spring_layout(G, seed=42)

# ─── Plotting ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors='black',
    linewidths=1
)

# Tooltips
for node, (x, y) in pos.items():
    count = in_degrees.get(node, 0)
    ax.annotate(
        f"{node}\nNominations: {count}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=8,
        color='gray'
    )

rads = {"Inclusive": -0.7, "Helpful": 0.0, "Collaborator": +0.7}
for cat in selected:
    color = categories[cat]
    edgelist = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("category") == cat and (
            focus_student == "All" or u == focus_student or v == focus_student
        )
    ]
    widths = [G[u][v]['weight'] for u, v in edgelist]
    for (u, v), w in zip(edgelist, widths):
        target_size = node_sizes_dict.get(v, 300)
        arrowsize = max(30, min(80, target_size * 0.05))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=color,
            width=w,
            arrowstyle='-|>' if is_directed else '-',
            arrowsize=arrowsize,
            connectionstyle=f'arc3,rad={rads[cat]}' if is_directed else None,
            arrows=is_directed
        )

label_pos = {k: v for k, v in pos.items() if focus_student == "All" or k == focus_student}
nx.draw_networkx_labels(G, label_pos, font_size=10)

legend_handles = [
    Patch(facecolor=clr, label=cat)
    for cat, clr in categories.items() if cat in selected
]
plt.legend(handles=legend_handles, title='Nomination Type', loc='lower left')

# Colorbar
if not cluster_toggle:
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Number of Nominations')

plt.title('Sociogram', fontsize=14)
plt.axis('off')
st.pyplot(fig)

# ─── Sidebar Summary ──────────────────────────────────────────
st.sidebar.title("📊 Summary")
st.sidebar.markdown(f"**Nodes:** {len(G.nodes())}")
st.sidebar.markdown(f"**Edges:** {len(G.edges())}")
st.sidebar.markdown("Popularity = In-degree (number of nominations received)")
st.sidebar.markdown("Use filters in ⚙️ Advanced Settings to explore variations.")

# Histogram
fig_hist, ax_hist = plt.subplots(figsize=(2.5, 2))
counts = list(in_degrees.values())
ax_hist.hist(counts, bins=range(0, max(counts)+2), color='gray', edgecolor='black')
ax_hist.set_title("Nominations Histogram", fontsize=8)
ax_hist.set_xlabel("Nominations", fontsize=7)
ax_hist.set_ylabel("# of Students", fontsize=7)
ax_hist.tick_params(labelsize=6)
st.sidebar.pyplot(fig_hist)

# Reset Button
if st.sidebar.button("🔄 Reset Filters"):
    for key in ["example_loaded"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# ─── PDF Report ───────────────────────────────────────────────
if st.button("📄 Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Sociogram Summary Report", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Total Students: {len(G.nodes())}", ln=True)
    pdf.cell(200, 10, txt=f"Total Nominations: {len(G.edges())}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Top 5 Most Nominated Students (Overall):", ln=True)
    top5 = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, deg in top5:
        pdf.cell(200, 10, txt=f"- {name}: {deg} nominations", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Top 3 Most Nominated Students by Category:", ln=True)
    for cat in categories:
        cat_degrees = defaultdict(int)
        for u, v, d in G.edges(data=True):
            if d.get("category") == cat:
                cat_degrees[v] += 1
        top3_cat = sorted(cat_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        pdf.cell(200, 10, txt=f"{cat}:", ln=True)
        for name, deg in top3_cat:
            pdf.cell(200, 10, txt=f"   - {name}: {deg} nominations", ln=True)

    if cluster_toggle:
        group_membership = {v: [] for v in set(partition.values())}
        for name, group in partition.items():
            group_membership[group].append(name)
        pdf.ln(5)
        pdf.cell(200, 10, txt="Group (Cluster) Membership Samples:", ln=True)
        for group, members in group_membership.items():
            members_str = ", ".join(members[:5]) + ("..." if len(members) > 5 else "")
            pdf.cell(200, 10, txt=f"- Group {group}: {members_str}", ln=True)

        pdf.ln(5)
        pdf.cell(200, 10, txt="Bridging Students Between Clusters:", ln=True)
        for n in G.nodes():
            if n in partition:
                neighbor_groups = set(partition.get(neigh) for neigh in G.neighbors(n))
                if len(neighbor_groups) > 1:
                    pdf.cell(200, 10, txt=f"- {n} (links to groups {sorted(neighbor_groups)})", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Socially Isolated Students:", ln=True)
    for n in G.nodes():
        if in_degrees.get(n, 0) == 0:
            pdf.cell(200, 10, txt=f"- {n}", ln=True)

    pdf_output_bytes = pdf.output(dest='S').encode('latin-1')
    st.download_button(
        label="⬇️ Download PDF Summary",
        data=pdf_output_bytes,
        file_name="sociogram_summary.pdf",
        mime="application/pdf"
    )
