# app_sociogram.py

import streamlit as st

st.set_page_config(page_title="Sociogram Generator", layout="wide")
st.title("📊 Sociogram Generator")

st.markdown('''
> **ℹ️ This Sociogram Generator is based on student responses to a survey.**  
> To access your own copy of the survey for use with your students, click here:  
> [Google Form Template](https://docs.google.com/forms/d/16ARyYjgnF0SN-5VO3ZNriftCPjHhI94ylKUk7t8jiFk/copy)
''')



# ─── Load and Normalize Data ───────────────────────────────────────
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

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

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
    st.session_state["sample_data"] = sample_data
    st.success("Loaded example data. You can explore the sociogram now!")
    st.session_state["example_loaded"] = True



# ─── Read CSV and Normalize ───────────────────────────────────────────
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif st.session_state.get("sample_data") is not None:
    df = pd.DataFrame(st.session_state["sample_data"])
else:
    st.info("Please upload a CSV exported from your Google Sheet.")
    st.stop()

name_col = df.columns[1]  # column B: "Your name"
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
if source and source.lower() != "nan":
    for cat in categories:
        for i in (1, 2):
            col = f"{cat} - Choice {i}"
            if col in df.columns:
                target = row[col]
                if pd.notna(target):
                    target = str(target).strip()
                    if target and target.lower() != "nan":
                        edges.append((source, target, cat))

# Create the directed graph
G = nx.DiGraph()
for u, v, cat in edges:
    G.add_edge(u, v, category=cat)

# ─── Compute Layout and Plot Graph ──────────────────────────────────

# Compute node in-degrees for sizing
in_degrees = dict(G.in_degree())
scale = 100
node_sizes = [(in_degrees.get(n, 0) + 1) ** 2 * scale for n in G.nodes()]

# Compute layout
pos = nx.spring_layout(G, seed=42)

# Plot the graph
fig, ax = plt.subplots(figsize=(12, 10))
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color='lightgray',
    edgecolors='black',
    linewidths=1
)

# Draw edges colored by category
rads = {"Inclusive": -0.7, "Helpful": 0.0, "Collaborator": 0.7}
for cat in categories:
    color = categories[cat]
    edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get("category") == cat]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edgelist,
        edge_color=color,
        arrowstyle='-|>',
        arrowsize=20,
        width=2,
        connectionstyle=f'arc3,rad={rads[cat]}'
    )

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10)

# Legend
legend_handles = [Patch(facecolor=clr, label=cat) for cat, clr in categories.items()]
plt.legend(handles=legend_handles, title='Nomination Type', loc='lower left')

plt.title('Sociogram', fontsize=16)
plt.axis('off')

st.pyplot(fig)

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

# If clustering enabled, compute communities
if cluster_coloring:
    partition = community_louvain.best_partition(G.to_undirected())
    unique_groups = sorted(set(partition.values()))
    color_map = cm.get_cmap('tab10', len(unique_groups))
    node_colors = [color_map(partition[n]) for n in G.nodes()]
else:
    max_deg = max(in_degrees.values()) if in_degrees else 1
    norm = Normalize(vmin=0, vmax=max_deg)
    node_colors = [cm.viridis(norm(in_degrees.get(n, 0))) for n in G.nodes()]

# ─── Redraw Graph Based on Sidebar Settings ─────────────────────────

# Replot graph with updated filters and node colors
fig, ax = plt.subplots(figsize=(12, 10))
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors='black',
    linewidths=1
)

# Draw only selected category edges
for cat in selected_categories:
    color = categories[cat]
    edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get("category") == cat]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edgelist,
        edge_color=color,
        arrowstyle='-|>',
        arrowsize=20,
        width=2,
        connectionstyle=f'arc3,rad={rads[cat]}'
    )

# Draw labels again
nx.draw_networkx_labels(G, pos, font_size=10)

# Update legend
legend_handles = [Patch(facecolor=clr, label=cat) for cat, clr in categories.items() if cat in selected_categories]
plt.legend(handles=legend_handles, title='Nomination Type', loc='lower left')

plt.title('Sociogram (Filtered)', fontsize=16)
plt.axis('off')

st.pyplot(fig)

# ─── Generate PDF Report ─────────────────────────────────────────────

if st.button("📄 Generate PDF Report"):
    with st.spinner("Generating PDF Report..."):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Sociogram Summary Report", ln=True, align='C')

        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Total Students: {len(G.nodes())}", ln=True)
        pdf.cell(200, 10, txt=f"Total Nominations: {len(G.edges())}", ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, txt="Top 5 Most Nominated Students:", ln=True)
        top5 = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, deg in top5:
            pdf.cell(200, 10, txt=f"- {name}: {deg} nominations", ln=True)

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
                pdf.cell(200, 10, txt=f"- {name}: {count} nominations", ln=True)

        if cluster_coloring:
            pdf.ln(10)
            group_membership = {v: [] for v in set(partition.values())}
            for name, group in partition.items():
                group_membership[group].append(name)
            pdf.cell(200, 10, txt="Cluster Groups:", ln=True)
            for group, members in group_membership.items():
                members_str = ", ".join(members[:5]) + ("..." if len(members) > 5 else "")
                pdf.cell(200, 10, txt=f"- Group {group}: {members_str}", ln=True)

        pdf.ln(10)
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
        st.success("✅ PDF generated successfully! Ready to download.")

# ─── Export Full Summary Table ──────────────────────────────────────────

summary_counts = {student: {"Inclusive": 0, "Helpful": 0, "Collaborator": 0} for student in G.nodes()}

for _, target, cat in edges:
    if target in summary_counts:
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

st.download_button(
    label="⬇️ Download Full Summary Table (CSV)",
    data=csv_export,
    file_name="sociogram_summary_table.csv",
    mime="text/csv"
)
