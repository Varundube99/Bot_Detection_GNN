import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Bot Detection using GNN",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1f26;
        padding-top: 30px;
        width: 290px;
    }

    /* Sidebar title */
    section[data-testid="stSidebar"] h1 {
        font-size: 28px;
        margin-bottom: 25px;
        font-weight: 600;
    }

    /* Space between navigation buttons */
    div[role="radiogroup"] > label {
        margin-top: 19px;
        margin-bottom: 19px;
        padding: 8px 6px;
        font-size: 18px;
        border-radius: 6px;
    }

    /* Hover effect */
    div[role="radiogroup"] > label:hover {
        background-color: rgba(255,255,255,0.08);
    }

    /* Selected navigation item */
    div[role="radiogroup"] input:checked + div {
        font-weight: 600;
        color: #ff4b4b;
    }

    /* Page title styling */
    h1 {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 25px;
    }

    /* Metric cards styling */
    div[data-testid="metric-container"] {
        padding: 16px;
        border-radius: 12px;
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Buttons styling */
    button {
        border-radius: 8px !important;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Add spacing between sidebar buttons
st.markdown(
    """
    <style>
    div[role="radiogroup"] > label {
        margin-bottom: 12px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Network Graph",
        "User Prediction",
        "Network Insights"
    ]
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("model/bot_detection_results.csv")
    edges = pd.read_csv("model/edges.csv").values
    return df, edges


df, edges = load_data()

# --------------------------------------------------
# Build Graph
# --------------------------------------------------

@st.cache_resource
def build_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


G = build_graph(edges)

# --------------------------------------------------
# Model Definition
# --------------------------------------------------

class GCN(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 32)

        self.classifier = Linear(32, 2)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.classifier(x)

        return x

# --------------------------------------------------
# Page 1 — Network Graph
# --------------------------------------------------

if page == "Network Graph":

    st.title("Network Graph Visualization (500 Nodes)")

    sample_nodes = list(G.nodes())[:500]
    G_small = G.subgraph(sample_nodes).copy()

    node_colors_small = [
        "red" if df.loc[node, "Predicted Label"] == 1
        else "green"
        for node in G_small.nodes()
    ]

    pos_small = nx.spring_layout(
        G_small,
        seed=42
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_nodes(
        G_small,
        pos_small,
        node_color=node_colors_small,
        node_size=30,
        alpha=0.9,
        ax=ax
    )

    nx.draw_networkx_edges(
        G_small,
        pos_small,
        alpha=0.2,
        width=0.5,
        ax=ax
    )

    ax.axis("off")

    st.pyplot(fig)

# --------------------------------------------------
# Page 2 — Check a User
# --------------------------------------------------

elif page == "User Prediction":

    st.title("Check if a User is a Bot")

    retweet = st.number_input(
        "Retweet Count",
        min_value=0
    )

    mention = st.number_input(
        "Mention Count",
        min_value=0
    )

    followers = st.number_input(
        "Follower Count",
        min_value=0
    )

    verified = st.selectbox(
        "Verified",
        ["No", "Yes"]
    )

    tweet_freq = st.number_input(
        "Tweet Frequency",
        min_value=0
    )

    hashtag = st.number_input(
        "Hashtag Count",
        min_value=0
    )

    account_age = st.number_input(
        "Account Age (days)",
        min_value=0
    )

    if st.button("Detect Bot"):

        try:

            with open("model/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            model = GCN(in_channels=8)
            model.load_state_dict(
                torch.load(
                    "model/gnn_model.pth",
                    map_location="cpu"
                )
            )

            model.eval()

            verified_value = 1 if verified == "Yes" else 0

            engagement_ratio = (
                retweet /
                (followers + 1)
            )

            input_data = np.array([[
                retweet,
                mention,
                followers,
                verified_value,
                tweet_freq,
                hashtag,
                account_age,
                engagement_ratio
            ]])

            input_scaled = scaler.transform(
                input_data
            )

            input_tensor = torch.tensor(
                input_scaled,
                dtype=torch.float
            )

            dummy_edge = torch.empty(
                (2, 0),
                dtype=torch.long
            )

            with torch.no_grad():

                output = model(
                    Data(
                        x=input_tensor,
                        edge_index=dummy_edge
                    )
                )

                prob = torch.softmax(
                    output,
                    dim=1
                )

                prediction = prob.argmax().item()

                confidence = prob[
                    0,
                    prediction
                ].item()

            if prediction == 1:

                st.error(
                    f"Prediction: BOT"
                )

            else:

                st.success(
                    f"Prediction: HUMAN"
                )

        except Exception as e:

            st.warning("Model or scaler file not found.")
            st.write(e)

# --------------------------------------------------
# Page 3 — Network Insights
# --------------------------------------------------

elif page == "Network Insights":

    st.title("Network Insights")

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    degrees = [d for n, d in G.degree()]

    avg_degree = sum(degrees) / total_nodes
    max_degree = max(degrees)
    min_degree = min(degrees)

    density = nx.density(G)

    num_components = nx.number_connected_components(G)

    clustering_coeff = nx.average_clustering(G)

    bot_percentage = (
        (df["Predicted Label"] == 1).sum()
        / total_nodes
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", total_nodes)
    col1.metric("Total Connections", total_edges)
    col1.metric("Connected Components", num_components)

    col2.metric("Average Degree", round(avg_degree, 2))
    col2.metric("Max Degree", max_degree)
    col2.metric("Min Degree", min_degree)

    col3.metric("Network Density", round(density, 4))
    col3.metric("Clustering Coefficient", round(clustering_coeff, 4))
    col3.metric("Bot Percentage", f"{bot_percentage:.2f}%")
