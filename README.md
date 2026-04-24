# 🤖 Bot Detection in Social Networks using Graph Neural Networks (GNN)

A complete **Graph Neural Network (GNN) application** for detecting automated bot accounts in a simulated social network using **Graph Theory** and **Deep Learning**.

The system models users as **nodes** and their behavioral similarity as **edges**, then applies a neural network to classify accounts as:

* **Bot**
* **Human**

The project includes an interactive **Streamlit dashboard** for visualization, prediction, and network analytics.

---

## 🌐 Live Demo

**Streamlit App**

https://bot-detection-using-gnn.streamlit.app/

---

# 📌 Project Overview

Social media platforms contain both genuine users and automated bot accounts. Bots can:

* Spread misinformation
* Manipulate engagement metrics
* Generate spam content
* Distort online conversations

Traditional machine learning models analyze users independently. However, real-world behavior is influenced by relationships between users.

This project solves that problem using:

* Graph-based modeling
* Graph Neural Networks
* Network analysis

---

# ✨ Key Features

* Graph-based user relationship modeling using **K-Nearest Neighbors (KNN)**
* Graph Neural Network (**GCN**) for bot classification
* Interactive network visualization
* Real-time bot prediction interface
* Network analytics dashboard
* Multi-page Streamlit application
* Deployable machine learning system
* Clean project structure for GitHub

---

# 🧠 How the System Works

The system follows a graph-based machine learning pipeline.

```text
User Data
   ↓
Feature Engineering
   ↓
Graph Construction
   ↓
Graph Neural Network
   ↓
Prediction
   ↓
Visualization
```

The model learns from:

* User behavior
* User relationships
* Network patterns

---

# 🧩 Graph Construction Methodology

The social network graph was generated programmatically using **Graph Theory**.

---

## Step 1 — Node Representation

Each row in the dataset represents a user node.

Each user contains behavioral features:

* Retweet Count
* Mention Count
* Follower Count
* Verified Status
* Tweet Frequency
* Hashtag Count
* Account Age
* Engagement Ratio

These values form the:

```text
Node Feature Vector
```

---

## Step 2 — Feature Scaling

All numerical features were standardized using:

```text
StandardScaler (Scikit-learn)
```

This ensures:

* Equal feature importance
* Stable model training
* Faster convergence

---

## Step 3 — Graph Creation using K-Nearest Neighbors (KNN)

The graph was constructed using:

```text
K-Nearest Neighbors (KNN)
```

Similarity metric:

```text
Euclidean Distance
```

Parameter used:

```text
k = 50
```

Meaning:

Each user is connected to their **50 most similar users**, creating a realistic network structure.

---

## Step 4 — Edge Generation

Connections between users were stored as:

```text
edges.csv
```

Format:

```text
source_node, target_node
```

Example:

```text
12,45
12,78
45,103
```

---

## Step 5 — Graph Conversion for GNN

The graph was converted into a format compatible with:

```text
PyTorch Geometric
```

Required components:

```text
x            → Node features
edge_index   → Graph connections
y            → Labels
```

This enables:

```text
Message Passing between nodes
```

---

# 🤖 Model Architecture

Model Type:

```text
Graph Convolutional Network (GCN)
```

Architecture:

```text
Input Features
      ↓
GCN Layer
      ↓
ReLU
      ↓
GCN Layer
      ↓
Classifier
      ↓
Prediction
```

Output:

```text
0 → Human
1 → Bot
```

---

# 🖥️ Application Pages

## 📊 Network Graph

Displays:

* User relationships
* Node connections
* Network structure
* Bot / Human classification

Helps visualize the social network.

---

## 🔍 Check a User

Users can manually input account behavior.

Input:

* Retweet Count
* Mention Count
* Follower Count
* Verified Status
* Tweet Frequency
* Hashtag Count
* Account Age

Output:

```text
Prediction: BOT / HUMAN
```

---

## 📈 Network Insights

Displays graph statistics.

Metrics include:

* Total Users
* Total Connections
* Average Degree
* Maximum Degree
* Minimum Degree
* Network Density
* Connected Components
* Clustering Coefficient
* Bot Percentage

---

# 📁 Project Structure

```text
Bot_Detection_GNN/
│
├── app.py
├── requirements.txt
│
├── model/
│     ├── gnn_model.pth
│     ├── scaler.pkl
│     ├── edges.csv
│     └── bot_detection_results.csv
│
├── Notebook/
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/Bot_Detection_GNN.git
cd Bot_Detection_GNN
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 🌐 Deployment

This project can be deployed using:

* Streamlit Cloud
* GitHub
* Localhost

Example deployment command:

```bash
streamlit run app.py
```

---

# ⚠️ Limitations

* Synthetic dataset used for demonstration
* Graph connections based on similarity
* Not trained on real social media data
* Large graphs require more memory
* Model accuracy depends on feature quality

---

# 🚀 Future Improvements

* Real social network dataset integration
* Community detection visualization
* Graph filtering and search
* Real-time streaming data
* API-based prediction service
* Model performance dashboard

---

# 🔧 Troubleshooting

Issue: Streamlit not starting

```bash
pip install -r requirements.txt
```

---

Issue: Model not loading

Check:

```text
File path
Dependencies
Model folder
```

---

Issue: Graph visualization slow

Solution:

```text
Reduce node count
Use subset visualization
```

---

# 👨‍💻 Author

Varun Dubey

AI / Machine Learning Enthusiast

GitHub
https://github.com/Varundube99

---

# 📄 License

This project is intended for academic and educational use.

The model and code are provided for demonstration purposes.
