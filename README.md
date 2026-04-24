\# 🤖 Bot Detection in Social Networks using Graph Neural Networks (GNN)



!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

!\[GNN](https://img.shields.io/badge/Model-Graph%20Neural%20Network-green)

!\[Status](https://img.shields.io/badge/Status-Deployed-success)



A complete \*\*Graph Neural Network (GNN) application\*\* for detecting automated bot accounts in a simulated social network using \*\*Graph Theory\*\* and \*\*Graph Neural Networks\*\*.



The system models:



\* Users → Nodes

\* Behavioral similarity → Edges

\* GNN → Bot / Human prediction



The project includes an interactive \*\*Streamlit dashboard\*\* for visualization, prediction, and network analysis.



\---



\## 🚀 Key Features



\* Graph-based user relationship modeling using \*\*K-Nearest Neighbors (KNN)\*\*

\* Graph Neural Network (\*\*GCN\*\*) classification

\* Interactive network graph visualization

\* Real-time bot prediction interface

\* Network analytics and graph metrics dashboard

\* Multi-page Streamlit application

\* Deployable machine learning system



\---



\## 🌐 Live Demo



Try the deployed application:



👉 \*\*https://bot-detection-using-gnn.streamlit.app/\*\*



\---



\## 📌 Project Overview



Social media platforms contain both legitimate users and automated bots. Bots can spread misinformation, manipulate engagement metrics, and disrupt online ecosystems.



Traditional machine learning models analyze users independently. However, real-world behavior is influenced by relationships between users. This project addresses that limitation by modeling the system as a \*\*graph\*\* and applying a \*\*Graph Neural Network (GNN)\*\* to learn patterns from both:



\* User behavior

\* User relationships



\---



\## 🎯 Problem Statement



Detect suspicious bot accounts in a social network by:



\* Modeling users as nodes

\* Creating connections based on behavioral similarity

\* Using Graph Neural Networks for classification

\* Visualizing network structure and analytics



\---



\## 🧠 Core Concepts Used



\### Graph Theory



\* Nodes (Users)

\* Edges (User similarity / interaction)

\* Degree

\* Network Density

\* Connected Components

\* Clustering Coefficient

\* Graph Construction using KNN



\### Machine Learning / Deep Learning



\* Graph Neural Networks (GCN)

\* Node Classification

\* Feature Engineering

\* Feature Scaling

\* Message Passing



\### Data Visualization



\* Network Graph Visualization

\* Real-time Prediction Interface

\* Network Metrics Dashboard



\---



\## 🏗️ System Architecture



User Data

→ Feature Engineering

→ Graph Construction

→ Graph Neural Network

→ Prediction

→ Visualization



\---



\## 🧩 Graph Construction Methodology



A key part of this project is how the \*\*social network graph\*\* was generated from structured user data.



Instead of using an existing network, the graph was programmatically constructed to simulate realistic relationships between users using Graph Theory principles.



\---



\### Step 1 — Node Representation



Each row in the dataset represents a \*\*user node\*\*.



Every user is described using behavioral features:



\* Retweet Count

\* Mention Count

\* Follower Count

\* Verified Status

\* Tweet Frequency

\* Hashtag Count

\* Account Age

\* Engagement Ratio



These features form the:



\*\*Node Feature Vector\*\*



\---



\### Step 2 — Feature Scaling



All numerical features were standardized using:



\*\*StandardScaler (Scikit-learn)\*\*



This ensures:



\* Equal feature importance

\* Stable model training

\* Faster convergence

\* Reduced numerical instability



\---



\### Step 3 — Graph Creation using K-Nearest Neighbors (KNN)



The graph was constructed using a \*\*K-Nearest Neighbors (KNN)\*\* algorithm.



Process:



Each user node is connected to its nearest neighbors based on similarity.



Similarity was calculated using:



\*\*Euclidean Distance\*\*



Parameter used:



\*\*k = 50\*\*



Meaning:



Each user is connected to their \*\*50 most similar users\*\*, forming a realistic social network structure.



\---



\### Step 4 — Edge Generation



Connections between users were stored as an \*\*edge list\*\*.



Format:



```

source\_node, target\_node

```



Example:



```

12, 45

12, 78

45, 103

```



Edges were saved into:



```

edges.csv

```



This file represents the complete graph structure.



\---



\### Step 5 — Graph Conversion for GNN



The graph was converted into a format compatible with:



\*\*PyTorch Geometric\*\*



Required components:



```

x            → Node feature matrix

edge\_index   → Graph connectivity

y            → Node labels

```



This structure enables:



\*\*Message Passing between nodes\*\*



Which allows the model to learn:



\* Local relationships

\* Network patterns

\* Coordinated behavior



\---



\## 🤖 Model Architecture



Model Type:



\*\*Graph Convolutional Network (GCN)\*\*



Architecture:



```

Input Features

&#x20;     ↓

GCN Layer

&#x20;     ↓

ReLU

&#x20;     ↓

GCN Layer

&#x20;     ↓

Classifier

&#x20;     ↓

Prediction

```



Output:



```

0 → Human

1 → Bot

```



\---



\## 🚀 Application Features



\### Network Graph Visualization



Displays:



\* Social network structure

\* User relationships

\* Bot / Human classification

\* Node connectivity patterns



Bots are highlighted using color coding.



\---



\### Check a User (Prediction Interface)



Users can manually input account features to detect whether the account is a bot.



Input Fields:



\* Retweet Count

\* Mention Count

\* Follower Count

\* Verified Status

\* Tweet Frequency

\* Hashtag Count

\* Account Age



Output:



```

Prediction: BOT / HUMAN

Confidence Score

```



\---



\### Network Insights Dashboard



Provides real graph analytics:



\* Total Users

\* Total Connections

\* Average Degree

\* Maximum Degree

\* Minimum Degree

\* Network Density

\* Connected Components

\* Clustering Coefficient

\* Bot Percentage



These metrics help analyze network structure and behavior.



\---



\## 📂 Project Structure



```

Bot\_Detection\_GNN/

│

├── app.py

├── requirements.txt

│

├── model/

│     ├── gnn\_model.pth

│     ├── scaler.pkl

│     ├── edges.csv

│     └── bot\_detection\_results.csv

│

├── Notebook/

├── .gitignore

└── README.md

```



\---



\## ⚙️ Installation (Run Locally)



\### Clone Repository



```

git clone https://github.com/YOUR\_USERNAME/Bot\_Detection\_GNN.git

cd Bot\_Detection\_GNN

```



\### Create Virtual Environment



```

python -m venv venv

```



\### Activate Environment



Windows:



```

venv\\Scripts\\activate

```



Mac / Linux:



```

source venv/bin/activate

```



\### Install Dependencies



```

pip install -r requirements.txt

```



\### Run Application



```

streamlit run app.py

```



\---



\## 📊 Example Network Metrics



The system computes:



\* Degree Distribution

\* Network Density

\* Clustering Coefficient

\* Connected Components

\* Bot Ratio



These metrics provide insights into:



\* Network connectivity

\* User clustering

\* Behavioral similarity

\* Bot concentration



\---



\## 🧪 Technologies Used



\* Python

\* Streamlit

\* PyTorch

\* PyTorch Geometric

\* NetworkX

\* Pandas

\* NumPy

\* Matplotlib

\* Scikit-learn



\---



\## 🎯 Applications



\* Social media bot detection

\* Fraud detection

\* Cybersecurity monitoring

\* Network anomaly detection

\* Behavioral analytics

\* Graph-based classification



\---



\## 📚 Academic Relevance



This project demonstrates practical applications of:



\* Graph Theory

\* Machine Learning

\* Graph Neural Networks

\* Network Analysis

\* Data Visualization



It is suitable for:



\* Minor / Major academic projects

\* Technical presentations

\* Portfolio projects

\* Research demonstrations



\---



\## 👨‍💻 Author



\*\*Varun Dubey\*\*



Machine Learning | Graph Neural Networks | AI Applications



GitHub:

https://github.com/Varundube99



\---



\## 📄 License



This project is intended for academic and educational use.



\---

