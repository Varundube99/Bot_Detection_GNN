\# 🤖 Bot Detection in Social Networks using Graph Neural Networks (GNN)



A complete \*\*Graph Neural Network (GNN) application\*\* for detecting automated bot accounts in a simulated social network using \*\*Graph Theory\*\* and \*\*Deep Learning\*\*.



The system models users as nodes and their behavioral similarity as edges, then applies a neural network to classify accounts as:



\* \*\*Bot\*\*

\* \*\*Human\*\*



The project includes an interactive \*\*Streamlit dashboard\*\* for visualization, prediction, and network analytics.



\---



\## 📋 Table of Contents



\* Overview

\* Key Features

\* Quick Start

\* How It Works

\* Graph Construction Methodology

\* Model Architecture

\* Application Pages

\* Project Structure

\* Technologies Used

\* Deployment

\* Limitations

\* Future Work

\* Troubleshooting

\* Author

\* License



\---



\# 📖 Overview



Social media platforms contain both genuine users and automated bot accounts. Bots can:



\* Spread misinformation

\* Manipulate engagement metrics

\* Generate spam content

\* Disrupt online ecosystems



Traditional machine learning models analyze users independently. However, real-world behavior is influenced by relationships between users.



This project addresses that limitation by:



\* Modeling the system as a \*\*graph\*\*

\* Using a \*\*Graph Neural Network (GNN)\*\*

\* Learning patterns from both behavior and relationships



\---



\# ✨ Key Features



🔗 Graph-based user relationship modeling using \*\*K-Nearest Neighbors (KNN)\*\*

🧠 Graph Neural Network (\*\*GCN\*\*) for bot classification

📊 Interactive network visualization

🔍 Real-time bot prediction interface

📈 Network analytics and graph metrics

🖥️ Multi-page Streamlit dashboard

⚡ Fast inference using PyTorch

🌐 Deployable machine learning application



\---



\# 🚀 Quick Start



\## Prerequisites



\* Python 3.8 or higher

\* pip package manager

\* Git



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/Bot\_Detection\_GNN.git

cd Bot\_Detection\_GNN

```



Create virtual environment:



```bash

python -m venv venv

```



Activate environment:



Windows:



```bash

venv\\Scripts\\activate

```



Mac / Linux:



```bash

source venv/bin/activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the application:



```bash

streamlit run app.py

```



Open browser:



```text

http://localhost:8501

```



\---



\# 🧠 How It Works



The system follows a graph-based machine learning workflow.



```text

User Data

&#x20;    ↓

Feature Engineering

&#x20;    ↓

Graph Construction

&#x20;    ↓

Graph Neural Network

&#x20;    ↓

Prediction

&#x20;    ↓

Visualization

```



The model learns not only from user behavior but also from relationships between users.



\---



\# 🧩 Graph Construction Methodology



Instead of using an existing social network, the graph was generated programmatically using \*\*Graph Theory\*\*.



\---



\## Step 1 — Node Representation



Each row in the dataset represents a user node.



Each user contains behavioral features:



\* Retweet Count

\* Mention Count

\* Follower Count

\* Verified Status

\* Tweet Frequency

\* Hashtag Count

\* Account Age

\* Engagement Ratio



These values form the:



```text

Node Feature Vector

```



\---



\## Step 2 — Feature Scaling



All numerical features were standardized using:



```text

StandardScaler (Scikit-learn)

```



This ensures:



\* Equal feature importance

\* Stable training

\* Faster convergence



\---



\## Step 3 — Graph Creation using KNN



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



Each user connects to their \*\*50 most similar users\*\*.



\---



\## Step 4 — Edge Generation



Connections between users were stored as:



```text

edges.csv

```



Format:



```text

source\_node, target\_node

```



Example:



```text

12, 45

12, 78

45, 103

```



\---



\## Step 5 — Graph Conversion for GNN



The graph was converted into a PyTorch Geometric format.



```text

x            → Node features

edge\_index   → Graph connections

y            → Labels

```



This enables:



```text

Message Passing between nodes

```



\---



\# 🤖 Model Architecture



Model Type:



```text

Graph Convolutional Network (GCN)

```



Architecture:



```text

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



```text

0 → Human

1 → Bot

```



\---



\# 🖥️ Application Pages



\## 📊 Network Graph



Displays:



\* User relationships

\* Node connections

\* Network structure

\* Bot / Human classification



Helps visualize how users interact in the network.



\---



\## 🔍 Check a User



Users can manually input account behavior.



Input:



\* Retweet Count

\* Mention Count

\* Follower Count

\* Verified Status

\* Tweet Frequency

\* Hashtag Count

\* Account Age



Output:



```text

Prediction: BOT / HUMAN

```



\---



\## 📈 Network Insights



Displays real-time graph analytics.



Metrics:



\* Total Users

\* Total Connections

\* Average Degree

\* Maximum Degree

\* Minimum Degree

\* Network Density

\* Connected Components

\* Clustering Coefficient

\* Bot Percentage



\---



\# 📁 Project Structure



```text

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



\# 🧪 Technologies Used



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



\# 🌐 Deployment



The application can be deployed using:



\* Streamlit Cloud

\* GitHub

\* Localhost



To deploy:



```bash

streamlit run app.py

```



Live deployment example:



```text

https://bot-detection-using-gnn.streamlit.app/

```



\---



\# ⚠️ Limitations



\* Synthetic dataset used for demonstration

\* Graph connections based on similarity rather than real social network data

\* Performance depends on feature quality

\* Large graphs require higher memory and computation

\* Model trained on structured data only



\---



\# 🚀 Future Work



\* Real-time social network integration

\* Community detection visualization

\* Graph filtering and search functionality

\* API-based prediction system

\* Larger dataset training

\* Real-time streaming analytics



\---



\# 🔧 Troubleshooting



Issue: Streamlit app not starting



Solution:



```bash

pip install -r requirements.txt

```



\---



Issue: Model not loading



Solution:



```bash

Check model file path

Ensure dependencies are installed

```



\---



Issue: Graph visualization slow



Solution:



```text

Reduce number of nodes

Use subset visualization

```



\---



\# 👨‍💻 Author



\*\*Varun Dubey\*\*



AI / Machine Learning Enthusiast



GitHub:

https://github.com/Varundube99



\---



\# 📄 License



This project is intended for academic and educational use.



Application code is open-source.

Model files are included for demonstration purposes.



