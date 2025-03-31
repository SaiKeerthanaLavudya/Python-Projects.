import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ------------------------------
# 🎯 DATA & MODEL FUNCTIONS
# ------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\ramch\Downloads\Tennis-Prediction-App-main\Tennis-Prediction-App-main\tennis.csv')
    return data

def preprocess_data(data):
    label_encoders = {}
    for column in ['outlook', 'temp', 'humidity', 'wind', 'play']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

def train_model(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

# ------------------------------
# 🏠 HOME PAGE
# ------------------------------

def home_page():
    st.title("🎾 **Play Tennis Prediction App** 🚀")

    st.markdown("""
    <div style="text-align: center;">
        <img src="https://images.pexels.com/photos/2339377/pexels-photo-2339377.jpeg?auto=compress&cs=tinysrgb&w=800" 
             width="100%" 
             style="border-radius: 15px;"/>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📌 **Introduction**")
    st.write("""
    Ever wondered if today is the perfect day for a game of tennis? 🎾  
    This app predicts whether you should **play tennis** based on **weather conditions** using **AI & Machine Learning**.  

    Simply go to the **Predict** page, enter weather details, and get instant insights! 💡
    """)

    st.markdown("---")

    st.subheader("🚀 **How It Works?**")
    st.write("""
    1️⃣ Navigate to the **Predict** page.  
    2️⃣ Enter the **weather conditions** (Outlook, Temperature, Humidity, Wind).  
    3️⃣ Click **Predict** and get an instant recommendation.  
    4️⃣ Enjoy playing tennis on the best days! 🎾  
    """)

    st.success("🏆 Ready? Head over to the **Predict** page now!")

# ------------------------------
# 🔮 PREDICTION PAGE
# ------------------------------

def predict_page():
    st.title("🎾 **Tennis Match Prediction** 🌦️")
    
    st.markdown("""
    <div style="text-align: center;">
        <img src="https://images.pexels.com/photos/1103829/pexels-photo-1103829.jpeg?auto=compress&cs=tinysrgb&w=800" 
             width="100%" 
             style="border-radius: 15px;"/>
    </div>
    """, unsafe_allow_html=True)

    data = load_data()
    data, label_encoders = preprocess_data(data)

    X = data.drop(['play', 'day'], axis=1)
    y = data['play']
    model = train_model(X, y)

    with st.form("predict_form", clear_on_submit=True):
        st.subheader("🌤️ **Enter Weather Conditions**")
        
        col1, col2 = st.columns(2)
        with col1:
            outlook = st.selectbox("🌞 Outlook", ["Sunny", "Overcast", "Rain"])
            temp = st.selectbox("🌡️ Temperature", ["Hot", "Mild", "Cool"])
        with col2:
            humidity = st.selectbox("💧 Humidity", ["High", "Normal"])
            wind = st.selectbox("💨 Wind", ["Weak", "Strong"])

        submit_button = st.form_submit_button("⚡ Predict", use_container_width=True)

    if submit_button:
        new_data = pd.DataFrame({
            'outlook': [outlook],
            'temp': [temp],
            'humidity': [humidity],
            'wind': [wind]
        })

        for column in new_data.columns:
            new_data[column] = label_encoders[column].transform(new_data[column])

        prediction = model.predict(new_data)
        result = label_encoders['play'].inverse_transform(prediction)[0]

        st.subheader("🧐 **Prediction Result**")
        if result == "Yes":
            st.success("🎾 Yes! You should **play tennis today**! Have fun! 🏸🎉")
        else:
            st.error("🚫 No! It's **not a great day** for tennis. Maybe try another time! ☁️")

# ------------------------------
# ℹ️ ABOUT PAGE
# ------------------------------

def about_page():
    st.title("ℹ️ **About This Project**")

    st.write("""
    This project is a **Machine Learning-powered app** that predicts if the weather is suitable for tennis.  
    It is powered by a **Decision Tree Model** and developed using **Streamlit** for an interactive UI.
    """)

    st.subheader("📊 **How the Model Works?**")
    st.write("""
    - The dataset contains **historical weather conditions**.  
    - It includes attributes like **Outlook, Temperature, Humidity, and Wind**.  
    - A **Decision Tree Algorithm** is trained on this dataset to learn patterns.  
    - The model predicts if it’s a **good day** to play tennis! ✅  
    """)

    st.markdown("---")

    st.subheader("👨‍💻 **About the Developer**")
    st.write("""
    This app is developed by **Ramavath Ramcharan Singh**, a passionate **Machine Learning & Data Science Enthusiast**.  
    Explore my other projects on GitHub and feel free to contribute! 🚀  
    """)

    st.markdown("""
    <div align="center">
        <a href="https://github.com/SaiKeerthanaLavudya" target="_blank">
            <img src="https://ghchart.rshah.org/00b4d8/SaiKeerthanaLavudya" alt="GitHub Heatmap" width="600px"/>
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📬 **Get in Touch!**")
    st.write("""
    - 🌎 **GitHub**: [github.com/SaiKeerthanaLavudya](https://github.com/SaiKeerthanaLavudya)  
    - ✉️ **Email**: work.saikeerthanalavudya@gmail.com 
    - 💬 Open for collaborations & discussions on AI & ML! 🚀  
    """)

# ------------------------------
# 🚀 MAIN APP
# ------------------------------

def main():
    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(145deg, #e0eafc, #cfdef3);
            }
            .stButton>button {
                background-color: #0077b6;
                color: #fff;
                border-radius: 10px;
                padding: 15px;
                transition: 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #0096c7;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation with Emojis
    nav = st.sidebar.radio("📌 **Navigation**", ["🏠 Home", "🔮 Predict", "ℹ️ About"])

    if nav == "🏠 Home":
        home_page()
    elif nav == "🔮 Predict":
        predict_page()
    elif nav == "ℹ️ About":
        about_page()

    st.markdown("---")
    st.markdown("<footer style='text-align: center;'>Developed by Ramavath Ramcharan Singh</footer>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
