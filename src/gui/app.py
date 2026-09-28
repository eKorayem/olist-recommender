# src/gui/app.py
import streamlit as st
import requests

API_URL = "http://backend:8000/recommend"
LOCAL_API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Olist Recommender", layout="wide")

st.title("🛒 E-Commerce Recommendation Engine")
st.markdown("Discover personalized products tailored to shopping patterns.")

st.sidebar.header("User Settings")
user_id = st.sidebar.text_input("Customer ID", placeholder="Enter customer unique ID...")
num_recs = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

if st.sidebar.button("Get Recommendations", type="primary"):
    if not user_id.strip():
        st.sidebar.error("Please enter a valid Customer ID.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                payload = {"user_id": user_id.strip(), "num_recommendations": num_recs}
                
                try:
                    res = requests.post(API_URL, json=payload, timeout=5)
                except requests.exceptions.ConnectionError:
                    res = requests.post(LOCAL_API_URL, json=payload, timeout=5)
                
                res.raise_for_status()
                data = res.json()
                
                if data["status"] == "cold_start_fallback":
                    st.info("🆕 **Cold Start User:** Showing top overall trending products.")
                else:
                    st.success("🎯 **Personalized Recommendations Ready!**")
                
                st.subheader(f"Top {num_recs} Recommended Products")
                
                # Display results in structured UI cards
                for i, item in enumerate(data["recommendations"], 1):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 7])
                        
                        col1.markdown(f"### #{i}")
                        col2.caption("CATEGORY")
                        col2.markdown(f"**🏷️ {item['category_name']}**")
                        
                        col3.caption("PRODUCT ID")
                        col3.code(item['product_id'])
                        
                        st.divider()

            except Exception as e:
                st.error(f"Error fetching recommendations: {e}")