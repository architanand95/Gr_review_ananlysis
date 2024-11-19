import streamlit as st
from G2lib import G2ProductFeatureList
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get the API token from environment variable
G2_API_TOKEN = os.getenv('API_KEY')

# Function to plot word cloud
def plot_wordcloud(top_features):
    feature_text = ' '.join([f"{feature['feature']} " * feature['importance'] for feature in top_features])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feature_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Main function
def main():
    st.title("G2 Product Feature Analysis")

    # Sidebar section
    analysis_type = st.selectbox("Select Analysis Type", ["Feature Analysis", "Geographic Analysis", "Time Series Analysis"]) 
    num_reviews = st.number_input("Number of Reviews", value=None, min_value=None, max_value=None, step=1, format="%d", help="Enter the number of reviews to analyze" , placeholder="e.g. 100 or leave empty all reviews")
    page_number = st.number_input("Page Number", value=None, min_value=None, max_value=None, step=1, format="%d", help="Enter the page number" , placeholder="e.g. 1 or leave empty for all reviews")

    country_name = None
    if analysis_type == "Geographic Analysis":
        country_name = st.text_input("Enter Country Name", value="", help="Enter the country name for geographic analysis" , placeholder="e.g. United States")

    rfc3339_date = None
    if analysis_type == "Time Series Analysis":
        rfc3339_date = st.text_input("Enter Date (RFC3339 format)", value="", help="Enter the date in RFC3339 format for time series analysis" , placeholder="e.g. 2021-09-01T00:00:00Z")

    show_features = st.checkbox("Show Features")

    # Button to run analysis
    if st.button("Run Analysis"):
        g2 = G2ProductFeatureList(G2_API_TOKEN)

        if analysis_type == "Feature Analysis":
            top_features = g2.run(show=show_features)
            plot_wordcloud(top_features)
            if show_features:
                df_features = pd.DataFrame(top_features)
                st.write("Top Features:")
                st.write(df_features)

        elif analysis_type == "Geographic Analysis":
            top_features = g2.geographic_features(country_name, num_of_reviews=num_reviews, page=page_number, show=show_features)
            plot_wordcloud(top_features)
            if show_features:
                df_features = pd.DataFrame(top_features)
                st.write("Top Features:")
                st.write(df_features)

        elif analysis_type == "Time Series Analysis":
            top_features = g2.time_features_extraction(rfc3339_date, num_of_reviews=num_reviews, page=page_number, show=show_features)
            plot_wordcloud(top_features)
            if show_features:
                df_features = pd.DataFrame(top_features)
                st.write("Top Features:")
                st.write(df_features)

        # Plot word cloud if show_features is checked
        

if __name__ == "__main__":
    main()
