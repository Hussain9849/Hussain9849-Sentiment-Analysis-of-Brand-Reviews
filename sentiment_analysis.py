import pandas as pd
import streamlit as st
from textblob import TextBlob
import cleantext
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import random
import json
# Load external CSS file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# SIDE NAVBAR
st.sidebar.title('Sentiment Analysis of Brand Reviews')
option = st.sidebar.selectbox('Choose a section', ['Home','Analyze E-commerce Reviews', 'Analyze CSV', 'Analyze Individual Review'])
 
df = None

# Home Page
if option == 'Home':
    st.markdown("""
        <div style="text-align: center;">
            <h2>Analyze Customer Feedback with Ease</h2>
            <p>Our tool helps you understand customer sentiments by analyzing reviews and providing actionable insights.</p>
            <img src="https://blog.brandbastion.com/hs-fs/hubfs/sentiment-analysis-comments-examples.png?width=1130&height=644&name=sentiment-analysis-comments-examples.png" alt="Sentiment Analysis" style="width: 100%; border-radius: 10px; margin-top: 20px; animation: fadeIn 2s ease-in-out;">
        </div>
    """, unsafe_allow_html=True)     
     
elif option == 'Analyze Individual Review':
    st.header('Analyze Individual Review')
    
    # Explanation of Sentiment Analysis
    st.write("""
    Sentiment analysis helps determine the emotional tone of a piece of text. 
    It provides two key metrics:
    - **Polarity**: Indicates whether the sentiment is positive, negative, or neutral.
    - **Subjectivity**: Indicates whether the text is more opinion-based or fact-based.
    """)

    # Input text for analysis
    text = st.text_input('Enter text to analyze:')
    
    if text:
        # Perform sentiment analysis
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 2)
        subjectivity = round(blob.sentiment.subjectivity, 2)
        
        # Determine polarity description and color
        if polarity > 0:
            polarity_description = "(Positive Sentiment)"
            polarity_color = "green"
        elif polarity < 0:
            polarity_description = "(Negative Sentiment)"
            polarity_color = "red"
        else:
            polarity_description = "(Neutral Sentiment)"
            polarity_color = "blue"
        
        # Determine subjectivity description and color
        if subjectivity > 0.5:
            subjectivity_description = "(Review contains more personal opinion)"
            subjectivity_color = "orange"
        else:
            subjectivity_description = "(Review contains more factual information)"
            subjectivity_color = "green"
        
        # Display polarity and subjectivity scores with descriptions and colors
        st.markdown(
            f"""
            <div style="font-size:18px;">
                <b>Polarity:</b> {polarity} <span style="color:{polarity_color};">{polarity_description}</span><br>
                <b>Subjectivity:</b> {subjectivity} <span style="color:{subjectivity_color};">{subjectivity_description}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # Add a horizontal line or spacing before the next section
    st.markdown("<hr style='border: 1px solid #ddd; margin: 20px 0;'>", unsafe_allow_html=True)  # Thin horizontal line
    st.write("")  # Add extra spacing    
        
    # Clean Section
    st.subheader("Text Preprocessing and Word Cloud")
    st.write("""
    Cleaning your text removes unnecessary elements like punctuation, numbers, and stopwords, 
    making it ready for analysis. This step is optional but can improve the accuracy of sentiment analysis.
    """)

    # Input text for cleaning
    pre = st.text_area('Enter text to clean and analyze:', placeholder="Paste your text here...")

    if pre:
        # Clean the text
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                    stopwords=True, lowercase=True, numbers=True, punct=True)
        
        # Display cleaned text
        st.write('**Cleaned Text:**')
        st.success(cleaned_text)
        
        # Word Cloud
        st.subheader("Word Cloud")
        st.write("A visual representation of the most frequently used words in the cleaned text.")

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# Analyze CSV Section
elif option == 'Analyze CSV':
    st.header('Analyze CSV')
    st.write("Upload your own file or use the sample data provided below for instant analysis.")
    
    # Data selection menu
    data_option = st.selectbox('Select Data Source', ['Sample Data', 'Upload File'])
    
    if data_option == 'Sample Data':
        sample_data = {
            'reviews': [
                "The quality of the product is excellent.",
                "The price is too high for the value.",
                "Customer service was very helpful.",
                "The product broke after a week of use.",
                "I love the design and the quality is top-notch.",
                "The price is reasonable and affordable.",
                "Customer service was rude and unhelpful."
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        st.write("Sample Data:")
        st.dataframe(sample_df)
        if st.button('Analyze Sample Data'):
            df = sample_df.copy()
    elif data_option == 'Upload File':
        upl = st.file_uploader('Upload file', type=['xlsx', 'xls'])
        if upl:
            df = pd.read_excel(upl, engine='openpyxl')
            if 'Unnamed: 0' in df.columns:
                del df['Unnamed: 0']
            st.write("Uploaded Data:")
            st.dataframe(df)
            if st.button('Analyze Uploaded Data'):
                df = df.copy()

    if df is not None:
        def score(x):
            blob1 = TextBlob(x)
            return blob1.sentiment.polarity

        def subjectivity(x):
            blob1 = TextBlob(x)
            return blob1.sentiment.subjectivity

        def analyze(x):
            if x <= 0:
                return 'Negative'
            elif x == 0:
                return 'Neutral'
            else:
                return 'Positive'

        if 'reviews' in df.columns:
            df['score'] = df['reviews'].apply(score)
            df['subjectivity'] = df['reviews'].apply(subjectivity)
            df['analysis'] = df['score'].apply(analyze)

            # Interactive Filters
            sentiment_filter = st.selectbox('Select Sentiment', ['All', 'Positive', 'Negative', 'Neutral'])
            aspect_filter = st.selectbox('Select Aspect', ['All', 'Quality', 'Price', 'Customer Service'])

            filtered_df = df.copy()
            if sentiment_filter != 'All':
                filtered_df = filtered_df[filtered_df['analysis'] == sentiment_filter]
            if aspect_filter != 'All':
                filtered_df = filtered_df[filtered_df['reviews'].str.contains(aspect_filter.lower())]

            # Highlight negative reviews
            def highlight_negative(s):
                return ['background-color: red' if v == 'Negative' else '' for v in s]

            st.write(filtered_df.style.apply(highlight_negative, subset=['analysis']).to_html(), unsafe_allow_html=True)

            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(filtered_df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )

            # Polarity Distribution Section
            st.subheader('Polarity Distribution')
            if df is not None and 'score' in df.columns:
                df['category'] = df['score'].apply(lambda x: 'Positive' if x > 0.5 else 'Negative' if x < 0 else 'Neutral')
                
                # Count the number of positive, negative, and neutral reviews
                positive_count = df[df['category'] == 'Positive'].shape[0]
                negative_count = df[df['category'] == 'Negative'].shape[0]
                neutral_count = df[df['category'] == 'Neutral'].shape[0]
                
                # Create a histogram with annotations
                fig = px.histogram(df, x='score', color='category', nbins=20, title='Polarity Distribution',
                                   color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
                
                # Add annotations for counts
                fig.update_layout(
                    annotations=[
                        dict(
                            x=0.75, y=positive_count, text=f"Positive: {positive_count}", showarrow=False, font=dict(color="green")),
                        dict(
                            x=-0.75, y=negative_count, text=f"Negative: {negative_count}", showarrow=False, font=dict(color="red")),
                        dict(
                            x=0, y=neutral_count, text=f"Neutral: {neutral_count}", showarrow=False, font=dict(color="blue"))
                    ]
                )
                
                st.plotly_chart(fig)
            else:
                st.info("Please upload a CSV file with a 'reviews' column to see the polarity distribution.")

            # Aspect-Based Sentiment Analysis Section
            st.subheader('Aspect-Based Sentiment Analysis')
            if df is not None and 'reviews' in df.columns:
                aspects = ['quality', 'price', 'customer service']
                aspect_sentiments = {aspect: {'Positive': 0, 'Negative': 0, 'Neutral': 0} for aspect in aspects}

                for review in df['reviews']:
                    for aspect in aspects:
                        if aspect in review.lower():
                            blob = TextBlob(review)
                            polarity = blob.sentiment.polarity
                            if polarity > 0.5:
                                aspect_sentiments[aspect]['Positive'] += 1
                            elif polarity <= 0:
                                aspect_sentiments[aspect]['Negative'] += 1
                            else:
                                aspect_sentiments[aspect]['Neutral'] += 1

                st.write("Aspect-Based Sentiment Analysis Results:")

                # Create subplots
                fig = make_subplots(rows=1, cols=len(aspects), subplot_titles=[aspect.capitalize() for aspect in aspects],
                                    specs=[[{'type': 'domain'}] * len(aspects)])

                for idx, (aspect, sentiments) in enumerate(aspect_sentiments.items()):
                    fig.add_trace(go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                                         values=[sentiments['Positive'], sentiments['Negative'], sentiments['Neutral']],
                                         marker=dict(colors=['green', 'red', 'blue']),
                                         name=aspect.capitalize(), textinfo='percent+label', textposition='inside'), 1, idx + 1)

                # Update layout for the combined figure
                fig.update_layout(height=400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please upload a CSV file with a 'reviews' column to see the aspect-based sentiment analysis.")
             # Insights Section
            st.subheader('Insights and Recommendations')
            total_reviews = df.shape[0]
            negative_count = df[df['analysis'] == 'Negative'].shape[0]
            if negative_count > total_reviews / 2:
                st.warning("There are more negative reviews than positive and neutral reviews.")
                st.write("Based on the analysis, here are some recommendations to improve brand perception:")
                insights = [
                    {
                        "icon": "🔍",
                        "title": "Investigate Issues",
                        "description": "Investigate common issues mentioned in negative reviews."
                    },
                    {
                        "icon": "🔧",
                        "title": "Enhance Quality",
                        "description": "Enhance product quality and features."
                    },
                    {
                        "icon": "🤝",
                        "title": "Improve Service",
                        "description": "Improve customer service and support."
                    },
                    {
                        "icon": "📢",
                        "title": "Address Feedback",
                        "description": "Address specific complaints and feedback from customers."
                    },
                    {
                        "icon": "📊",
                        "title": "Monitor Trends",
                        "description": "Regularly monitor sentiment trends to identify areas for improvement."
                    },
                    {
                        "icon": "💡",
                        "title": "Innovate",
                        "description": "Introduce new features or products based on customer feedback."
                    }
                ]
                for insight in insights:
                    st.markdown(
                        f"""
                        <div class="suggestion-card">
                            <h4>{insight['icon']} {insight['title']}</h4>
                            <p>{insight['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.success("The sentiment analysis shows a balanced or positive sentiment towards the brand.")
                st.write("Keep up the good work! Here are some recommendations to maintain and further improve brand perception:")
                positive_insights = [
                    {
                        "icon": "👍",
                        "title": "Maintain Quality",
                        "description": "Continue to maintain high product quality."
                    },
                    {
                        "icon": "😊",
                        "title": "Customer Satisfaction",
                        "description": "Ensure customer satisfaction remains a top priority."
                    },
                    {
                        "icon": "📈",
                        "title": "Expand Offerings",
                        "description": "Consider expanding product offerings based on positive feedback."
                    },
                    {
                        "icon": "💬",
                        "title": "Engage Customers",
                        "description": "Engage with customers through social media and other channels."
                    }
                ]
                for insight in positive_insights:
                    st.markdown(
                        f"""
                        <div class="suggestion-card">
                            <h4>{insight['icon']} {insight['title']}</h4>
                            <p>{insight['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.error("The uploaded file does not contain a 'reviews' column.")

# Analyze E-commerce Reviews Section
elif option == 'Analyze E-commerce Reviews':
    st.header('Analyze E-commerce Reviews')
    
    try:
        with open('products.json', 'r') as f:
            products = json.load(f)
    except FileNotFoundError:
        st.error("products.json file not found. Please ensure it exists in the project directory.")
        products = []

    # Define available e-commerce websites
    ecommerce_websites = ['Fake Store', 'Amazon(Future Enhancement)']

    # Sidebar: Choose E-commerce Website
    selected_website = st.sidebar.selectbox(
        'Choose E-commerce Website:',
        ['Select Website'] + ecommerce_websites  # Add "Select Website" option
    )

    if selected_website != 'Select Website':
        # Filter products by selected website
        filtered_products = [product for product in products if product.get('website', '').lower() == selected_website.lower()]
    # Extract product names and IDs for suggestions
        product_names = [product['name'] for product in products]
        product_ids = [product['id'] for product in products]
        
        # Initialize session state for the search query
        if "search_query" not in st.session_state:
            st.session_state["search_query"] = ""
            
        # Suggest product names dynamically as the user types (only for partial matches)
        if st.session_state["search_query"]:
            matching_products = [
                name for name in product_names
                if st.session_state["search_query"].lower() in name.lower() and st.session_state["search_query"].lower() != name.lower()
            ]
            if matching_products:
                st.sidebar.write("Suggestions:")
                for idx, product in enumerate(matching_products):
                    # Create clickable suggestions with unique keys
                    if st.sidebar.button(product, key=f"suggestion_{idx}"):
                        # Update the session state before rendering the search bar
                        st.session_state["search_query"] = product

        # Search bar with suggestions
        search_query = st.sidebar.text_input(
            'Search by Product ID or Name (Start typing for suggestions):',
            placeholder='Enter Product ID or Name',
            key="search_query"
        )

        if st.sidebar.button('Fetch Reviews'):
            try:
                # Check if the search query matches a product ID or Name
                product = next(
                    (item for item in products if item["id"] == search_query or item["name"].lower() == search_query.lower()),
                    None
                )
                if product:
                    product_name = product.get('name', 'Unknown Product')  # Fetch product name
                    reviews = product.get('reviews', [])
                    if reviews:
                        st.write(f"**Product Name:** {product_name}")  # Display product name
                        st.write("Fetched Reviews:")
                        df = pd.DataFrame(reviews, columns=['review'])
                        st.dataframe(df)
                    else:
                        st.error("No reviews found for this product.")
                else:
                    st.error("Product not found. Please check the product ID or name and try again.")
            except FileNotFoundError:
                st.error("products.json file not found. Please ensure it exists in the project directory.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

        if df is not None:

            def score(x):
                blob1 = TextBlob(x)
                return blob1.sentiment.polarity

            def subjectivity(x):
                blob1 = TextBlob(x)
                return blob1.sentiment.subjectivity

            def analyze(x):
                if x < 0:
                    return 'Negative'
                elif x == 0:
                    return 'Neutral'
                else:
                    return 'Positive'

            if 'review' in df.columns:
                df['score'] = df['review'].apply(score)
                df['subjectivity'] = df['review'].apply(subjectivity)
                df['analysis'] = df['score'].apply(analyze)

                # Interactive Filters
                sentiment_filter = st.selectbox('Select Sentiment', ['All', 'Positive', 'Negative', 'Neutral'])
                aspect_filter = st.selectbox('Select Aspect', ['All', 'Quality', 'Price', 'Customer Service'])

                filtered_df = df.copy()
                if sentiment_filter != 'All':
                    filtered_df = filtered_df[filtered_df['analysis'] == sentiment_filter]
                if aspect_filter != 'All':
                    filtered_df = filtered_df[filtered_df['review'].str.contains(aspect_filter.lower())]

                
                # Highlight negative reviews
                def highlight_negative(s):
                    return ['background-color: red' if v == 'Negative' else '' for v in s]

                st.write(filtered_df.style.apply(highlight_negative, subset=['analysis']).to_html(), unsafe_allow_html=True)

                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(filtered_df)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='sentiment.csv',
                    mime='text/csv',
                )

                # Polarity Distribution Section
                st.subheader('Polarity Distribution')
                if df is not None and 'score' in df.columns:
                    df['category'] = df['score'].apply(lambda x: 'Positive' if x > 0.5 else 'Negative' if x < 0 else 'Neutral')
                    
                    # Count the number of positive, negative, and neutral reviews
                    positive_count = df[df['category'] == 'Positive'].shape[0]
                    negative_count = df[df['category'] == 'Negative'].shape[0]
                    neutral_count = df[df['category'] == 'Neutral'].shape[0]
                    
                    # Create a histogram with annotations
                    fig = px.histogram(df, x='score', color='category', nbins=20, title='Polarity Distribution',
                                    color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
                    
                    # Add annotations for counts
                    fig.update_layout(
                        annotations=[
                            dict(
                                x=0.75, y=positive_count, text=f"Positive: {positive_count}", showarrow=False, font=dict(color="green")),
                            dict(
                                x=-0.75, y=negative_count, text=f"Negative: {negative_count}", showarrow=False, font=dict(color="red")),
                            dict(
                                x=0, y=neutral_count, text=f"Neutral: {neutral_count}", showarrow=False, font=dict(color="blue"))
                        ]
                    )
                    
                    st.plotly_chart(fig)
                else:
                    st.info("Please upload a CSV file with a 'reviews' column to see the polarity distribution.")

                # Aspect-Based Sentiment Analysis Section
                st.subheader('Aspect-Based Sentiment Analysis')
                if df is not None and 'review' in df.columns:
                    aspects = ['quality', 'price', 'customer service']
                    aspect_sentiments = {aspect: {'Positive': 0, 'Negative': 0, 'Neutral': 0} for aspect in aspects}

                    for review in df['review']:
                        for aspect in aspects:
                            if aspect in review.lower():
                                blob = TextBlob(review)
                                polarity = blob.sentiment.polarity
                                if polarity > 0.5:
                                    aspect_sentiments[aspect]['Positive'] += 1
                                elif polarity <= 0:
                                    aspect_sentiments[aspect]['Negative'] += 1
                                else:
                                    aspect_sentiments[aspect]['Neutral'] += 1

                    st.write("Aspect-Based Sentiment Analysis Results:")

                    # Create subplots
                    fig = make_subplots(rows=1, cols=len(aspects), subplot_titles=[aspect.capitalize() for aspect in aspects],
                                        specs=[[{'type': 'domain'}] * len(aspects)])

                    for idx, (aspect, sentiments) in enumerate(aspect_sentiments.items()):
                        fig.add_trace(go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                                            values=[sentiments['Positive'], sentiments['Negative'], sentiments['Neutral']],
                                            marker=dict(colors=['green', 'red', 'blue']),
                                            name=aspect.capitalize(), textinfo='percent+label', textposition='inside'), 1, idx + 1)

                    # Update layout for the combined figure
                    fig.update_layout(height=400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please upload a CSV file with a 'reviews' column to see the aspect-based sentiment analysis.")

                # Insights Section
                st.subheader('Insights and Recommendations')
                total_reviews = df.shape[0]
                negative_count = df[df['analysis'] == 'Negative'].shape[0]
                if negative_count > total_reviews / 2:
                    st.warning("There are more negative reviews than positive and neutral reviews.")
                    st.write("Based on the analysis, here are some recommendations to improve brand perception:")
                    insights = [
                        {
                            "icon": "🔍",
                            "title": "Investigate Issues",
                            "description": "Investigate common issues mentioned in negative reviews."
                        },
                        {
                            "icon": "🔧",
                            "title": "Enhance Quality",
                            "description": "Enhance product quality and features."
                        },
                        {
                            "icon": "🤝",
                            "title": "Improve Service",
                            "description": "Improve customer service and support."
                        },
                        {
                            "icon": "📢",
                            "title": "Address Feedback",
                            "description": "Address specific complaints and feedback from customers."
                        },
                        {
                            "icon": "📊",
                            "title": "Monitor Trends",
                            "description": "Regularly monitor sentiment trends to identify areas for improvement."
                        },
                        {
                            "icon": "💡",
                            "title": "Innovate",
                            "description": "Introduce new features or products based on customer feedback."
                        }
                    ]
                    for insight in insights:
                        st.markdown(
                            f"""
                            <div class="suggestion-card">
                                <h4>{insight['icon']} {insight['title']}</h4>
                                <p>{insight['description']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.success("The sentiment analysis shows a balanced or positive sentiment towards the brand.")
                    st.write("Keep up the good work! Here are some recommendations to maintain and further improve brand perception:")
                    positive_insights = [
                        {
                            "icon": "👍",
                            "title": "Maintain Quality",
                            "description": "Continue to maintain high product quality."
                        },
                        {
                            "icon": "😊",
                            "title": "Customer Satisfaction",
                            "description": "Ensure customer satisfaction remains a top priority."
                        },
                        {
                            "icon": "📈",
                            "title": "Expand Offerings",
                            "description": "Consider expanding product offerings based on positive feedback."
                        },
                        {
                            "icon": "💬",
                            "title": "Engage Customers",
                            "description": "Engage with customers through social media and other channels."
                        }
                    ]
                    for insight in positive_insights:
                        st.markdown(
                            f"""
                            <div class="suggestion-card">
                                <h4>{insight['icon']} {insight['title']}</h4>
                                <p>{insight['description']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.error("The uploaded file does not contain a 'reviews' column.")
    else:
        st.info("Please select an e-commerce website to proceed.")   