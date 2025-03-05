import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import sys
import nltk
from nltk.tokenize import word_tokenize

# Set page configuration
st.set_page_config(page_title="Context-Aware Internal Link Finder", page_icon="🔗", layout="wide")

st.title("Context-Aware Automatic Keyword Interlinker")
st.markdown("""
This app helps you find contextually relevant internal linking opportunities within your content.
Upload your Google Search Console data and content file to get started.
""")

# Check for NLP libraries
with st.expander("Environment Setup"):
    st.write("Checking NLP libraries...")
    
    # Check and download NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        st.success("NLTK Punkt tokenizer is installed.")
    except LookupError:
        st.info("Downloading NLTK Punkt tokenizer...")
        nltk.download('punkt')
        st.success("NLTK Punkt tokenizer installed successfully.")
    
    try:
        nltk.data.find('corpora/stopwords')
        st.success("NLTK Stopwords are installed.")
    except LookupError:
        st.info("Downloading NLTK Stopwords...")
        nltk.download('stopwords')
        st.success("NLTK Stopwords installed successfully.")
    
    # Check for spaCy
    import importlib.util
    spacy_spec = importlib.util.find_spec("spacy")
    if spacy_spec is not None:
        st.success("spaCy is installed.")
        
        # Try to import spaCy
        try:
            import spacy
            # Check if a model is installed
            try:
                nlp = spacy.load("en_core_web_sm")
                st.success("spaCy model (en_core_web_sm) is installed.")
            except:
                st.warning("No spaCy model found.")
                if st.button("Install basic spaCy model"):
                    import subprocess
                    st.info("Installing spaCy model. This may take a minute...")
                    result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                                           capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("spaCy model installed successfully.")
                    else:
                        st.error(f"Failed to install spaCy model: {result.stderr}")
        except Exception as e:
            st.error(f"Error with spaCy: {e}")
    else:
        st.error("spaCy is not installed. Advanced NLP features will not be available.")

    # Check for sentence-transformers
    st_spec = importlib.util.find_spec("sentence_transformers")
    if st_spec is not None:
        st.success("sentence-transformers is installed.")
    else:
        st.warning("sentence-transformers is not installed. Semantic similarity will use a simpler method.")

# Main app functionality
st.header("Upload Your Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Google Search Console Data")
    gsc_file = st.file_uploader("Upload GSC Performance Report (CSV or XLSX)", type=["csv", "xlsx"])

with col2:
    st.subheader("Content Data")
    content_file = st.file_uploader("Upload Content File (CSV or XLSX)", type=["csv", "xlsx"])

# Display sample formats
with st.expander("See sample data formats"):
    st.subheader("GSC Performance Report Format")
    sample_gsc = pd.read_csv("sample_data_format.csv" if os.path.exists("sample_data_format.csv") 
                               else "https://raw.githubusercontent.com/username/repo/main/sample_data_format.csv")
    st.dataframe(sample_gsc)
    
    st.subheader("Content File Format")
    sample_content = pd.read_csv("sample_content_format.csv" if os.path.exists("sample_content_format.csv")
                                 else "https://raw.githubusercontent.com/username/repo/main/sample_content_format.csv")
    st.dataframe(sample_content)

# Simplified version
def find_internal_links(gsc_data, content_data, top_n=10, similarity_threshold=70):
    """Simplified version that works with basic libraries"""
    # Process GSC data for top queries
    results = []
    
    # Prepare TF-IDF for content similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Get all content documents
    all_content = content_data['Content'].fillna('').tolist()
    url_list = content_data['URL'].tolist()
    
    # Create document vectors
    try:
        tfidf_matrix = vectorizer.fit_transform(all_content)
    except Exception as e:
        st.error(f"Error processing content: {e}")
        return pd.DataFrame()
    
    # Get top queries per page
    top_queries = {}
    for url in gsc_data['URL'].unique():
        page_data = gsc_data[gsc_data['URL'] == url].sort_values('Clicks', ascending=False).head(top_n)
        if not page_data.empty:
            top_queries[url] = page_data['Query'].tolist()
    
    # Process each URL pair
    for i, source_row in content_data.iterrows():
        source_url = source_row['URL']
        source_content = str(source_row['Content'])
        source_idx = url_list.index(source_url)
        
        # Skip if no content
        if pd.isna(source_content) or not source_content.strip():
            continue
        
        # Get top queries for this URL
        queries = top_queries.get(source_url, [])
        if not queries:
            continue
        
        # For each destination URL
        for j, dest_row in content_data.iterrows():
            if i == j:  # Skip self
                continue
                
            dest_url = dest_row['URL']
            dest_content = str(dest_row['Content'])
            dest_idx = url_list.index(dest_url)
            
            # Skip if no content
            if pd.isna(dest_content) or not dest_content.strip():
                continue
            
            # Calculate similarity
            similarity = cosine_similarity(
                tfidf_matrix[source_idx:source_idx+1], 
                tfidf_matrix[dest_idx:dest_idx+1]
            )[0][0] * 100
            
            # Only process if similarity is above threshold
            if similarity >= similarity_threshold:
                # Check each query as potential anchor text
                for query in queries:
                    # Skip very short queries
                    if len(query) < 3:
                        continue
                        
                    # Check if query appears in source content
                    if query.lower() in source_content.lower():
                        # Extract context
                        query_pos = source_content.lower().find(query.lower())
                        start_pos = max(0, query_pos - 100)
                        end_pos = min(len(source_content), query_pos + len(query) + 100)
                        
                        context = source_content[start_pos:end_pos]
                        if start_pos > 0:
                            context = "..." + context
                        if end_pos < len(source_content):
                            context = context + "..."
                            
                        # Highlight the query in context
                        orig_query = source_content[query_pos:query_pos+len(query)]
                        context = context.replace(orig_query, f"**{orig_query}**")
                        
                        # Check if already linked
                        pattern = rf'<a\s+[^>]*href=["\']([^"\']*)["\'][^>]*>{re.escape(query)}</a>'
                        existing_anchor = bool(re.search(pattern, source_content, re.IGNORECASE))
                        
                        # Add to results
                        results.append({
                            'Source URL': source_url,
                            'Anchor Text': query,
                            'Similarity Score': round(similarity, 2),
                            'Destination URL': dest_url,
                            'Content Context': context,
                            'Existing Anchor?': 'Yes' if existing_anchor else 'No',
                            'New Content Suggestion': "" if existing_anchor else f"Consider adding a link to {dest_url} using '{query}' as anchor text."
                        })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# Parameters
st.header("Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    top_queries = st.number_input("Number of top queries per page", min_value=5, max_value=50, value=10)

with col2:
    similarity_threshold = st.slider("Similarity Threshold (%)", min_value=50, max_value=95, value=70)

with col3:
    max_suggestions = st.number_input("Maximum suggestions per page", min_value=1, max_value=50, value=5)

# Process data
if st.button("Find Internal Linking Opportunities"):
    if gsc_file is None or content_file is None:
        st.error("Please upload both required files.")
    else:
        with st.spinner("Processing data and finding opportunities..."):
            # Load GSC data
            if gsc_file.name.endswith('.csv'):
                gsc_data = pd.read_csv(gsc_file)
            else:
                gsc_data = pd.read_excel(gsc_file)
            
            # Load content data
            if content_file.name.endswith('.csv'):
                content_data = pd.read_csv(content_file)
            else:
                content_data = pd.read_excel(content_file)
            
            # Check if required columns exist
            required_gsc_cols = ['URL', 'Query', 'Clicks', 'Impressions']
            required_content_cols = ['URL', 'Content']
            
            missing_gsc_cols = [col for col in required_gsc_cols if col not in gsc_data.columns]
            missing_content_cols = [col for col in required_content_cols if col not in content_data.columns]
            
            if missing_gsc_cols or missing_content_cols:
                if missing_gsc_cols:
                    st.error(f"GSC file is missing required columns: {', '.join(missing_gsc_cols)}")
                if missing_content_cols:
                    st.error(f"Content file is missing required columns: {', '.join(missing_content_cols)}")
            else:
                # Find internal linking opportunities
                results = find_internal_links(
                    gsc_data, 
                    content_data, 
                    top_n=top_queries,
                    similarity_threshold=similarity_threshold
                )
                
                if not results.empty:
                    # Sort and filter results
                    results = results.sort_values('Similarity Score', ascending=False)
                    
                    # Limit suggestions per page
                    page_counts = results['Source URL'].value_counts()
                    filtered_results = []
                    
                    for _, row in results.iterrows():
                        source_url = row['Source URL']
                        if page_counts[source_url] <= max_suggestions:
                            filtered_results.append(row)
                        else:
                            page_counts[source_url] -= 1
                    
                    final_results = pd.DataFrame(filtered_results) if filtered_results else results
                    
                    # Display results
                    st.header("Internal Linking Opportunities")
                    st.dataframe(final_results, use_container_width=True)
                    
                    # Download button
                    csv = final_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="internal_linking_opportunities.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No linking opportunities found with the current settings. Try adjusting the similarity threshold.")

# Footer
st.markdown("---")
st.markdown("Context-Aware Automatic Keyword Interlinker - Simplified Version")
