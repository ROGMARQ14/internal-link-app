import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Set page config
st.set_page_config(page_title="Context-Aware Internal Link Finder", page_icon="🔗", layout="wide")

st.title("Context-Aware Automatic Keyword Interlinker")
st.markdown("""
This app helps you find contextually relevant internal linking opportunities within your content.
Upload your Google Search Console data and content file to get started.
""")

# Download NLTK data if needed
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Load data at startup
download_nltk_data()

# Functions for processing
def extract_keywords(text, top_n=20):
    """Extract keywords using TF-IDF"""
    # Simple preprocessing
    text = text.lower()
    
    # Extract important words using TF-IDF
    try:
        vectorizer = TfidfVectorizer(max_features=top_n*2, 
                                     stop_words='english', 
                                     ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores for keywords
        dense = tfidf_matrix.todense()
        scores = dense.tolist()[0]
        scored_words = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        sorted_words = sorted(scored_words, key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, score in sorted_words[:top_n]]
    except Exception as e:
        st.warning(f"Error in keyword extraction: {e}")
        # Fallback to simple word extraction
        words = re.findall(r'\b\w{4,}\b', text.lower())
        # Count frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]

def generate_variations(keywords):
    """Generate simple variations of keywords"""
    variations = []
    
    # Start with the original keywords
    variations.extend(keywords)
    
    # Add simple variations
    for keyword in keywords:
        # Singular/plural variations
        if keyword.endswith('s'):
            variations.append(keyword[:-1])  # Remove 's'
        else:
            variations.append(keyword + 's')  # Add 's'
            
        # Add variations with common prefixes/suffixes
        if len(keyword) > 4:  # Only for longer words
            variations.append(keyword + 'ing')
            variations.append(keyword + 'ed')
        
        # Add hyphenated and non-hyphenated versions
        if '-' in keyword:
            variations.append(keyword.replace('-', ' '))
        elif ' ' in keyword:
            variations.append(keyword.replace(' ', '-'))
    
    # Remove duplicates
    return list(set(variations))

def calculate_similarity(text1, text2):
    """Calculate TF-IDF similarity between two texts"""
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        # Fallback to simpler method if vectorizer fails
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return round(similarity * 100, 2)

def extract_snippets(content, keyword, window_size=150):
    """Extract content snippets around keyword occurrences"""
    snippets = []
    content_lower = content.lower()
    keyword_lower = keyword.lower()
    
    # Find all occurrences of the keyword
    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
    matches = list(re.finditer(pattern, content_lower))
    
    for match in matches:
        start_pos = match.start()
        end_pos = match.end()
        
        # Determine snippet boundaries
        snippet_start = max(0, start_pos - window_size)
        snippet_end = min(len(content), end_pos + window_size)
        
        # Extract snippet
        snippet = content[snippet_start:snippet_end]
        
        # Add ellipsis if needed
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."
        
        # Highlight the keyword (preserve case)
        original_case_keyword = content[start_pos:end_pos]
        snippet = snippet.replace(original_case_keyword, f"**{original_case_keyword}**")
        
        snippets.append(snippet)
    
    return snippets

def check_existing_anchor(content, keyword, destination_url):
    """Check if there's already an anchor with the keyword to the destination"""
    pattern = rf'<a\s+[^>]*href=["\']([^"\']*)["\'][^>]*>{re.escape(keyword)}</a>'
    matches = re.finditer(pattern, content, re.IGNORECASE)
    
    for match in matches:
        href = match.group(1)
        if destination_url in href:
            return True
    
    return False

def suggest_content_modification(content, keyword):
    """Suggest new content to include the keyword"""
    # Tokenize content into sentences
    sentences = sent_tokenize(content)
    
    # Find the most relevant sentence for insertion
    best_sentence_idx = -1
    best_similarity = -1
    
    for i, sentence in enumerate(sentences):
        # Calculate simple word overlap
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        keyword_words = set(re.findall(r'\b\w+\b', keyword.lower()))
        
        if sentence_words and keyword_words:
            similarity = len(sentence_words.intersection(keyword_words)) / len(keyword_words)
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence_idx = i
    
    # If no good match found or keyword already in content
    if best_sentence_idx < 0 or keyword.lower() in content.lower():
        return f"The keyword '{keyword}' could be added to the content in a new paragraph or section that relates to this topic."
    
    # Generate a suggestion by modifying the best sentence
    original_sentence = sentences[best_sentence_idx]
    
    # Find a good insertion point (after a comma or period)
    insertion_points = [m.start() for m in re.finditer(r'[,.]', original_sentence)]
    if insertion_points:
        insertion_point = insertion_points[-1] + 1
        new_sentence = original_sentence[:insertion_point] + f" {keyword} is also relevant here." + original_sentence[insertion_point:]
    else:
        # If no good insertion point, append to the end
        new_sentence = original_sentence + f" {keyword} is also relevant in this context."
    
    return f"Original: {original_sentence}\nSuggested: {new_sentence}"

# UI Components
st.header("Upload Your Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Google Search Console Data")
    gsc_file = st.file_uploader("Upload GSC Performance Report (CSV or XLSX)", type=["csv", "xlsx"])

with col2:
    st.subheader("Content Data")
    content_file = st.file_uploader("Upload Content File (CSV or XLSX)", type=["csv", "xlsx"])

# Configuration parameters
st.header("Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    top_queries = st.number_input("Number of top queries per page", min_value=5, max_value=50, value=10)

with col2:
    similarity_threshold = st.slider("Similarity Threshold (%)", min_value=50, max_value=95, value=70)

with col3:
    max_suggestions = st.number_input("Maximum suggestions per page", min_value=1, max_value=50, value=5)

# Main processing
if st.button("Find Internal Linking Opportunities"):
    if gsc_file is None or content_file is None:
        st.error("Please upload both required files.")
    else:
        with st.spinner("Processing data and finding opportunities..."):
            # Load GSC data
            try:
                if gsc_file.name.endswith('.csv'):
                    gsc_data = pd.read_csv(gsc_file)
                else:
                    gsc_data = pd.read_excel(gsc_file)
            except Exception as e:
                st.error(f"Error loading GSC file: {e}")
                st.stop()
            
            # Load content data
            try:
                if content_file.name.endswith('.csv'):
                    content_data = pd.read_csv(content_file)
                else:
                    content_data = pd.read_excel(content_file)
            except Exception as e:
                st.error(f"Error loading content file: {e}")
                st.stop()
            
            # Check required columns
            required_gsc_cols = ['URL', 'Query', 'Clicks', 'Impressions']
            required_content_cols = ['URL', 'Content']
            
            missing_gsc_cols = [col for col in required_gsc_cols if col not in gsc_data.columns]
            missing_content_cols = [col for col in required_content_cols if col not in content_data.columns]
            
            if missing_gsc_cols or missing_content_cols:
                if missing_gsc_cols:
                    st.error(f"GSC file is missing required columns: {', '.join(missing_gsc_cols)}")
                if missing_content_cols:
                    st.error(f"Content file is missing required columns: {', '.join(missing_content_cols)}")
                st.stop()
                
            # Process data
            progress_bar = st.progress(0)
            
            # Get top queries per URL
            top_queries_per_url = {}
            for url in gsc_data['URL'].unique():
                page_queries = gsc_data[gsc_data['URL'] == url].sort_values(by='Clicks', ascending=False).head(top_queries)
                if not page_queries.empty:
                    top_queries_per_url[url] = page_queries['Query'].tolist()
            
            # Find linking opportunities
            results = []
            total_combinations = len(content_data) * (len(content_data) - 1)
            processed = 0
            
            # Create lookup dict for content
            content_dict = {}
            for _, row in content_data.iterrows():
                url = row['URL']
                content = str(row['Content'])
                if pd.notna(content) and content.strip():
                    content_dict[url] = content
            
            # For each source page
            for source_idx, source_row in content_data.iterrows():
                source_url = source_row['URL']
                
                # Skip if no content or queries
                if source_url not in content_dict or source_url not in top_queries_per_url:
                    continue
                    
                source_content = content_dict[source_url]
                source_queries = top_queries_per_url[source_url]
                
                # Extract keywords from content
                content_keywords = extract_keywords(source_content, top_n=top_queries)
                
                # Combine queries and content keywords
                all_keywords = list(set(source_queries + content_keywords))
                
                # Generate variations
                keyword_variations = generate_variations(all_keywords)
                
                # For each destination page
                for dest_idx, dest_row in content_data.iterrows():
                    if source_idx == dest_idx:  # Skip self-links
                        continue
                        
                    dest_url = dest_row['URL']
                    
                    # Skip if no content
                    if dest_url not in content_dict:
                        continue
                        
                    dest_content = content_dict[dest_url]
                    
                    # Calculate base similarity between pages
                    base_similarity = calculate_similarity(source_content[:5000], dest_content[:5000])
                    
                    # Only consider if above threshold
                    if base_similarity >= similarity_threshold:
                        # Check each potential anchor text
                        for keyword in keyword_variations:
                            # Skip very short keywords
                            if len(keyword) < 3:
                                continue
                                
                            # Only consider keywords that appear in source content
                            if keyword.lower() in source_content.lower():
                                # Check keyword relevance to destination
                                keyword_similarity = calculate_similarity(keyword, dest_content[:1000])
                                
                                if keyword_similarity >= similarity_threshold:
                                    # Extract snippets
                                    snippets = extract_snippets(source_content, keyword)
                                    
                                    # Check if anchor already exists
                                    has_existing_anchor = check_existing_anchor(source_content, keyword, dest_url)
                                    
                                    # Generate suggestion if no anchor or snippets
                                    content_suggestion = ""
                                    if not has_existing_anchor and not snippets:
                                        content_suggestion = suggest_content_modification(source_content, keyword)
                                    
                                    # Add to results if we have snippets or suggestions
                                    if snippets or content_suggestion:
                                        context = snippets[0] if snippets else "No direct match found in content."
                                        
                                        results.append({
                                            'Source URL': source_url,
                                            'Anchor Text': keyword,
                                            'Similarity Score': keyword_similarity,
                                            'Destination URL': dest_url,
                                            'Content Context': context,
                                            'Existing Anchor?': 'Yes' if has_existing_anchor else 'No',
                                            'New Content Suggestion': content_suggestion if not has_existing_anchor and not snippets else ""
                                        })
                    
                    # Update progress
                    processed += 1
                    progress_bar.progress(min(processed / total_combinations, 1.0))
            
            # Create results dataframe
            if results:
                df_results = pd.DataFrame(results)
                
                # Sort by similarity score
                df_results = df_results.sort_values(by='Similarity Score', ascending=False)
                
                # Limit suggestions per page
                page_counts = df_results['Source URL'].value_counts()
                filtered_results = []
                
                for _, row in df_results.iterrows():
                    source_url = row['Source URL']
                    if page_counts[source_url] <= max_suggestions:
                        filtered_results.append(row)
                    else:
                        page_counts[source_url] -= 1
                
                final_results = pd.DataFrame(filtered_results) if filtered_results else df_results
                
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

# Footer with info
st.markdown("---")
st.markdown("""
### About this Tool
This internal linking tool helps you discover contextually relevant linking opportunities within your content. 
It analyzes Google Search Console data to identify the most valuable keywords for each page, and then finds semantically 
related opportunities to link to relevant content elsewhere on your site.

For best results:
1. Make sure your GSC data includes URL, Query, Clicks, and Impressions columns
2. Ensure your content file includes the complete content of each page
3. Adjust the similarity threshold to find the right balance of suggestions
""")
