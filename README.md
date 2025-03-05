# Context-Aware Automatic Keyword Interlinker

This Streamlit application helps you find contextually relevant internal linking opportunities within your content. It processes Google Search Console data and your website content to suggest where and how to add internal links based on semantic similarity.

## Features

- **Semantic Keyword Extraction**: Identifies the most relevant keywords from your content
- **Contextual Matching**: Finds linking opportunities that make sense in context
- **Smart Anchor Text Suggestions**: Recommends semantically appropriate anchor texts
- **Content Improvement Suggestions**: Provides recommendations for content updates when direct linking isn't possible
- **Similarity Scoring**: Ranks suggestions by semantic relevance

## Input Requirements

The application requires two input files:

1. **Google Search Console Performance Report (CSV/XLSX)** with columns:
   - URL: Landing page URLs
   - Query: Keywords/queries
   - Clicks: Number of clicks
   - Impressions: Number of impressions

2. **Content File (CSV/XLSX)** with columns:
   - URL: Landing page URLs
   - Content: The content of each page

## Output

The application generates a report with the following columns:

- **Source URL**: The landing page where the link will be placed
- **Anchor Text**: Suggested anchor text for the link
- **Similarity Score**: Semantic similarity score between the anchor text and destination page
- **Destination URL**: The landing page the link should point to
- **Content Context**: Content snippet showing where the link should be placed
- **Existing Anchor?**: Indicates if there's already a link
- **New Content Suggestion**: Suggests content modifications if no direct linking opportunity exists

## How to Use

1. Upload your Google Search Console performance report
2. Upload your content file
3. Configure parameters:
   - Number of top queries to consider per page
   - Similarity threshold for matching
   - Maximum suggestions per page
4. Click "Find Internal Linking Opportunities"
5. Review and download the results

## Installation

### Option 1: Full Installation (with all NLP features)
```bash
# Clone the repository
git clone <repository-url>
cd internal-link-finder

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 2: Simplified Installation (fewer dependencies)
```bash
# Clone the repository
git clone <repository-url>
cd internal-link-finder

# Install simplified dependencies
pip install -r simple_requirements.txt

# Run the simplified version
streamlit run streamlit_app.py
```

### Option 3: Minimal Installation (best for Python 3.12+ compatibility)
```bash
# Clone the repository
git clone <repository-url>
cd internal-link-finder

# Install minimal dependencies
pip install -r minimal_requirements.txt

# Run the minimal version
streamlit run minimal_app.py
```

If you encounter any issues with the full installation, try the simplified or minimal version which use fewer dependencies while maintaining core functionality. The minimal version is specifically designed to work with Python 3.12+ environments.

## Installation for Streamlit Deployment

For deploying to Streamlit Cloud, the full version should work properly with the following considerations:

1. **Handling spaCy on Streamlit Cloud**
   - The application now uses `spacy_streamlit` to manage proper installation of spaCy and its models
   - spaCy models are loaded directly from GitHub releases to ensure compatibility

2. **GPU Acceleration**
   - If CUDA is available, the application will automatically use GPU acceleration
   - If not, it will fall back to CPU processing with minimal performance loss

3. **Error Handling**
   - Robust error handling ensures the application works even if some models fail to load
   - Clear status messages show which models are available and functioning

4. **Memory Management**
   - The app processes text in chunks when needed to avoid memory issues with large texts
   - Long operations show progress indicators to keep users informed

## Deployment

The application can be deployed on Streamlit Cloud or any other platform that supports Streamlit applications.

## Troubleshooting

If you encounter issues with the application:

1. **spaCy Model Loading**
   - If you encounter errors with spaCy models, the application will attempt to download them automatically
   - You can manually install spaCy models with: `python -m spacy download en_core_web_sm`

2. **Memory Issues**
   - For processing very large files, increase the memory allocation in Streamlit settings
   - Break up very large content files into smaller chunks for better performance

3. **Package Compatibility**
   - If you encounter package compatibility issues, you can specify exact versions in requirements.txt
   - The current requirements are tested to work with Python 3.12+

## Requirements

See `requirements.txt` for a complete list of dependencies.
