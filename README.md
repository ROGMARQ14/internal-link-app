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

```bash
# Clone the repository
git clone <repository-url>
cd internal-link-finder

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Deployment

The application can be deployed on Streamlit Cloud or any other platform that supports Streamlit applications.

## Requirements

See `requirements.txt` for a complete list of dependencies.
