# Internal Linking Opportunity Analyzer

This tool analyzes a website's content and Google Search Console data to identify internal linking opportunities based on keyword relevance.

## Overview

The Internal Linking Opportunity Analyzer helps website owners and SEO specialists improve their internal linking strategy by:

1. Analyzing website content from URLs in an XML sitemap
2. Processing Google Search Console (GSC) performance reports
3. Identifying keywords that appear in content but aren't currently linked
4. Generating a report of internal linking opportunities

## Requirements

- Python 3.6+
- Google Colab environment (recommended)
- Required packages:
  - requests
  - beautifulsoup4
  - pandas
  - spacy
  - numpy

## Installation

The easiest way to use this tool is through Google Colab. Upload the Python script and follow the instructions in the sample notebook.

To install the required packages manually:

```bash
pip install requests beautifulsoup4 pandas spacy
python -m spacy download en_core_web_sm
```

## Usage

### Option 1: Run in Google Colab with the provided notebook

1. Upload `internal_linking_analyzer.py` and `sample_usage.ipynb` to Google Colab
2. Run the notebook cells following the instructions

### Option 2: Run the script directly

```python
from internal_linking_analyzer import run_internal_linking_analysis

# Execute the main function
run_internal_linking_analysis()
```

## Input Files

### XML Sitemap

The tool accepts standard XML sitemaps following the sitemap protocol. Example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.example.com/</loc>
    <lastmod>2023-01-01</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <!-- more URLs -->
</urlset>
```

### Google Search Console Performance Report

Export your GSC performance report as a CSV file with the following columns:
- Landing Page (URL)
- Query (keyword)
- Additional metrics (clicks, impressions, etc.) are optional

## Output

The tool generates a CSV file with internal linking opportunities:

| source_url | keyword | text_snippet | target_url |
|------------|---------|--------------|------------|
| https://example.com/page1 | internal linking | This article discusses internal linking strategies | https://example.com/page2 |

## Features

- Extracts main content from web pages while excluding headers, footers, and sidebars
- Normalizes URLs to handle variations (HTTP/HTTPS, with/without trailing slashes)
- Checks if keywords are already linked to avoid duplicate suggestions
- Prioritizes longer keywords for better anchor text optimization
- Handles errors gracefully (404 pages, timeouts, etc.)
- Provides detailed progress information during processing

## Limitations

- The tool respects robots.txt and implements reasonable delays between requests to avoid overwhelming servers
- Processing large websites may take significant time
- The content extraction algorithm uses common patterns and may not work perfectly for all website layouts
- No semantic analysis for keyword variations (only exact matches)

## License

MIT License

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues.
