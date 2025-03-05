"""
Internal Linking Opportunity Analyzer

This script analyzes a website's content and Google Search Console data to identify
internal linking opportunities based on keyword relevance.

Author: Cascade AI
Date: 2025-03-05
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
import spacy
import time
import csv
from io import StringIO
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# Load NLP model for semantic analysis
try:
    nlp = spacy.load("en_core_web_sm")
    nlp_available = True
    print("NLP model loaded successfully")
except:
    nlp_available = False
    print("NLP model not available. Will use basic text matching only.")
    print("To enable semantic analysis, run: !python -m spacy download en_core_web_sm")


class InternalLinkingAnalyzer:
    """Main class for analyzing internal linking opportunities."""

    def __init__(self):
        self.sitemap_urls = []
        self.gsc_data = None
        self.html_contents = {}
        self.main_contents = {}
        self.internal_linking_opportunities = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def extract_urls_from_sitemap(self, sitemap_content):
        """Extract URLs from an XML sitemap.
        
        Args:
            sitemap_content (str): XML sitemap content
            
        Returns:
            list: List of URLs from the sitemap
        """
        urls = []
        try:
            # Parse the XML content
            root = ET.fromstring(sitemap_content)
            
            # Extract URLs (handle different sitemap formats)
            # Standard sitemap format
            namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            for url in root.findall('.//ns:url/ns:loc', namespaces):
                urls.append(url.text.strip())
                
            # If no URLs found with namespace, try without namespace
            if not urls:
                for url in root.findall('.//loc'):
                    urls.append(url.text.strip())
                    
            print(f"Extracted {len(urls)} URLs from sitemap")
        except ET.ParseError as e:
            print(f"Error parsing sitemap XML: {e}")
        
        return urls
    
    def parse_gsc_data(self, gsc_content):
        """Parse Google Search Console data.
        
        Args:
            gsc_content (str): CSV content from GSC export
            
        Returns:
            DataFrame: Processed GSC data
        """
        try:
            df = pd.read_csv(StringIO(gsc_content))
            
            # Handle variations in GSC column names
            landing_page_cols = [col for col in df.columns if 'landing' in col.lower() or 'page' in col.lower() or 'url' in col.lower()]
            query_cols = [col for col in df.columns if 'query' in col.lower() or 'keyword' in col.lower()]
            
            if not landing_page_cols or not query_cols:
                print("Warning: Could not identify landing page or query columns in GSC data")
                print("Available columns:", df.columns.tolist())
                return None
            
            landing_page_col = landing_page_cols[0]
            query_col = query_cols[0]
            
            # Select relevant columns and rename for consistency
            gsc_df = df[[landing_page_col, query_col]].copy()
            gsc_df.columns = ['landing_page', 'query']
            
            # Clean URLs (remove protocol variations, trailing slashes)
            gsc_df['landing_page'] = gsc_df['landing_page'].apply(self._normalize_url)
            
            # Remove empty queries
            gsc_df = gsc_df[gsc_df['query'].notna() & (gsc_df['query'] != '')].reset_index(drop=True)
            
            print(f"Processed {len(gsc_df)} GSC data entries")
            return gsc_df
        
        except Exception as e:
            print(f"Error parsing GSC data: {e}")
            return None
    
    def _normalize_url(self, url):
        """Normalize URL to handle variations.
        
        Args:
            url (str): URL to normalize
            
        Returns:
            str: Normalized URL
        """
        if pd.isna(url):
            return ""
        
        parsed = urlparse(url)
        path = parsed.path
        if path.endswith('/'):
            path = path[:-1]
        
        # Reconstruct without protocol to handle http/https variations
        normalized = parsed.netloc + path
        return normalized.lower()
    
    def scrape_urls(self, max_urls=None, delay=1):
        """Scrape HTML content from URLs in the sitemap.
        
        Args:
            max_urls (int, optional): Maximum number of URLs to scrape. Defaults to None (all).
            delay (int, optional): Delay between requests in seconds. Defaults to 1.
        """
        if not self.sitemap_urls:
            print("No URLs to scrape. Please load sitemap first.")
            return
        
        urls_to_scrape = self.sitemap_urls
        if max_urls:
            urls_to_scrape = urls_to_scrape[:max_urls]
        
        print(f"Scraping {len(urls_to_scrape)} URLs...")
        
        for i, url in enumerate(urls_to_scrape):
            try:
                print(f"Scraping {i+1}/{len(urls_to_scrape)}: {url}")
                
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    self.html_contents[url] = response.text
                    # Extract main content immediately to free up memory
                    self.extract_main_content(url)
                else:
                    print(f"Failed to fetch {url}. Status code: {response.status_code}")
                
                # Add delay to avoid overwhelming the server
                if i < len(urls_to_scrape) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        
        print(f"Scraped {len(self.html_contents)} URLs successfully")
    
    def extract_main_content(self, url):
        """Extract main content from HTML, excluding headers, footers, and sidebars.
        
        Args:
            url (str): URL of the page
        """
        if url not in self.html_contents:
            print(f"HTML content for {url} not found")
            return
        
        html = self.html_contents[url]
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove common non-content elements
        for element in soup.select('header, footer, nav, .sidebar, .widget, .menu, .comments, .advertisement, .breadcrumbs, aside'):
            element.extract()
        
        # Try to find main content using common content containers
        content = None
        content_selectors = [
            'main', 'article', '.content', '.post-content', '.entry-content', 
            '#content', '#main', '.main-content', '.post-body', '.article-content'
        ]
        
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = content_element
                break
        
        # If no content container is found, use the body as fallback
        if not content:
            content = soup.body
            
        # Store main content
        if content:
            # Extract all paragraphs
            paragraphs = content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            self.main_contents[url] = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        else:
            self.main_contents[url] = []
        
        # Free up memory
        del self.html_contents[url]
    
    def find_internal_linking_opportunities(self):
        """Identify internal linking opportunities by matching GSC keywords with content."""
        if not self.gsc_data or not self.main_contents:
            print("Missing GSC data or webpage content. Please load both first.")
            return
        
        opportunities = []
        
        # Get all landing pages from GSC data
        all_landing_pages = self.gsc_data['landing_page'].unique()
        
        # For each URL in main contents
        for source_url, paragraphs in self.main_contents.items():
            source_url_normalized = self._normalize_url(source_url)
            
            # Skip self-referencing
            relevant_landing_pages = [page for page in all_landing_pages if self._normalize_url(page) != source_url_normalized]
            
            # Get relevant queries for this source URL
            source_queries = set(self.gsc_data[self.gsc_data['landing_page'] == source_url_normalized]['query'].str.lower())
            
            # For each potential target URL (landing page)
            for target_landing_page in relevant_landing_pages:
                # Get the keywords associated with this landing page
                target_queries = set(self.gsc_data[self.gsc_data['landing_page'] == target_landing_page]['query'].str.lower())
                
                # For each paragraph in the source content
                for paragraph in paragraphs:
                    paragraph_lower = paragraph.lower()
                    
                    # For each query associated with the target landing page
                    for query in target_queries:
                        if not query or len(query) < 3:  # Skip very short queries
                            continue
                            
                        # Check if the query appears in the paragraph but is not in the source URL's queries
                        # (to avoid linking to competing terms)
                        if query in paragraph_lower and query not in source_queries:
                            # Get the original case from the paragraph
                            query_pattern = re.compile(re.escape(query), re.IGNORECASE)
                            match = query_pattern.search(paragraph)
                            if match:
                                original_case_query = match.group(0)
                                
                                # Check if the query is already linked in the HTML
                                is_already_linked = self._check_if_already_linked(source_url, original_case_query)
                                
                                if not is_already_linked:
                                    # Get target URL from the sitemap URLs that matches the landing page
                                    target_urls = [url for url in self.sitemap_urls 
                                                   if self._normalize_url(url) == target_landing_page]
                                    
                                    if target_urls:
                                        target_url = target_urls[0]  # Use the first match
                                        
                                        # Add to opportunities
                                        opportunities.append({
                                            'source_url': source_url,
                                            'keyword': original_case_query,
                                            'text_snippet': paragraph,
                                            'target_url': target_url
                                        })
        
        # Deduplicate opportunities (same source, keyword, target)
        if opportunities:
            df = pd.DataFrame(opportunities)
            df = df.drop_duplicates(subset=['source_url', 'keyword', 'target_url'])
            
            # Prioritize opportunities
            df['keyword_length'] = df['keyword'].str.len()
            df = df.sort_values(['source_url', 'keyword_length'], ascending=[True, False])
            
            self.internal_linking_opportunities = df.to_dict('records')
            print(f"Found {len(self.internal_linking_opportunities)} internal linking opportunities")
        else:
            print("No internal linking opportunities found")
    
    def _check_if_already_linked(self, url, keyword):
        """Check if a keyword is already used as a link in the HTML.
        
        Args:
            url (str): URL to check
            keyword (str): Keyword to check for links
            
        Returns:
            bool: True if already linked, False otherwise
        """
        # Re-fetch the HTML since we don't store it anymore
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return False
                
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all anchor tags
            links = soup.find_all('a')
            
            # Check if the keyword is used as link text
            for link in links:
                link_text = link.get_text(strip=True)
                if keyword.lower() in link_text.lower():
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error checking links in {url}: {e}")
            return False
    
    def export_opportunities(self, output_filename='internal_linking_opportunities.csv'):
        """Export internal linking opportunities to CSV.
        
        Args:
            output_filename (str, optional): Output filename. Defaults to 'internal_linking_opportunities.csv'.
        """
        if not self.internal_linking_opportunities:
            print("No opportunities to export")
            return
            
        df = pd.DataFrame(self.internal_linking_opportunities)
        df.to_csv(output_filename, index=False)
        print(f"Exported {len(df)} opportunities to {output_filename}")
        
        # For Google Colab, provide a download link
        try:
            files.download(output_filename)
            print("Download initiated")
        except:
            print(f"File saved to {output_filename}. Please download manually.")

# Main execution function for Google Colab
def run_internal_linking_analysis():
    """Run the internal linking analysis process in Google Colab."""
    
    analyzer = InternalLinkingAnalyzer()
    
    print("=== Internal Linking Opportunity Analyzer ===")
    print("This tool analyzes your website content and GSC data to find internal linking opportunities.")
    print("\n1. Upload your XML sitemap file")
    
    uploaded = files.upload()
    sitemap_file = list(uploaded.keys())[0]
    sitemap_content = uploaded[sitemap_file].decode('utf-8')
    
    # Extract URLs from sitemap
    analyzer.sitemap_urls = analyzer.extract_urls_from_sitemap(sitemap_content)
    
    print("\n2. Upload your Google Search Console CSV export")
    uploaded = files.upload()
    gsc_file = list(uploaded.keys())[0]
    gsc_content = uploaded[gsc_file].decode('utf-8')
    
    # Parse GSC data
    analyzer.gsc_data = analyzer.parse_gsc_data(gsc_content)
    
    # Ask for scraping limit
    max_urls = int(input("\nEnter maximum number of URLs to scrape (leave blank for all): ") or "0")
    if max_urls <= 0:
        max_urls = None
    
    # Scrape the URLs
    analyzer.scrape_urls(max_urls=max_urls)
    
    # Find opportunities
    analyzer.find_internal_linking_opportunities()
    
    # Export results
    if analyzer.internal_linking_opportunities:
        output_filename = input("\nEnter output filename (default: internal_linking_opportunities.csv): ")
        if not output_filename:
            output_filename = "internal_linking_opportunities.csv"
        analyzer.export_opportunities(output_filename)
    
    print("\nAnalysis complete!")

# For direct execution in Colab
if __name__ == "__main__":
    run_internal_linking_analysis()
