import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time
import logging

class WebScraper:
    def __init__(self, base_url, output_directory="scraped_content"):
        """
        Initialize the web scraper with a base URL and output directory.
        
        Args:
            base_url (str): The starting URL to scrape
            output_directory (str): Directory to save scraped content
        """
        self.base_url = base_url
        self.output_directory = output_directory
        self.visited_urls = set()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
    def is_valid_url(self, url):
        """Check if URL belongs to the same domain as base_url."""
        return urlparse(self.base_url).netloc == urlparse(url).netloc
    
    def clean_text(self, text):
        """Clean extracted text by removing extra whitespace."""
        return ' '.join(text.split())
    
    def safe_filename(self, url):
        """Create a safe filename from URL."""
        return urlparse(url).path.replace('/', '_') or 'index'
    
    def scrape_page(self, url):
        """
        Scrape content from a single page.
        
        Args:
            url (str): URL to scrape
            
        Returns:
            tuple: (text content, list of links found)
        """
        try:
            # Add delay to be respectful to the server
            time.sleep(1)
            
            # Send request with headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Extract text content
            content = soup.get_text()
            content = self.clean_text(content)
            
            # Find all links
            links = []
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if self.is_valid_url(full_url):
                    links.append(full_url)
            
            return content, links
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return None, []
    
    def save_content(self, url, content):
        """Save scraped content to a file."""
        filename = os.path.join(self.output_directory, self.safe_filename(url) + '.txt')
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Source URL: {url}\n\n")
                f.write(content)
            logging.info(f"Saved content to {filename}")
        except Exception as e:
            logging.error(f"Error saving content from {url}: {str(e)}")
    
    def scrape_website(self, max_pages=10):
        """
        Scrape website starting from base_url.
        
        Args:
            max_pages (int): Maximum number of pages to scrape
        """
        urls_to_visit = [self.base_url]
        pages_scraped = 0
        
        while urls_to_visit and pages_scraped < max_pages:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            logging.info(f"Scraping {url}")
            content, links = self.scrape_page(url)
            
            if content:
                self.save_content(url, content)
                self.visited_urls.add(url)
                pages_scraped += 1
                
                # Add new links to visit
                urls_to_visit.extend([
                    link for link in links 
                    if link not in self.visited_urls 
                    and link not in urls_to_visit
                ])
            
            logging.info(f"Progress: {pages_scraped}/{max_pages} pages scraped")

def main():
    # Example usage
    website_url = "https://ebooks.inflibnet.ac.in/hrmp01/chapter/235/"  # Replace with your target website
    scraper = WebScraper(website_url)
    scraper.scrape_website(max_pages=5)  # Adjust max_pages as needed

if __name__ == "__main__":
    main()