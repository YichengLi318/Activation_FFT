import json
import os
import re
import requests
import arxiv
import time
from bs4 import BeautifulSoup

# Using simple word count as a proxy for token count.
TARGET_MIN_WORDS = 250
TARGET_MAX_WORDS = 450
NUM_ENTRIES = 100

def clean_gutenberg_text(text):
    """Removes the header and footer from a Project Gutenberg text."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    try:
        start_index = text.index(start_marker)
        # Find the end of the start marker line
        start_index = text.find('\n', start_index)
        if start_index != -1:
            # Move past the newline character
            start_index += 1
    except ValueError:
        start_index = 0

    try:
        end_index = text.rindex(end_marker)
    except ValueError:
        end_index = len(text)

    text = text[start_index:end_index].strip()
    # Remove chapter headings and other artifacts
    text = re.sub(r'\n[ \t]*Chapter [IVXLCDM\d]+.*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n[ \t]*[IVXLCDM\d]+.*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text) # Replace multiple spaces/newlines with a single space
    return text

def get_novel_corpus():
    print("Building novel corpus from Project Gutenberg...")
    corpus = []
    book_urls = {
        'https://www.gutenberg.org/files/2701/2701-0.txt': 'Moby Dick; or, The Whale',
        'https://www.gutenberg.org/files/1342/1342-0.txt': 'Pride and Prejudice',
        'https://www.gutenberg.org/files/84/84-0.txt': 'Frankenstein; or, The Modern Prometheus',
        'https://www.gutenberg.org/files/11/11-0.txt': "Alice's Adventures in Wonderland"
    }
    
    full_text = ""
    for url, title in book_urls.items():
        try:
            response = requests.get(url)
            response.encoding = response.apparent_encoding
            if response.status_code == 200:
                print(f"Successfully downloaded {title}")
                full_text += clean_gutenberg_text(response.text) + " "
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")
            continue

    # Split text into chunks of desired length
    words = [word for word in full_text.split() if word]
    
    i = 0
    while len(corpus) < NUM_ENTRIES and i < len(words):
        # Take a chunk of words
        chunk_size = (TARGET_MIN_WORDS + TARGET_MAX_WORDS) // 2
        chunk_words = words[i:i + chunk_size]
        content = " ".join(chunk_words)
        
        # Check if the chunk is substantial
        if len(chunk_words) >= TARGET_MIN_WORDS:
            corpus.append({
                "id": len(corpus),
                "category": "novel",
                "content": content,
                "source": "Project Gutenberg" # Source is generic as it's mixed
            })
        i += chunk_size

    print(f"Generated {len(corpus)} entries for novel corpus.")
    return corpus

def get_science_corpus():
    print("Building science corpus from arXiv...")
    corpus = []
    # Query for a mix of computer science, physics, and math papers
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.CL OR cat:math.CO OR cat:physics.gen-ph",
        max_results=300, # Fetch more to have enough to filter from
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=3
    )

    for result in client.results(search):
        if len(corpus) >= NUM_ENTRIES:
            break
        
        abstract = result.summary.replace('\n', ' ') # Clean newlines
        word_count = len(abstract.split())
        
        if TARGET_MIN_WORDS <= word_count <= TARGET_MAX_WORDS:
            corpus.append({
                "id": len(corpus),
                "category": "science",
                "content": abstract,
                "source": result.entry_id
            })

    print(f"Generated {len(corpus)} entries for science corpus.")
    return corpus

def main():
    output_dir = 'dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Novel Corpus ---
    novel_corpus = get_novel_corpus()
    if novel_corpus:
        novel_filepath = os.path.join(output_dir, 'novel.json')
        with open(novel_filepath, 'w', encoding='utf-8') as f:
            json.dump(novel_corpus, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved novel corpus to {novel_filepath}")
    else:
        print("Could not generate novel corpus.")

    # --- Science Corpus ---
    science_corpus = get_science_corpus()
    if science_corpus:
        science_filepath = os.path.join(output_dir, 'science.json')
        with open(science_filepath, 'w', encoding='utf-8') as f:
            json.dump(science_corpus, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved science corpus to {science_filepath}")
    else:
        print("Could not generate science corpus.")


if __name__ == '__main__':
    main()