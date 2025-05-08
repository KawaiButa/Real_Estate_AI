import csv
import time
import json
import os
import random
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_PATH = "data/airbnb/listings.csv"
IMG_DIR = "images"
JSON_OUT = "listings.json"
MAX_WORKERS = 10  

os.makedirs(IMG_DIR, exist_ok=True)


def extract_image_urls_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    pattern = re.compile(r'"(https://a0.muscache.com/im/pictures/[^"]+)"')

    links = []

    for script in soup.find_all("script"):
        if script.string:
            matches = pattern.findall(
                script.string
            )  # .findall returns only the captured group
            links.extend(matches)
    return links[:10] if len(links) > 10 else links


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
]


def fetch_listing_images(listing_url: str) -> list[str]:
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        time.sleep(random.uniform(1.0, 3.0))
        resp = requests.get(listing_url, headers=headers, timeout=10)
        resp.raise_for_status()
        return extract_image_urls_from_html(resp.text)
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            wait = random.uniform(5, 10)
            print(f"⚡ 429 Too Many Requests. Sleeping {wait:.1f}s...")
            time.sleep(wait)
        return []
    except:
        return []

def download_image(url: str) -> str | None:
    """Download one image, return local path or None if failure."""
    filename = os.path.basename(urlparse(url).path)
    local_path = os.path.join(IMG_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    except:
        return None


# 1) Load CSV
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 2) In parallel: fetch image URLs for each listing
total_images_found = 0

with ThreadPoolExecutor(MAX_WORKERS) as executor:
    future_to_row = {
        executor.submit(fetch_listing_images, row["listing_url"]): row
        for row in rows
        if row.get("listing_url")
    }

    with tqdm(total=len(future_to_row), desc="Fetching listing pages") as pbar:
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                image_urls = future.result()
                row["image_urls"] = image_urls
                total_images_found += len(image_urls)
                pbar.set_postfix(
                    {
                        "listing_url": row["listing_url"][:50],
                        "total_images": total_images_found,
                    }
                )
            except Exception as e:
                row["image_urls"] = []
                pbar.set_postfix({"error": str(e)})
            pbar.update(1)
# 6) Write JSON
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(
    f"✅ Done! {len(rows)} listings"
)

# 3) Flatten all image URLs (dedupe optional)
all_images = []
for row in rows:
    urls = row.get("image_urls", [])
    row["primary_image_url"] = urls[0] if urls else None
    all_images.extend(urls)

all_images = list(dict.fromkeys(all_images))  # remove duplicates (preserve order)

# 4) Download all images in parallel
with ThreadPoolExecutor(MAX_WORKERS) as executor:
    futures = {executor.submit(download_image, url): url for url in all_images}

    local_map = {}
    with tqdm(total=len(futures), desc="Downloading images") as pbar:
        for future in as_completed(futures):
            url = futures[future]
            try:
                path = future.result()
                if path:
                    local_map[url] = path
                pbar.set_postfix({"image": os.path.basename(urlparse(url).path)})
            except Exception as e:
                pbar.set_postfix({"error": str(e)})
            pbar.update(1)

# 5) Attach local paths back to each row
for row in rows:
    row["local_image_paths"] = [
        local_map[url] for url in row.get("image_urls", []) if url in local_map
    ]

# 6) Write JSON
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(
    f"✅ Done! {len(rows)} listings → {len(local_map)} images downloaded → {JSON_OUT}"
)
