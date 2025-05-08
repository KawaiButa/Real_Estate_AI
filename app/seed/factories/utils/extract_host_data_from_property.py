from typing import List, Dict, Any
import json
# Fields you want to extract for each host:
HOST_FIELDS = [
    "host_id",
    "host_name",
    "host_since",
    "host_location",
    "host_about",
    "host_thumbnail_url",
    "host_neighbourhood",
]

def extract_unique_hosts(
    listings: List[Dict[str, Any]],
    host_fields: List[str]
) -> List[Dict[str, Any]]:
    host_map: Dict[str, Dict[str, Any]] = {}
    for listing in listings:
        hid = listing.get("host_id")
        if not hid:
            continue
        # Build or update this host’s entry:
        host_entry = host_map.setdefault(hid, {})
        for field in host_fields:
            if field in listing:
                # strip the "host_" prefix if you prefer shorter keys:
                key = field[len("host_"):] if field.startswith("host_") else field
                host_entry[key] = listing[field]
    # Return as a list of host dicts
    return list(host_map.values())
file_path = "properties.json"
with open(file_path, 'r', encoding='utf-8') as f:
    listings = json.load(f) 
unique_hosts = extract_unique_hosts(listings, HOST_FIELDS)

with open("hosts.json", "w+ ", encoding="utf-8") as f:
    json.dump(unique_hosts, f, ensure_ascii=False, indent=2)
