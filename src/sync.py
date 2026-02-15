import os
from dotenv import load_dotenv
from notion_client import Client
import chromadb
from chromadb.config import Settings
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load .env
# -----------------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

NOTION_TOKEN = os.getenv("NOTION_TOKEN")

if not NOTION_TOKEN:
    raise ValueError("NOTION_TOKEN not found in .env")

# -----------------------------
# Initialize Notion
# -----------------------------
notion = Client(auth=NOTION_TOKEN)

# -----------------------------
# Initialize Chroma (persistent)
# -----------------------------
chroma_client = chromadb.Client(
    Settings(
        persist_directory="chromadb",
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection(
    name="eldath_lore"
)

# -----------------------------
# Load Local Embedding Model
# -----------------------------
print("🔄 Loading local embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded.")

# -----------------------------
# Config
# -----------------------------
ROOT_PAGE_ID = "5b82e3411ef844d98c47971ec4b8e918"


# -----------------------------
# Embedding Function
# -----------------------------
def embed(text):
    return model.encode(text).tolist()


# -----------------------------
# Extract Text
# -----------------------------
def extract_text(blocks):
    text = ""

    for block in blocks:
        block_type = block["type"]

        if block_type in [
            "paragraph",
            "heading_1",
            "heading_2",
            "heading_3",
            "bulleted_list_item",
            "numbered_list_item",
        ]:
            for t in block[block_type]["rich_text"]:
                text += t["plain_text"] + "\n"

    return text.strip()


# -----------------------------
# Sync Page
# -----------------------------
def sync_page(page_id):
    print(f"\n📄 Syncing page {page_id}")

    blocks = notion.blocks.children.list(block_id=page_id)["results"]

    content = extract_text(blocks)

    if content:
        embedding = embed(content)

        collection.upsert(
            documents=[content],
            embeddings=[embedding],
            ids=[page_id]
        )

        print("✅ Saved page content")

    for block in blocks:

        # Child page
        if block["type"] == "child_page":
            sync_page(block["id"])

        # Child database
        if block["type"] == "child_database":
            sync_database(block["id"])


# -----------------------------
# Sync Database
# -----------------------------
def sync_database(database_id):
    print(f"\n📚 Syncing database {database_id}")

    pages = notion.databases.query(database_id=database_id)

    for page in pages["results"]:
        page_id = page["id"]

        print(f"   ↳ Entry {page_id}")

        blocks = notion.blocks.children.list(block_id=page_id)["results"]
        content = extract_text(blocks)

        if not content:
            continue

        embedding = embed(content)

        collection.upsert(
            documents=[content],
            embeddings=[embedding],
            ids=[page_id]
        )

        print("   ✅ Saved entry")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting full workspace sync...")
    sync_page(ROOT_PAGE_ID)
    print("\n🎉 Sync complete.")
