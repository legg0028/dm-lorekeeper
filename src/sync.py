import os
from dotenv import load_dotenv
from notion_client import Client
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not NOTION_TOKEN:
    raise ValueError("NOTION_TOKEN not found.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found.")

notion = Client(auth=NOTION_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.Client(
    Settings(
        persist_directory="chromadb",
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection(
    name="eldath_lore"
)

ROOT_PAGE_ID = "5b82e3411ef844d98c47971ec4b8e918"


def extract_text(blocks):
    text = ""
    for block in blocks:
        block_type = block["type"]
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
            for t in block[block_type]["rich_text"]:
                text += t["plain_text"] + "\n"
    return text.strip()


def get_page_content(page_id):
    response = notion.blocks.children.list(block_id=page_id)
    return extract_text(response["results"])


def embed(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def sync_page(page_id):
    print("Fetching page...")
    content = get_page_content(page_id)

    if not content:
        print("No content found.")
        return

    print("Embedding...")
    embedding = embed(content)

    print("Saving...")
    collection.upsert(
        documents=[content],
        embeddings=[embedding],
        ids=[page_id]
    )

    print("Sync complete.")


if __name__ == "__main__":
    sync_page(ROOT_PAGE_ID)
