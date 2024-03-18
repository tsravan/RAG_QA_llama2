from read_data import get_cord19_data_dir
from chromadb_main import check_create_chromadb


# Getting the directory of covid data
cord19_data_dir = get_cord19_data_dir()

# Check if any chromaDB exists in a given path if not Create chromaDB in the specified path
check_create_chromadb(cord19_data_dir, data_format='*.json')