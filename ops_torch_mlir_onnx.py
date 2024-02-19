import requests
import re
from typing import List, Tuple
import json
import os 
import time

def download_file(url: str) -> str:
    """
    Downloads a file from a given URL.
    
    :param url: URL of the file to download.
    :return: The content of the file as a string.
    """
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Downloaded successfully from {url}")
        return response.text
    else:
        print(f"Failed to download from {url}. Status code: {response.status_code}")
        return ""

def parse_supported_ops(file_contents: str) -> List[Tuple[str, int]]:
    """
    Parses the C++ file contents and extracts a list of supported operations along with their versions.
    
    :param file_contents: The content of the C++ file as a string.
    :return: A list of tuples where each tuple contains the operation name and its version.
    """
    op_pattern = re.compile(r'patterns\.onOp\(\s*"([^"]+)"\s*,\s*(\d+)\s*,', re.MULTILINE)

    matches = op_pattern.findall(file_contents)
    return [(match[0], int(match[1])) for match in matches]

def get_supported_ops() -> List[Tuple[str, int]]:
    # check if cached supported ops from the last 2 hours exist
    # if not, download the files and parse them
    # return the list of supported ops

    cache_file_name = "supported_ops_cache.json"

    # check cache file existance
    if os.path.exists(cache_file_name) and  (time.time() - os.path.getmtime(cache_file_name)) / 3600 < 2:
            # read the cache file
        with open(cache_file_name, "r") as cache_file:
            return json.load(cache_file)

    urls = [
        "https://raw.githubusercontent.com/llvm/torch-mlir/main/lib/Conversion/TorchOnnxToTorch/DefaultDomainAtoF.cpp",
        "https://raw.githubusercontent.com/llvm/torch-mlir/main/lib/Conversion/TorchOnnxToTorch/DefaultDomainGtoP.cpp",
        "https://raw.githubusercontent.com/llvm/torch-mlir/main/lib/Conversion/TorchOnnxToTorch/DefaultDomainQtoZ.cpp"
    ]

    all_supported_ops = []

    for url in urls:
        file_contents = download_file(url)
        if file_contents:
            supported_ops = parse_supported_ops(file_contents)
            all_supported_ops.extend(supported_ops)
            print(f"Operations parsed from {url.split('/')[-1]}: {len(supported_ops)}")


    json.dump(all_supported_ops, open(cache_file_name, "w"))
    return all_supported_ops






if __name__ == "__main__":
    all_supported_ops = get_supported_ops()
    print("\nAll Supported Operations and Versions:")
    for op, version in all_supported_ops:
        print(f"{op}: Version {version}")
        # total number of ops
        total_ops = len(all_supported_ops)
        print(f"Total number of ops: {total_ops}")
    print(all_supported_ops[:5])