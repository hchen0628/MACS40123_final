import json
import re
from collections import defaultdict
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from mlxtend.frequent_patterns import apriori
import pandas as pd

def load_json(file_path):
    """Load JSON file data."""
    with open(file_path, mode='r', encoding='utf-8') as file:
        return json.load(file)

def filter_single_characters(words):
    """Filter out single-character words."""
    return [word for word in words if len(word) > 1]

def filter_non_chinese(words):
    """Filter out non-Chinese words."""
    chinese_only = re.compile(r'^[\u4e00-\u9fff]+$')
    return [word for word in words if chinese_only.match(word)]

def create_transactions(data, min_length=2):
    """Create transactions suitable for Apriori algorithm from the data."""
    transactions = []
    for post in data:
        filtered_words = filter_non_chinese(filter_single_characters(post))
        if len(filtered_words) >= min_length:
            transactions.append(filtered_words)
    return transactions

def run_apriori(transactions, min_support=0.01, max_length=2):
    """Run Apriori algorithm on the transactions."""
    # Convert transactions to a one-hot encoded DataFrame
    all_words = set(word for transaction in transactions for word in transaction)
    one_hot_data = pd.DataFrame(
        [{word: (word in transaction) for word in all_words} for transaction in transactions]
    )
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(one_hot_data, min_support=min_support, use_colnames=True, max_len=max_length)
    return frequent_itemsets

def find_co_occurring_words(data, initial_keyword, min_support=0.01, max_set_size=6):
    """Identify co-occurring words with the initial keyword using Apriori."""
    # Create transactions from the data
    transactions = create_transactions(data)
    
    # Run Apriori algorithm to find frequent itemsets
    frequent_itemsets = run_apriori(transactions, min_support=min_support, max_length=max_set_size)
    
    # Filter itemsets containing the initial keyword
    relevant_itemsets = frequent_itemsets[
        frequent_itemsets['itemsets'].apply(lambda x: initial_keyword in x and len(x) > 1)
    ]
    
    return relevant_itemsets

def analyze_with_apriori(file_path, initial_keyword, min_support=0.01, max_set_size=6):
    """Perform Apriori analysis starting with an initial keyword."""
    # Load the JSON data
    data = load_json(file_path)
    
    # Extract co-occurring words with the Apriori algorithm
    relevant_itemsets = find_co_occurring_words(data, initial_keyword, min_support, max_set_size)
    
    # Save the results to a JSON file
    output_file_path = f"D:\\BaiduNetdiskDownload\\{initial_keyword}_apriori_results.json"
    relevant_itemsets.to_json(output_file_path, force_ascii=False, orient='records', indent=4)
    print(f"Results saved to {output_file_path}")

def process_all_keywords(input_file_path, keywords, min_support=0.01, max_set_size=6):
    """Process all keywords in parallel to generate frequent itemsets using Apriori."""
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(analyze_with_apriori, input_file_path, keyword, min_support, max_set_size)
            for keyword in keywords
        ]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    # Input file path
    input_file_path = r"D:\Download\segmented_data_cleaned.json"

    # List of keywords
    keywords = ["民主", "公义", "公平", "投票", "利益", "保衛", "普選",
                "示威者", "示威", "抗争", "文宣", "打壓", "对抗", "落街",
                "政府", "特首", "黑社会", "滲透", "極權"]

    # Process all keywords in parallel using the Apriori algorithm
    process_all_keywords(input_file_path, keywords, min_support=0.01, max_set_size=6)
