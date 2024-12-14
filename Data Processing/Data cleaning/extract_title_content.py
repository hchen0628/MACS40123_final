import json

def extract_title_and_content(json_file_path, output_file_path):
    try:
        # Open and read the JSON file content
        with open(json_file_path, mode='r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Extract "title" and "item_data" from each post
        filtered_data = [{"title": post.get("title", ""), "item_data": post.get("item_data", "")} for post in data]

        # Save the extracted data to a new JSON file
        with open(output_file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(filtered_data, output_file, ensure_ascii=False, indent=4)

        print(f"Data extraction completed. Extracted {len(filtered_data)} posts and saved to {output_file_path}")

    except FileNotFoundError:
        print("Error: JSON file not found. Please check the file path.")
    except json.JSONDecodeError:
        print("Error: JSON decoding failed. Please check if the file content is a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Input and output file paths
    json_file_path = r"D:\Download\data_40123.json"
    output_file_path = r"D:\Download\extracted_data.json"

    # Extract the "title" and "item_data" from the posts
    extract_title_and_content(json_file_path, output_file_path)