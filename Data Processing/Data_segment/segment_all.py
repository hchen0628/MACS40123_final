import json
import jieba

def segment_large_dataset(json_file_path, output_file_path, chunk_size=10000):
    try:
        # Open the input JSON file
        with open(json_file_path, mode='r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Get the total number of posts
        total_posts = len(data)

        # Process in chunks to avoid loading too much data into memory at once
        with open(output_file_path, mode='w', encoding='utf-8') as output_file:
            output_file.write("[")  # Start of JSON array

            for start in range(0, total_posts, chunk_size):
                # Get the current chunk of data
                chunk = data[start:start + chunk_size]

                segmented_chunk = []
                for post in chunk:
                    text = post.get('title', '') + " " + post.get('item_data', '')
                    segmented_words = list(jieba.cut(text))  # Perform segmentation using jieba
                    segmented_chunk.append(segmented_words)

                # Write the segmented results to the output file
                for i, words in enumerate(segmented_chunk):
                    json.dump(words, output_file, ensure_ascii=False)
                    # Add a comma separator unless it's the last element
                    if start + i + 1 < total_posts:
                        output_file.write(",\n")

            output_file.write("]")  # End of JSON array

        print(f"Segmentation completed for all {total_posts} posts. Results saved to {output_file_path}")

    except FileNotFoundError:
        print("Error: JSON file not found. Please check the file path.")
    except json.JSONDecodeError:
        print("Error: JSON decoding failed. Please check if the file content is a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # JSON file paths
    json_file_path = r"D:\Download\cleaned_data_final_final.json"
    # Output JSON file path
    output_file_path = r"D:\Download\segmented_data_cleaned.json"

    # Segment all posts and save the results to a new JSON file
    segment_large_dataset(json_file_path, output_file_path, chunk_size=10000)
