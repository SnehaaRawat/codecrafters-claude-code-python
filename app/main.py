import sys
import json

def main():
    # Step 1: Read the assistant's response JSON from stdin
    response = sys.stdin.read()
    data = json.loads(response)

    # Step 2: Extract the first tool call
    tool_call = data["choices"][0]["message"]["tool_calls"][0]

    # Step 3: Get the function name and arguments
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])

    # Step 4: Handle the "Read" tool
    if function_name == "Read":
        file_path = arguments["file_path"]

        # Step 5: Open the file and print its contents
        with open(file_path, "r") as f:
            print(f.read())

if __name__ == "__main__":
    main()