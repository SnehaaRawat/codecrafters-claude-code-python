import sys
import json
import os
from openai import OpenAI
def main():
    # Parse the prompt from command line arguments
    prompt = None
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-p" and i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]
            break
    if not prompt:
        print("Usage: -p <prompt>", file=sys.stderr)
        sys.exit(1)
    # Get API configuration from environment variables
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Create the OpenAI client
    client = OpenAI(base_url=base_url, api_key=api_key)
    # Make the API request with the Read tool
    response = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ],
    )
    # Check if there are tool calls in the response
    message = response.choices[0].message
    if message.tool_calls:
        # Extract the first tool call
        tool_call = message.tool_calls[0]
        # Get the function name and arguments
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        # Handle the "Read" tool
        if function_name == "Read":
            file_path = arguments["file_path"]
            # Open the file and print its contents
            with open(file_path, "r") as f:
                print(f.read(), end="")
    # If there's text content, print it
    if message.content:
        print(message.content, end="")
    sys.exit(0)
if __name__ == "__main__":
    main()