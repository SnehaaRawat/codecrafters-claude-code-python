import os
import sys
import json
import subprocess
import argparse
from functools import wraps
from typing import Callable


# Dynamically import the OpenAI client to avoid static import errors in editors
# and provide a clear runtime message if the package is missing.
import importlib
try:
    _openai_mod = importlib.import_module("openai")
    OpenAI = getattr(_openai_mod, "OpenAI")
except Exception:
    OpenAI = None


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
IS_LOCAL = os.getenv("IS_LOCAL", False)


class Parameter:
    def __init__(self, type: str, name: str, description: str, required: bool = True):
        self.type = type
        self.name = name
        self.description = description
        self.required = required


def toolcall(name: str, description: str, parameters: list[Parameter]) -> Callable:
    def _decorator(func):
        def wrapper(*args, **kw):
            return func(*args, **kw)

        wrapper.name = name
        wrapper.description = description
        wrapper.parameters = parameters
        return wrapper

    return _decorator


def init_toolcalls(*funcs):
    results = {}
    toolcalls = []
    for func in funcs:
        results[func.name] = func
        props = {}
        for p in func.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
        toolcalls.append(
            {
                "type": "function",
                "function": {
                    "name": func.name,
                    "description": func.description,
                    "parameters": {
                        "type": "object",
                        "required": [p.name for p in func.parameters if p.required],
                        "properties": props,
                    },
                },
            }
        )
    return results, toolcalls


def log(msg: str):
    print(msg, file=sys.stderr)


@toolcall(
    name="Read",
    description="Read and return the contents of a file",
    parameters=[
        Parameter(
            type="string", name="file_path", description="The path to the file to read"
        )
    ],
)
def Read(**args):
    log(f"Read: {args}")
    file_path = args["file_path"]
    log(f"Read: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        return str(e)


@toolcall(
    name="Write",
    description="Write content to a file",
    parameters=[
        Parameter(
            type="string",
            name="file_path",
            description="The path to the file to write to",
        ),
        Parameter(
            type="string",
            name="content",
            description="The content to write to the file",
        ),
    ],
)
def Write(**args):
    log(f"Write: {args}")
    file_path = args["file_path"]
    content = args["content"]
    log(f"Write: {file_path}: {content}")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return ""
    except IOError as e:
        return str(e)


@toolcall(
    name="Bash",
    description="Execute a shell command",
    parameters=[
        Parameter(type="string", name="command", description="The command to execute")
    ],
)
def Bash(**args):
    log(f"Bash: {args}")
    command = args["command"]
    log(f"Bash: {command}")
    result = subprocess.run(command.split(), capture_output=True)
    return f"exit_code: ${result.returncode}\noutput:\n${result.stdout}\nerror:\n${result.stderr}"


functions, tools = init_toolcalls(Read, Write, Bash)


def call_function(tool_call):
    log(f"call: {tool_call.function.name}")
    fn = functions[tool_call.function.name]
    args = json.loads(tool_call.function.arguments)
    log("parameters: " + json.dumps(args))
    return fn(**args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", required=True, help="The prompt to send to the model")
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' Python package is not installed or could not be imported; install it with: pip install openai"
        )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    model = "z-ai/glm-4.5-air:free" if IS_LOCAL else "anthropic/claude-haiku-4.5"
    messages = [{"role": "user", "content": args.prompt}]

    steps = 0

    while steps < 100:
        steps += 1
        chat = client.chat.completions.create(model=model, messages=messages, tools=tools)

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        choice = chat.choices[0]
        messages.append(choice.message)

        log(choice.message.content)

        tool_calls = getattr(choice.message, "tool_calls", None)
        if tool_calls and len(tool_calls) > 0:
            for tool_call in tool_calls:
                if tool_call.type == "function":
                    result = call_function(tool_call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
        else:
            # print the model output when no tool calls are requested
            print(choice.message.content)

        if getattr(choice, "finish_reason", None) == "stop":
            break


if __name__ == "__main__":
    main()


