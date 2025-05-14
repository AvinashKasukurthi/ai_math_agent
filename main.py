from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(a: float, b: float, operator: str) -> str:
    """
    Useful for performing basic arithmetic calculation with numbers with operator +, -, *, %, / or any operation
    """
    print("Calculator Tool invoked")
    ans = eval(f"{a} {operator} {b}")

    if isinstance(ans, (int, float)):
        return f"The result of {a} {operator} {b} is: {ans}"
    elif isinstance(ans, str):
        return f"The result of {a} {operator} {b} is: '{ans}'"
    elif isinstance(ans, list):
        return f"The result of {a} {operator} {b} is: [{', '.join(map(str, ans))}]"
    elif isinstance(ans, dict):
        return f"The result of {a} {operator} {b} is: {{{', '.join([f'{k}: {v}' for k, v in ans.items()])}}}"
    elif isinstance(ans, tuple):
        return f"The result of {a} {operator} {b} is: ({', '.join(map(str, ans))})"
    elif isinstance(ans, set):
        return f"The result of {a} {operator} {b} is: {{{', '.join(map(str, ans))}}}"
    elif isinstance(ans, complex):
        return f"The result of {a} {operator} {b} is: {ans.real} + {ans.imag}i"
    elif isinstance(ans, bytes):
        return f"The result of {a} {operator} {b} is: {ans.decode()}"
    elif isinstance(ans, memoryview):
        return f"The result of {a} {operator} {b} is: {ans.tobytes().decode()}"
    elif isinstance(ans, range):
        return f"The result of {a} {operator} {b} is: {list(ans)}"
    elif isinstance(ans, frozenset):
        return f"The result of {a} {operator} {b} is: {{{', '.join(map(str, ans))}}}"
    elif isinstance(ans, bytearray):
        return f"The result of {a} {operator} {b} is: {ans.decode()}"
    elif isinstance(ans, type(None)):
        return f"The result of {a} {operator} {b} is: None"
    elif isinstance(ans, NotImplemented):
        return f"The result of {a} {operator} {b} is: NotImplemented"
    elif isinstance(ans, Ellipsis):
        return f"The result of {a} {operator} {b} is: Ellipsis"
    elif isinstance(ans, object):
        return f"The result of {a} {operator} {b} is: {str(ans)}"
    elif isinstance(ans, Exception):
        return f"The result of {a} {operator} {b} is: {str(ans)}"
    elif isinstance(ans, type):
        return f"The result of {a} {operator} {b} is: {ans.__name__}"
    elif isinstance(ans, NotImplementedError):
        return f"The result of {a} {operator} {b} is: NotImplementedError"
    elif isinstance(ans, StopIteration):
        return f"The result of {a} {operator} {b} is: StopIteration"
    elif isinstance(ans, GeneratorExit):
        return f"The result of {a} {operator} {b} is: GeneratorExit"
    elif isinstance(ans, KeyboardInterrupt):
        return f"The result of {a} {operator} {b} is: KeyboardInterrupt"
    elif isinstance(ans, SystemExit):
        return f"The result of {a} {operator} {b} is: SystemExit"
    elif isinstance(ans, AssertionError):
        return f"The result of {a} {operator} {b} is: AssertionError"
    elif isinstance(ans, ImportError):
        return f"The result of {a} {operator} {b} is: ImportError"
    elif isinstance(ans, IndexError):
        return f"The result of {a} {operator} {b} is: IndexError"
    elif isinstance(ans, KeyError):
        return f"The result of {a} {operator} {b} is: KeyError"
    elif isinstance(ans, ValueError):
        return f"The result of {a} {operator} {b} is: ValueError"
    else:
        return f"The result of {a} {operator} {b} is: {str(ans)}"


def main():
    model = ChatOllama(temperature=0, model="llama3.2")

    tools = [calculator]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break

        print("\n🤖Assistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")

        print()


if __name__ == "__main__":
    main()
