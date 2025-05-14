from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(eval_expression: str) -> str:
    """
    Useful for performing basic arithmetic calculation with numbers with operator +, -, *, %, / or any operation
    """
    print("Calculator Tool invoked")
    ans = eval(eval_expression)

    return str(ans)


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
