import sys

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.5:2b"

# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    prices = {"laptop":  1299.99, "headphones": 199.99, "smartphone": 89.50}
    result = prices.get(product.lower(), -1.0)
    print(f"    >> Executing get_product_price with product: {product}, result: {result}")
    return result

@tool
def apply_discount(price: float, product: str, discount_tier: float) -> float:
    """Apply a discount to a price.
    Available tiers: bronze, silver, gold,"""
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    print(f"    >> Executing apply_discount: {discount} on price: {price}")
    return round(price * (1-discount/100), 2)

# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    
    llm = init_chat_model(model=f"anthropic/claude-3-5-sonnet-20241022", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant."
                "You have access to a product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly.\n"
                "1. NEVER guess or assume any product price."
                "You MUST call get_product_price to first get the real price of a product.\n"
                "2. Only call apply_discount after you have obtained the price from the get_product_price tool."
                "Pass the exact price returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate the discount yourself using math.\n"
                "Always use the apply_discount tool to calculate the discounted price.\n"
                "4. If the user does not specify a discount tier,"
                "ask them which tier to use - do NOT assume one.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration} of {MAX_ITERATIONS}")
        ai_message = llm_with_tools.invoke(messages)
    
        tool_calls = ai_message.tool_calls

        # If no tool calls, the LLM is either done or asking a clarifying
        # question. Show it to the user and let their reply decide which:
        # a blank reply means "this is the final answer", any other reply
        # is treated as an answer to the clarification and fed back in.
        if not tool_calls:
            print(f"\nAssistant: {ai_message.content}")
            messages.append(ai_message)

            user_reply = input("Your response (press Enter to accept as final answer): ").strip()
            if not user_reply:
                print(f"\nFinal Answer: {ai_message.content}")
                return ai_message.content

            messages.append(HumanMessage(content=user_reply))
            continue

        # Process only the FIRST tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool Selected: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict.")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"Tool Result: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent *(.bind_tools)!")
    print()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ").strip()

    result = run_agent(question)