import sys

from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.5:2b"

# --- Tools (LangChain @tool decorator) ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    prices = {"laptop":  1299.99, "headphones": 199.99, "smartphone": 89.50}
    result = prices.get(product.lower(), -1.0)
    print(f"    >> Executing get_product_price with product: {product}, result: {result}")
    return result

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price.
    Available tiers: bronze, silver, gold,"""
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    print(f"    >> Executing apply_discount: {discount} on price: {price}")
    return round(price * (1-discount/100), 2)

# Difference 2: Without @tool, we need to manually create a dictionary of tools for the agent to use. This is done in the run_agent function, where we create a list of tools and then convert it into a dictionary for easy access.
# This is exactly what LangChain's @tool decorator does for us automatically, but here we are doing it manually to show how it works under the hood.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# ---- Helper ----
# Difference 3: Without LangChain, we must manually trace LLM calls and tool calls. This is done in the run_agent function, where we print out the question, the iteration number, the tool selected, and the tool result. This is similar to what LangChain's @traceable decorator does for us automatically, but here we are doing it manually to show how it works under the hood.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    """Call the Ollama chat model with the given messages and tools."""
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools_for_llm
    )
    return response

# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }
    
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system", 
            "content": (
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
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration} of {MAX_ITERATIONS}")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke() - we pass the tools and messages directly to the Ollama chat model, which handles tool calls automatically.
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

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
        # Difference 6: Attribute access (.function.name) instead of dictionary access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool Selected: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict.")
        
        # Difference 7: Direct function call instead of tool.invoke() 
        observation = tool_to_use(**tool_args)

        print(f"Tool Result: {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent *(.bind_tools)!")
    print()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ").strip()

    # Initial question: What is the price of the laptop after getting a gold discount ?

    result = run_agent(question)