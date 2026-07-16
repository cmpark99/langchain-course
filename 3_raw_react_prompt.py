import sys
import re
import inspect

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
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    print(f"    >> Executing apply_discount: {discount} on price: {price}")
    return round(price * (1-discount/100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text. 
# We derive descriptions from the functions themselves using inspect.

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers to get the original function for signature and docstring inspection
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price to first get the real price of a product.
2. Only call apply_discount after you have obtained the price from the get_product_price tool. Pass the exact price returned by get_product_price - do NOT pass a made-up number.
3. NEVER calculate the discount yourself using math. Always use the apply_discount tool to calculate the discounted price.
4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {{question}}
Thought:"""

# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent
# all agency comes from the prompt above and our regex parsing below.

@traceable(name="Ollama Chat with React Prompt", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options, think=False)

# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration} of {MAX_ITERATIONS}")
        full_prompt = prompt + scratchpad

        # Stop token prevents the LLM from generating its own Observation -
        # we inject the real tool result instead
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation:"], "temperature": 0.0}
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"\n[Parsed] Final Answer: {final_answer}")
            return final_answer

        # Change 6: Parse tool calls from the LLM output using regex instead of relying on a structured tool call object.
        action_match = re.search(r"Action:\s*([^\s(]+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match:
            print("ERROR: Could not parse Action from LLM output.")
            break

        tool_name = action_match.group(1).strip()

        if action_input_match:
            tool_input_raw = action_input_match.group(1).strip()
        else:
            # Some models skip the "Action Input:" line and instead write the
            # call inline, e.g. "Action: get_product_price(product=\"laptop\")".
            inline_args_match = re.search(r"Action:\s*[^\s(]+\((.*)\)", output)
            if not inline_args_match:
                print("ERROR: Could not parse Action Input from LLM output.")
                break
            tool_input_raw = inline_args_match.group(1).strip()

        print(f"Tool Selected: {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"    [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"Tool Result: {observation}")

        # Change 7: History is one growing string re-set every iteration (replaces messages.append)
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
