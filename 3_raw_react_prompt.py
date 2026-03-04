from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

import re
import inspect

MAX_ITERATIONS = 10

MODEL = "qwen3.5:2b"


# --- Tools  ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product."""
    print(f"  >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "keyboard": 89.50,
        "headphones": 149.95,
    }
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return final price.
    Available tiers: bronze, silver, gold"""

    print(f"  >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)  # Ensure price is a float
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100),2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_func in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g @traceable) to get to the original function and its signature
        original_func = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_func)
        docstring = inspect.getdoc(original_func) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You must call the get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received  a price from get_product_price. Pass the exact price returned from get_product_price - do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool to calculate the final price after discount.
4. If the user does not specify a discount tier, ask them which tier to use - do Not assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# Helper : traced Ollama call

@traceable(name="Ollama Chat raw", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# --- Agent Loop ---
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # CHANGE: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""
    

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad
        
        # Difference: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(
            model=MODEL, 
            messages=[{"role": "user", "content": full_prompt}], 
            options={
                "stop": ["\nObservation:"],  # Stop generation when LLM outputs "Observation:" to force one tool call at a time
                "temperature": 0,
            }
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in the LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "="*60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # CHANGE - Parse tool calls from raw text with regex - fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in the LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        

        if not action_match or not action_input_match:
            print(f"  [Parsing] ERROR:  Could not parse Action/Action Input from LLM output.")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected]: {tool_name} with args: {tool_input_raw}")

        # Split comma separated args : strip key= prefix if LLM outputs key=value format
        raw_args = [arg.strip() for arg in tool_input_raw.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]

        print(f" [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"ERROR: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result]: {observation}")

        # CHANGE: History is one groowing string re-sent every iteration(replaces messages.append)
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR : Reached maximum iterations without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")

    print()
    result = run_agent("What is the price of a laptop with a silver discount?")