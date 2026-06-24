import os
import json
from google import genai
from google.genai import types

# load orders
with open("orders.json") as f:
    ORDERS = json.load(f)

# tool implementations

def lookup_order(order_id: str) -> str:
    """Return order details as a readable string, or an error if not found."""
    order = ORDERS.get(order_id)
    if not order:
        return f"Order {order_id} not found."
    return (
        f"Order {order_id}: {order['item']} — "
        f"${order['price']}, purchased {order['purchased']}, "
        f"{order['warranty_months']}-month warranty."
    )


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression and return the result."""
    # only allow digits, spaces, and basic operators – nothing fancy
    allowed = set("0123456789 +-*/.,()")
    if not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}})  # no builtins → safe enough here
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# tool registry: name → callable 
TOOL_FUNCTIONS = {
    "lookup_order": lookup_order,
    "calculate": calculate,
}

# Gemini tool definitions 
TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="lookup_order",
                description="Look up a customer order by its order ID and return item, price, purchase date, and warranty length.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "order_id": types.Schema(
                            type=types.Type.STRING,
                            description="The order ID to look up, e.g. 'A1001'.",
                        )
                    },
                    required=["order_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="calculate",
                description="Evaluate a simple arithmetic expression and return the numeric result.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "expression": types.Schema(
                            type=types.Type.STRING,
                            description="A plain arithmetic expression, e.g. '1200 * 3'.",
                        )
                    },
                    required=["expression"],
                ),
            ),
        ]
    )
]

# the loop 

def run_turn(client: genai.Client, messages: list, user_input: str, step_limit: int = 5) -> str:
    """
    Add the user message to memory, then run the tool-call loop until the model
    gives a plain-text answer (or we hit the step limit).

    Returns the final text answer.
    """
    # append the new user message to shared memory
    messages.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
    print(f"USER: {user_input}")


    for step in range(1, step_limit + 1):
        print(f"\n[step {step}] calling model …")

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=messages,
            config=types.GenerateContentConfig(tools=TOOLS),
        )

        candidate = response.candidates[0]
        model_parts = candidate.content.parts

        # append the model's raw response to memory so it knows what it said
        messages.append(types.Content(role="model", parts=model_parts))

        # check what the model returned
        tool_calls = [p for p in model_parts if p.function_call is not None]
        text_parts = [p for p in model_parts if p.text]

        if not tool_calls:
            # model gave a final text answer — we're done
            answer = " ".join(p.text for p in text_parts).strip()
            print(f"\nMODEL ANSWER: {answer}")
            return answer

        # execute every tool the model asked for
        tool_result_parts = []
        for part in tool_calls:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)
            print(f"  → tool call : {fn_name}({fn_args})")

            if fn_name in TOOL_FUNCTIONS:
                result_text = TOOL_FUNCTIONS[fn_name](**fn_args)
            else:
                result_text = f"Error: unknown tool '{fn_name}'."

            print(f"  ← tool result: {result_text}")

            tool_result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response={"result": result_text},
                    )
                )
            )

        # append tool results to memory and loop back
        messages.append(types.Content(role="user", parts=tool_result_parts))

    # hit the step limit without a final answer
    print("\n[!] step limit reached — couldn't finish in time.")
    return "Sorry, I couldn't finish answering within the step limit."


# main: two-turn memory demo

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GOOGLE_API_KEY before running.")

    client = genai.Client(api_key=api_key)

    # shared memory — persists across both turns
    messages: list = []

    # Turn 1
    run_turn(client, messages, "What did order A1001 cost?")

    # Turn 2 — only answerable because Turn 1 is still in `messages`
    run_turn(client, messages, "And what about three of them?")

    print("\n\nFull conversation memory:")
    print(f"  {len(messages)} message(s) in history\n")


if __name__ == "__main__":
    main()
