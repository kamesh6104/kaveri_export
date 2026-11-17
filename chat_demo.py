
"""
Midstone 4: A three-turn conversation with streaming and one tool call.

Run:
    python chat_demo.py

Creates:
    run_log.jsonl
"""

import os

from sqlalchemy import create_engine, text

from src.llm_client import (
    Conversation,
    LLMClient,
    RunLog,
    tool_spec
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

LOG = "run_log.jsonl"

engine = create_engine(
    os.getenv("DATABASE_URL", "sqlite:///kaveri.db")
)


# --------------------------------------------------
# 2. DATABASE TOOL
# --------------------------------------------------

def get_revenue(country: str, year: int) -> float:
    """
    Get total revenue from one country's customers
    in one year using a read-only SELECT query.
    """

    sql = text("""
        SELECT COALESCE(SUM(o.total_amount), 0)
        FROM orders AS o
        JOIN customers AS c
            ON o.customer_id = c.customer_id
        WHERE c.country = :country
          AND o.order_date LIKE :year
    """)

    with engine.connect() as con:

        result = con.execute(
            sql,
            {
                "country": country,
                "year": f"{int(year)}-%"
            }
        ).scalar()

        return float(result or 0)


# --------------------------------------------------
# 3. DEFINE AI TOOL
# --------------------------------------------------

TOOLS = [
    tool_spec(
        "get_revenue",

        "Revenue in USD from Kaveri Exports customers "
        "in one country in one year.",

        {
            "country": {
                "type": "string",
                "description": "Country name, e.g. Germany"
            },

            "year": {
                "type": "integer",
                "description": "Four-digit year, e.g. 2024"
            }
        },

        ["country", "year"]
    )
]


# --------------------------------------------------
# 4. THREE USER QUESTIONS
# --------------------------------------------------

TURNS = [

    # Turn 1: Normal conversation
    "Hi! In one sentence, what can you help me with?",

    # Turn 2: Requires database tool calling
    "What was our revenue from Germany in 2024? Use the tool.",

    # Turn 3: Tests conversation history
    "Can you briefly repeat the Germany revenue you just looked up?"
]


# --------------------------------------------------
# 5. MAIN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    # Remove previous log file
    if os.path.exists(LOG):
        os.remove(LOG)

    # Create LLM client
    client = LLMClient(
        log=RunLog(LOG)
    )

    # Create chatbot conversation
    chat = Conversation(

        client,

        system=(
            "You are the Kaveri Exports sales assistant. "
            "Be brief. "
            "For revenue questions always call get_revenue; "
            "never guess numbers."
        ),

        tools=TOOLS,

        functions={
            "get_revenue": get_revenue
        }
    )

    # Process all three questions
    for question in TURNS:

        print(
            f"\nYOU: {question}\nAI:  ",
            end="",
            flush=True
        )

        # Ask AI and display streaming response
        chat.ask(
            question,
            on_token=lambda t: print(
                t,
                end="",
                flush=True
            )
        )

        print()

    # Display log filename
    print(f"\nwrote {LOG}")
