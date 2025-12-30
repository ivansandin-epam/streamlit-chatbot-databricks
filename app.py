import streamlit as st
import os
import re
import pandas as pd
from databricks import sql
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

MODEL = "gpt-4o"

client = AzureOpenAI(
    api_key         = os.environ["AZURE_OPENAI_API_KEY"],
    api_version     = "2025-04-01-preview",
    azure_endpoint  = os.environ["AZURE_ENDPOINT"]
)

# --- Databricks query execution function ---
def execute_select_query(query: str):
    """
    Executes a SELECT query on Databricks and returns the results as a pandas DataFrame.
    Only SELECT statements are allowed for security reasons.
    """
    if not re.match(r"^\s*SELECT\s", query, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    connection = sql.connect(
        server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"]
    )

    try:
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    finally:
        connection.close()

# --- Function schema for LLM function calling ---
functions = [
    {
        "name": "generate_sql",
        "description": "Generate a safe SQL SELECT query to answer the user's question about the employee training database.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A complete SQL SELECT query that answers the user's question."
                }
            },
            "required": ["sql_query"]
        }
    }
]

# --- Streamlit UI ---
st.title("Chatbot for Employee Training Tracker")
st.caption("Streamlit application using DIAL API and Databricks SQL")

# --- System prompt with schema and security instructions ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": """
You are an Employee Training Tracker Bot. 
You answer questions about employee training progress using the following database schema:

Tables:
1. employees (id INT PRIMARY KEY, name STRING, department STRING, hire_date DATE)
2. courses (id INT PRIMARY KEY, title STRING, required_for STRING)
3. completions (id INT PRIMARY KEY, employee_id INT, course_id INT, completion_date DATE)

Relationships:
- completions.employee_id is a foreign key to employees.id
- completions.course_id is a foreign key to courses.id

Best Practices & Security Instructions:
- Only generate SQL queries that use SELECT statements. Do NOT generate or suggest any queries that modify data (no INSERT, UPDATE, DELETE, MERGE, DROP, ALTER, TRUNCATE, or CREATE).
- Never expose or request sensitive information such as passwords, emails, or personal identifiers beyond what is present in the schema.
- Do not attempt to change the database schema or structure.
- Always validate user intent and clarify ambiguous requests before generating SQL.
- If a user asks for something outside the scope of the schema or for restricted operations, politely refuse and explain the limitation.
- Format SQL queries clearly and concisely.
- When summarizing results, do not fabricate data—only use what is returned from the database.

Use this schema and these rules to generate SQL queries and answer user questions about employee training, course completions, and departmental analytics.
"""
    }]

# --- Display chat history ---
for message in st.session_state.messages:
    if message["role"] == "system":
        continue 
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("How can I help you?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Ask LLM to generate the SQL query using function calling
    sql_response = client.chat.completions.create(
        model=MODEL,
        messages=st.session_state.messages,
        functions=functions,
        function_call={"name": "generate_sql"}
    )

    # Extract the SQL query from the function call arguments
    sql_args = sql_response.choices[0].message.function_call.arguments
    sql_query = json.loads(sql_args)["sql_query"]

    # Display the generated SQL query for transparency
    with st.expander("Generated SQL Query"):
        st.code(sql_query, language="sql")

    # Step 2: Execute the SQL query and show results
    try:
        df = execute_select_query(sql_query)
        if df is not None and not df.empty:
            st.dataframe(df)
            # Step 3: Ask LLM to summarize the results for the user
            # Add the SQL and result as context
            summary_prompt = st.session_state.messages + [
                {"role": "system", "content": f"The following SQL query was executed to answer the user's question:\n{sql_query}\n\nHere are the results:\n{df.head(20).to_markdown(index=False)}\n\nSummarize the results for the user in a concise and clear way."}
            ]
            summary_response = client.chat.completions.create(
                model=MODEL,
                messages=summary_prompt,
            )
            assistant_replied = summary_response.choices[0].message.content
        elif df is not None and df.empty:
            assistant_replied = "The query executed successfully, but there are no results to display."
        else:
            assistant_replied = "There was an error executing the query."
    except Exception as e:
        assistant_replied = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_replied})
    with st.chat_message("assistant"):
        st.markdown(assistant_replied)
