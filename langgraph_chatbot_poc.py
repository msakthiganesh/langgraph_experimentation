# %% [markdown]
# # LangGraph Chatbot POC
#
# This notebook demonstrates a simple chatbot using LangGraph that:
# 1. Takes user queries
# 2. Determines intent and relevance
# 3. Converts natural language to SQL
# 4. Executes SQL queries
# 5. Formats responses back to natural language

# %% [markdown]
# ## 1. Setup and Dependencies

# %%
# Install required packages (run this in your environment)
# !pip install langgraph pandas python-dotenv snowflake-connector-python cai

from snowflake.connector import DictCursor
import snowflake.connector
from cai import gdk
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# LangGraph imports

# GenAI Platform SDK for LLM calls

# Snowflake connector

# Initialize CFG GenAI GDK client
USECASE_ID = os.getenv("CFG_USECASE_ID", "chatbot_usecase")
EXPERIMENT_NAME = os.getenv("CFG_EXPERIMENT_NAME", "langgraph_chatbot")
EXPERIMENT_DESC = os.getenv("CFG_EXPERIMENT_DESC",
                            "LangGraph chatbot with CFG GenAI")

try:
    gdk = CFGGenAIGDK(USECASE_ID, EXPERIMENT_NAME, EXPERIMENT_DESC)
    print("✅ CFG GenAI GDK initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize CFG GenAI GDK: {e}")
    raise e


def call_llm(prompt: str, system_prompt: str = "", model: str = None) -> str:
    """
    Robust helper function to make LLM calls using CFG GenAI GDK
    Handles different response structures
    """
    try:
        # Use model from environment variable if not specified
        if model is None:
            model = os.getenv("CFG_MODEL_ID", "md005_openai_gpt4o")

        # Combine system prompt and user prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\nUser Query: {prompt}"
        else:
            combined_prompt = prompt

        # Prepare prompt template for CFG GenAI
        prompt_template = {
            "prompt_template": [
                {"role": "system", "content": combined_prompt}
            ]
        }

        # Set hyperparameters
        hyperparam = {
            'max_tokens': 2000,
            'temperature': 0.1
        }

        # Make the LLM call
        gdk_response = gdk.invoke_llmgateway(
            prompt_template=prompt_template,
            hyperparam=hyperparam,
            model_id=model
        )

        # Try different ways to extract the content based on response structure
        generated_response = None

        # Method 1: Standard OpenAI-like structure
        try:
            if isinstance(gdk_response, dict):
                generated_response = gdk_response['genResponse']['choices'][0]['message']['content']
        except (KeyError, TypeError, IndexError):
            pass

        # Method 2: Direct choices access
        if generated_response is None:
            try:
                if isinstance(gdk_response, dict) and 'choices' in gdk_response:
                    generated_response = gdk_response['choices'][0]['message']['content']
            except (KeyError, TypeError, IndexError):
                pass

        # Method 3: Direct response field
        if generated_response is None:
            try:
                if isinstance(gdk_response, dict):
                    if 'response' in gdk_response:
                        generated_response = gdk_response['response']
                    elif 'content' in gdk_response:
                        generated_response = gdk_response['content']
                    elif 'text' in gdk_response:
                        generated_response = gdk_response['text']
            except (KeyError, TypeError):
                pass

        # Method 4: Tuple/List response
        if generated_response is None:
            try:
                if isinstance(gdk_response, (list, tuple)) and len(gdk_response) > 0:
                    first_element = gdk_response[0]
                    if isinstance(first_element, str):
                        generated_response = first_element
                    elif isinstance(first_element, dict):
                        if 'content' in first_element:
                            generated_response = first_element['content']
                        elif 'message' in first_element:
                            generated_response = first_element['message']
                        elif 'text' in first_element:
                            generated_response = first_element['text']
            except (IndexError, TypeError):
                pass

        # Method 5: String response
        if generated_response is None:
            try:
                if isinstance(gdk_response, str):
                    generated_response = gdk_response
            except Exception:
                pass

        # If all methods failed, return the raw response as string
        if generated_response is None:
            generated_response = str(gdk_response)

        return generated_response.strip() if generated_response else "No response generated"

    except Exception as e:
        print(f"❌ CFG GenAI LLM call failed: {e}")
        print(
            f"Response type: {type(gdk_response) if 'gdk_response' in locals() else 'Unknown'}")
        if 'gdk_response' in locals():
            print(f"Response: {gdk_response}")
        return f"Error: {str(e)}"


print("✅ Dependencies loaded successfully!")

# %% [markdown]
# ## 2. Database Schema Definition and Snowflake Connection Setup
#
# Configure database schema and Snowflake connection parameters

# %%


class DatabaseSchema:
    """
    Load and manage database schema from external text file
    """

    def __init__(self, schema_file_path: str = "database_schema.txt"):
        self.schema_file_path = schema_file_path
        self.tables = {}
        self.relationships = {}
        self.raw_schema_text = ""
        self.load_schema()

    def load_schema(self):
        """
        Load database schema from text file - REQUIRED for operation
        """
        try:
            if os.path.exists(self.schema_file_path):
                with open(self.schema_file_path, 'r', encoding='utf-8') as file:
                    self.raw_schema_text = file.read()
                print(f"✅ Database schema loaded from {self.schema_file_path}")
                self.parse_schema()
            else:
                raise FileNotFoundError(
                    f"Schema file {self.schema_file_path} not found. Please create this file with your database schema.")
        except Exception as e:
            print(f"❌ Error loading schema file: {e}")
            raise e

    def parse_schema(self):
        """
        Parse the schema text file and extract table information
        This is a flexible parser that can handle various text formats
        """
        try:
            lines = self.raw_schema_text.strip().split('\n')
            current_table = None

            for line in lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Check if this is a table definition
                if line.upper().startswith('TABLE:') or line.upper().startswith('TABLE '):
                    current_table = line.split(':', 1)[1].strip(
                    ) if ':' in line else line.split(' ', 1)[1].strip()
                    current_table = current_table.upper()
                    self.tables[current_table] = {
                        'description': f"Table: {current_table}",
                        'columns': {}
                    }

                # Check if this is a table description
                elif line.upper().startswith('DESCRIPTION:') and current_table:
                    description = line.split(':', 1)[1].strip()
                    self.tables[current_table]['description'] = description

                # Check if this is a column definition
                elif current_table and ('|' in line or '\t' in line or '  ' in line):
                    # Handle different column formats: "COLUMN_NAME | TYPE | DESCRIPTION"
                    # or "COLUMN_NAME    TYPE    DESCRIPTION" (tab/space separated)
                    parts = []
                    if '|' in line:
                        parts = [part.strip() for part in line.split('|')]
                    elif '\t' in line:
                        parts = [part.strip()
                                 for part in line.split('\t') if part.strip()]
                    else:
                        # Handle space-separated (multiple spaces)
                        parts = [part.strip()
                                 for part in line.split() if part.strip()]

                    if len(parts) >= 2:
                        col_name = parts[0].upper()
                        col_type = parts[1].upper()
                        col_description = parts[2] if len(
                            parts) > 2 else f"{col_name} column"

                        self.tables[current_table]['columns'][col_name] = {
                            'type': col_type,
                            'description': col_description
                        }

            print(f"✅ Parsed {len(self.tables)} tables from schema file")

        except Exception as e:
            print(f"❌ Error parsing schema: {e}")
            raise e

    def get_schema_description(self) -> str:
        """
        Generate a comprehensive schema description for the LLM
        """
        if self.raw_schema_text and len(self.tables) > 1:
            # If we have the raw schema text and multiple tables, use it directly
            schema_text = "DATABASE SCHEMA:\n\n"
            schema_text += self.raw_schema_text
            schema_text += "\n\nAVAILABLE TABLES:\n"
            for table_name in self.tables.keys():
                schema_text += f"- {table_name}\n"
            return schema_text
        else:
            # Fallback to structured format
            schema_text = "DATABASE SCHEMA:\n\n"

            for table_name, table_info in self.tables.items():
                schema_text += f"Table: {table_name}\n"
                schema_text += f"Description: {table_info['description']}\n"
                schema_text += "Columns:\n"

                for col_name, col_info in table_info['columns'].items():
                    schema_text += f"  - {col_name} ({col_info['type']}): {col_info['description']}\n"
                schema_text += "\n"

            return schema_text

    def get_table_names(self) -> List[str]:
        """
        Get list of available table names
        """
        return list(self.tables.keys())

    def reload_schema(self):
        """
        Reload schema from file (useful for updates)
        """
        self.tables = {}
        self.relationships = {}
        self.raw_schema_text = ""
        self.load_schema()


class SnowflakeConfig:
    """
    Snowflake connection configuration
    """

    def __init__(self):
        # Load from environment variables - all required
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.database = os.getenv("SNOWFLAKE_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self.role = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

        # Validate required fields
        required_fields = {
            'account': self.account,
            'user': self.user,
            'password': self.password,
            'database': self.database
        }

        missing_fields = [field for field,
                          value in required_fields.items() if not value]
        if missing_fields:
            raise ValueError(
                f"Missing required Snowflake configuration: {', '.join(missing_fields)}. Please check your .env file.")


# Global config instances - will be initialized when needed
snowflake_config = None
database_schema = None


def initialize_configs():
    """
    Initialize global configuration instances
    """
    global snowflake_config, database_schema

    if snowflake_config is None:
        snowflake_config = SnowflakeConfig()

    if database_schema is None:
        database_schema = DatabaseSchema()

    return snowflake_config, database_schema


def get_snowflake_connection():
    """
    Create and return a Snowflake connection
    """
    global snowflake_config
    if snowflake_config is None:
        snowflake_config = SnowflakeConfig()

    try:
        conn = snowflake.connector.connect(
            account=snowflake_config.account,
            user=snowflake_config.user,
            password=snowflake_config.password,
            warehouse=snowflake_config.warehouse,
            database=snowflake_config.database,
            schema=snowflake_config.schema,
            role=snowflake_config.role
        )
        print("✅ Snowflake connection established successfully!")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to Snowflake: {e}")
        raise e


def execute_snowflake_query(sql_query: str) -> Dict[str, Any]:
    """
    Execute a SQL query against Snowflake and return results

    Args:
        sql_query: The SQL query to execute

    Returns:
        Dictionary containing query results and metadata
    """
    conn = None
    try:
        # Get connection
        conn = get_snowflake_connection()

        # Execute query
        cursor = conn.cursor(DictCursor)
        cursor.execute(sql_query)

        # Fetch results
        results = cursor.fetchall()

        # Get column names
        columns = [desc[0]
                   for desc in cursor.description] if cursor.description else []

        print(f"📊 Query executed successfully. Returned {len(results)} rows.")

        return {
            "success": True,
            "data": results,
            "columns": columns,
            "row_count": len(results)
        }

    except Exception as e:
        print(f"❌ Error executing Snowflake query: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": [],
            "columns": [],
            "row_count": 0
        }
    finally:
        if conn:
            conn.close()


print("✅ Snowflake connection setup complete!")

# Initialize configurations on startup to validate setup
try:
    initialize_configs()
    print("✅ All configurations initialized successfully!")
except Exception as e:
    print(f"⚠️ Configuration initialization failed: {e}")
    print("Please check your .env file and database_schema.txt file before running queries.")

# %% [markdown]
# ## 3. State Management
#
# Define the state structure for our LangGraph workflow

# %%


@dataclass
class ConversationTurn:
    """
    Represents a single conversation turn (query + response)
    """
    query: str
    response: str
    sql_query: str
    query_result: Dict[str, Any]
    timestamp: datetime
    intent: str


class SessionManager:
    """
    Manages conversation sessions and context for follow-up questions
    """

    def __init__(self):
        self.sessions: Dict[str, List[ConversationTurn]] = {}
        self.max_history = 5  # Keep last 5 conversation turns

    def add_turn(self, session_id: str, turn: ConversationTurn):
        """Add a conversation turn to the session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(turn)

        # Keep only the last max_history turns
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]

    def get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation context for LLM"""
        if session_id not in self.sessions or not self.sessions[session_id]:
            return "No previous conversation history."

        context = "PREVIOUS CONVERSATION HISTORY:\n"
        for i, turn in enumerate(self.sessions[session_id], 1):
            context += f"\n{i}. User: {turn.query}"
            context += f"\n   Response: {turn.response}"
            if turn.sql_query:
                context += f"\n   SQL Used: {turn.sql_query}"

        context += "\n\nUse this context to understand follow-up questions and references like 'that', 'those', 'compare to previous', etc."
        return context

    def get_last_query_info(self, session_id: str) -> Optional[ConversationTurn]:
        """Get the last query information for context"""
        if session_id not in self.sessions or not self.sessions[session_id]:
            return None
        return self.sessions[session_id][-1]

    def clear_session(self, session_id: str):
        """Clear session history"""
        if session_id in self.sessions:
            del self.sessions[session_id]


class WorkflowState(TypedDict):
    """
    Enhanced state structure for our LangGraph workflow with conversation context
    """
    session_id: str
    user_query: str
    original_query: str  # Store original query before context enhancement
    intent: str
    is_relevant: bool
    is_followup: bool  # Whether this is a follow-up question
    sql_query: str
    query_result: Dict[str, Any]
    final_response: str
    error: str
    context: Dict[str, Any]
    conversation_context: str  # Previous conversation context


# Global session manager
session_manager = SessionManager()


def initialize_state(session_id: str, user_query: str) -> WorkflowState:
    """
    Initialize a new workflow state with conversation context
    """
    # Get conversation context
    conversation_context = session_manager.get_conversation_context(session_id)

    return WorkflowState(
        session_id=session_id,
        user_query=user_query,
        original_query=user_query,
        intent="",
        is_relevant=False,
        is_followup=False,
        sql_query="",
        query_result={},
        final_response="",
        error="",
        context={},
        conversation_context=conversation_context
    )


print("✅ Enhanced state management with conversation context setup complete!")

# %% [markdown]
# ## 4. Tool Functions
#
# These are the core tools that our LangGraph workflow will use

# %%


def context_enhancer_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool 0: Enhance user query with conversation context for follow-up questions
    """
    print(f"🔗 Enhancing query with context: {state['user_query']}")

    # Check if there's previous conversation context
    if state['conversation_context'] == "No previous conversation history.":
        print("✅ No previous context - using original query")
        return state

    # Use LLM to determine if this is a follow-up and enhance the query
    system_prompt = f"""You are a context enhancement assistant for a business data analysis chatbot.

{state['conversation_context']}

Your job is to:
1. Determine if the current user query is a follow-up question that references previous conversation
2. If it's a follow-up, enhance the query with proper context to make it standalone
3. If it's not a follow-up, return the original query unchanged

Follow-up indicators include:
- References like "that", "those", "it", "them"
- Comparative phrases like "compared to that", "vs the previous", "how about"
- Continuation phrases like "what about", "and for", "also show me"
- Time references building on previous queries like "and last month", "for the same period"

IMPORTANT: 
- If it's a follow-up, rewrite the query to be completely standalone and clear
- If it's NOT a follow-up, return exactly: "NOT_FOLLOWUP: [original query]"
- If it IS a follow-up, return: "FOLLOWUP: [enhanced standalone query]"

Examples:
- "What about premium customers?" → "FOLLOWUP: What are the sales metrics for premium customers?"
- "Compare that to last month" → "FOLLOWUP: Compare the transaction volume from the previous query to last month's transaction volume"
- "Show me product sales" → "NOT_FOLLOWUP: Show me product sales"
"""

    try:
        llm_response = call_llm(
            prompt=f"Current user query: '{state['user_query']}'",
            system_prompt=system_prompt
        )

        print(f"🤖 Context Enhancement Response: {llm_response}")

        if llm_response.startswith("FOLLOWUP:"):
            # This is a follow-up question - use enhanced query
            enhanced_query = llm_response.replace("FOLLOWUP:", "").strip()
            state['user_query'] = enhanced_query
            state['is_followup'] = True
            print(f"✅ Enhanced follow-up query: {enhanced_query}")
        else:
            # Not a follow-up - keep original query
            state['is_followup'] = False
            print("✅ Not a follow-up question - using original query")

    except Exception as e:
        state['error'] = f"Error in context enhancement: {str(e)}"
        print(f"❌ Error in context enhancement: {e}")

    # Mark completion
    state['context_enhanced'] = True
    return state


def intent_classifier_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool 1: Determine if the user query is relevant to our use case using LLM
    """
    print(f"🔍 Analyzing intent for: {state['user_query']}")

    system_prompt = """You are an intent classifier for a comprehensive business data analysis system.
    
    Your job is to determine if a user query is relevant to business data analysis using our available tables:
    - TRANSACTIONS: Transaction records with volumes, dates, customer and product IDs
    - CUSTOMERS: Customer information including segments, locations, registration dates
    - PRODUCTS: Product catalog with categories, prices, brands, suppliers
    
    Relevant queries include:
    - Transaction analysis (volumes, amounts, counts, trends)
    - Customer analysis (segments, behavior, demographics, registration patterns)
    - Product analysis (categories, performance, pricing, brand analysis)
    - Sales and revenue analysis across any dimension
    - Time-based queries (last Friday, yesterday, this week, monthly trends, etc.)
    - Comparative analysis (compare periods, segments, products, customers)
    - Cross-table analysis (customer transaction patterns, product performance by segment, etc.)
    - Data aggregation requests (total, sum, average, count, etc.)
    - Business intelligence queries combining multiple data sources
    
    Irrelevant queries include:
    - General conversation, greetings
    - Questions about weather, news, personal topics
    - Technical support unrelated to data
    - Requests for information outside of our business data domain
    
    Respond with ONLY one of these formats:
    RELEVANT: data_query
    IRRELEVANT: general_conversation"""

    try:
        llm_response = call_llm(
            prompt=f"Classify this user query: '{state['user_query']}'",
            system_prompt=system_prompt
        )

        print(f"🤖 LLM Intent Response: {llm_response}")

        if "RELEVANT" in llm_response.upper():
            state['intent'] = "data_query"
            state['is_relevant'] = True
            print("✅ Query is relevant - proceeding with data analysis")
        else:
            state['intent'] = "irrelevant"
            state['is_relevant'] = False
            state['final_response'] = "I'm sorry, but I can only help with transaction and data-related queries. Please ask about transaction volumes, sales data, or similar topics."
            print("❌ Query is not relevant to our use case")

    except Exception as e:
        state['error'] = f"Error in intent classification: {str(e)}"
        print(f"❌ Error in intent classification: {e}")

    # Mark completion
    state['intent_classified'] = True
    return state


def nl_to_sql_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool 2: Convert natural language to SQL using LLM with multi-table support
    """
    if not state['is_relevant']:
        return state

    print(f"🔄 Converting to SQL: {state['user_query']}")

    # Get current date for context
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_day = datetime.now().strftime('%A')

    # Initialize configs and get the comprehensive schema description
    global snowflake_config, database_schema
    if snowflake_config is None:
        snowflake_config = SnowflakeConfig()
    if database_schema is None:
        database_schema = DatabaseSchema()

    schema_description = database_schema.get_schema_description()
    table_names = database_schema.get_table_names()

    system_prompt = f"""You are an expert SQL query generator for a Snowflake database with multiple related tables.

{schema_description}

Database Configuration:
- Database: {snowflake_config.database}
- Schema: {snowflake_config.schema}
- Current date: {current_date} ({current_day})

SNOWFLAKE SQL GENERATION RULES:

1. QUERY STRUCTURE:
   - Generate ONLY the SQL query, no explanations or markdown
   - Use fully qualified table names: {snowflake_config.database}.{snowflake_config.schema}.TABLE_NAME
   - Always use uppercase for SQL keywords and table/column names

2. MULTI-TABLE QUERIES:
   - Use JOINs when the query requires data from multiple tables
   - Common patterns:
     * Customer analysis: JOIN TRANSACTIONS with CUSTOMERS
     * Product analysis: JOIN TRANSACTIONS with PRODUCTS  
     * Complete analysis: JOIN all three tables
   - Use appropriate JOIN types (INNER JOIN for existing relationships)

3. DATE FUNCTIONS (Snowflake-specific):
   - Current date: CURRENT_DATE()
   - Yesterday: DATEADD('day', -1, CURRENT_DATE())
   - Last Friday: DATEADD('day', -1, DATE_TRUNC('week', CURRENT_DATE()) + 4)
   - Last week: DATE_TRUNC('week', DATEADD('week', -1, CURRENT_DATE()))
   - This month: DATE_TRUNC('month', CURRENT_DATE())
   - Last month: DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE()))

4. AGGREGATIONS:
   - Use SUM() for volume/amount calculations
   - Use COUNT() for transaction counts
   - Use AVG() for averages
   - Use GROUP BY for breakdowns by categories, dates, etc.

5. COMMON QUERY PATTERNS:
   - Transaction volume by date: GROUP BY TRANSACTION_DATE
   - Customer analysis: GROUP BY customer attributes
   - Product analysis: GROUP BY product attributes
   - Time comparisons: Use date ranges with BETWEEN or IN clauses

EXAMPLE QUERIES:

Simple transaction query:
"What was the transaction volume last Friday?"
→ SELECT SUM(VOLUME) as TOTAL_VOLUME FROM {snowflake_config.database}.{snowflake_config.schema}.TRANSACTIONS WHERE TRANSACTION_DATE = DATEADD('day', -1, DATE_TRUNC('week', CURRENT_DATE()) + 4)

Customer analysis query:
"Which customer segment had the highest sales this month?"
→ SELECT C.CUSTOMER_SEGMENT, SUM(T.VOLUME) as TOTAL_SALES FROM {snowflake_config.database}.{snowflake_config.schema}.TRANSACTIONS T INNER JOIN {snowflake_config.database}.{snowflake_config.schema}.CUSTOMERS C ON T.CUSTOMER_ID = C.CUSTOMER_ID WHERE T.TRANSACTION_DATE >= DATE_TRUNC('month', CURRENT_DATE()) GROUP BY C.CUSTOMER_SEGMENT ORDER BY TOTAL_SALES DESC

Product analysis query:
"What are the top 5 products by sales volume?"
→ SELECT P.PRODUCT_NAME, P.CATEGORY, SUM(T.VOLUME) as TOTAL_SALES FROM {snowflake_config.database}.{snowflake_config.schema}.TRANSACTIONS T INNER JOIN {snowflake_config.database}.{snowflake_config.schema}.PRODUCTS P ON T.PRODUCT_ID = P.PRODUCT_ID GROUP BY P.PRODUCT_NAME, P.CATEGORY ORDER BY TOTAL_SALES DESC LIMIT 5

Complex multi-table query:
"Show me sales by customer segment and product category for last month"
→ SELECT C.CUSTOMER_SEGMENT, P.CATEGORY, SUM(T.VOLUME) as TOTAL_SALES, COUNT(T.TRANSACTION_ID) as TRANSACTION_COUNT FROM {snowflake_config.database}.{snowflake_config.schema}.TRANSACTIONS T INNER JOIN {snowflake_config.database}.{snowflake_config.schema}.CUSTOMERS C ON T.CUSTOMER_ID = C.CUSTOMER_ID INNER JOIN {snowflake_config.database}.{snowflake_config.schema}.PRODUCTS P ON T.PRODUCT_ID = P.PRODUCT_ID WHERE T.TRANSACTION_DATE >= DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE())) AND T.TRANSACTION_DATE < DATE_TRUNC('month', CURRENT_DATE()) GROUP BY C.CUSTOMER_SEGMENT, P.CATEGORY ORDER BY TOTAL_SALES DESC"""

    try:
        llm_response = call_llm(
            prompt=f"Convert this natural language query to SQL: '{state['user_query']}'",
            system_prompt=system_prompt
        )

        # Clean up the response to extract just the SQL
        sql_query = llm_response.strip()

        # Remove any markdown formatting or extra text
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].strip()

        # Remove any trailing semicolon and clean up
        sql_query = sql_query.rstrip(';').strip()

        state['sql_query'] = sql_query
        print(f"📝 Generated SQL: {state['sql_query']}")

    except Exception as e:
        state['error'] = f"Error generating SQL: {str(e)}"
        print(f"❌ Error in SQL generation: {e}")

    # Mark completion
    state['sql_generated'] = True
    return state


def sql_executor_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool 3: Execute the generated SQL query against Snowflake database
    """
    if not state['is_relevant'] or state['error'] or not state['sql_query']:
        return state

    print(f"⚡ Executing SQL against Snowflake: {state['sql_query']}")

    try:
        # Execute the query against Snowflake
        query_result = execute_snowflake_query(state['sql_query'])

        if query_result['success']:
            # Process the results into a format suitable for response generation
            data = query_result['data']

            if not data:
                state['query_result'] = {
                    'message': 'No data found for the specified criteria'}
            elif len(data) == 1:
                # Single row result
                state['query_result'] = dict(data[0])
            else:
                # Multiple rows - convert to a more readable format
                if len(data) <= 10:  # For small result sets, include all data
                    state['query_result'] = {
                        'results': [dict(row) for row in data],
                        'row_count': len(data)
                    }
                else:
                    # For large result sets, provide summary
                    state['query_result'] = {
                        'sample_results': [dict(row) for row in data[:5]],
                        'total_rows': len(data),
                        'message': f'Showing first 5 of {len(data)} results'
                    }

            print(
                f"📊 Query executed successfully. Result: {state['query_result']}")

        else:
            # Query failed
            state['error'] = f"Snowflake query failed: {query_result['error']}"
            print(f"❌ Snowflake query failed: {query_result['error']}")

    except Exception as e:
        state['error'] = f"Error executing SQL: {str(e)}"
        print(f"❌ Error in SQL execution: {e}")

    # Mark completion
    state['sql_executed'] = True
    return state


def response_formatter_tool(state: WorkflowState) -> WorkflowState:
    """
    Tool 4: Format the query results into natural language response using LLM
    If formatting fails, return the raw dataframe/results
    """
    if not state['is_relevant'] or state['error']:
        return state

    print("📝 Formatting response to natural language")

    system_prompt = """You are a data analyst assistant that converts query results into clear, natural language responses.

Guidelines:
1. Be conversational and helpful
2. Include specific numbers with proper formatting (commas for thousands)
3. For comparisons, calculate and mention percentage changes
4. Use clear date references
5. Keep responses concise but informative
6. If showing multiple data points, organize them clearly

Examples:
- Single value: "The total transaction volume for last Friday was 123,456."
- Comparison: "Transaction volume increased from 110,000 on August 22nd to 123,456 on August 29th, representing a 12.2% increase."
- Multiple values: "Here are the recent transaction volumes: August 29th: 123,456, August 28th: 98,765"
"""

    try:
        # Prepare the data context for the LLM
        data_context = f"""
Original Query: {state['user_query']}
SQL Query Used: {state['sql_query']}
Query Results: {json.dumps(state['query_result'], indent=2)}
"""

        llm_response = call_llm(
            prompt=f"Convert this query result into a natural language response:\n\n{data_context}",
            system_prompt=system_prompt
        )

        # Check if the LLM response is valid and not an error
        if llm_response and not llm_response.startswith("Error:") and len(llm_response.strip()) > 0:
            state['final_response'] = llm_response.strip()
            print(f"✅ Final response: {state['final_response']}")
        else:
            raise ValueError("LLM returned invalid or empty response")

    except Exception as e:
        print(f"⚠️ Error in response formatting: {e}")
        print("📊 Returning raw data instead of formatted response")

        # Return the raw dataframe/results when formatting fails
        query_result = state.get('query_result', {})

        if isinstance(query_result, dict):
            if 'results' in query_result:
                # Multiple rows - convert to DataFrame-like string representation
                results = query_result['results']
                if results:
                    # Create a simple table representation
                    df_str = format_results_as_table(results)
                    state['final_response'] = f"Here are the query results:\n\n{df_str}"
                else:
                    state['final_response'] = "No data found for your query."
            elif 'sample_results' in query_result:
                # Large result set - show sample
                sample_results = query_result['sample_results']
                total_rows = query_result.get(
                    'total_rows', len(sample_results))
                df_str = format_results_as_table(sample_results)
                state['final_response'] = f"Here are the first {len(sample_results)} results out of {total_rows} total:\n\n{df_str}"
            elif 'message' in query_result:
                # No data message
                state['final_response'] = query_result['message']
            else:
                # Single row or simple result
                df_str = format_single_result(query_result)
                state['final_response'] = f"Query result:\n\n{df_str}"
        else:
            # Fallback for unexpected result format
            state['final_response'] = f"Query completed. Result: {str(query_result)}"

        print(f"📋 Raw data response: {state['final_response'][:200]}...")

    # Mark completion
    state['response_formatted'] = True
    return state


def format_results_as_table(results: List[Dict]) -> str:
    """
    Format query results as a simple table string
    """
    if not results:
        return "No data available"

    try:
        # Get all unique keys from all results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())

        headers = list(all_keys)

        # Create header row
        table_str = " | ".join(headers) + "\n"
        table_str += "-" * len(table_str) + "\n"

        # Add data rows
        for result in results:
            row_values = []
            for header in headers:
                value = result.get(header, "")
                # Format numbers with commas if they're large
                if isinstance(value, (int, float)) and abs(value) >= 1000:
                    value = f"{value:,}"
                row_values.append(str(value))
            table_str += " | ".join(row_values) + "\n"

        return table_str

    except Exception as e:
        # Fallback to simple string representation
        return "\n".join([str(result) for result in results])


def format_single_result(result: Dict) -> str:
    """
    Format a single result dictionary as a readable string
    """
    try:
        formatted_lines = []
        for key, value in result.items():
            # Format numbers with commas if they're large
            if isinstance(value, (int, float)) and abs(value) >= 1000:
                value = f"{value:,}"
            formatted_lines.append(f"{key}: {value}")

        return "\n".join(formatted_lines)

    except Exception as e:
        return str(result)


def workflow_orchestrator_tool(state: WorkflowState) -> WorkflowState:
    """
    Dynamic workflow orchestrator that decides execution strategy and tool sequence
    """
    print(f"🎯 Orchestrating dynamic workflow for: {state['user_query']}")

    system_prompt = """You are a dynamic workflow orchestrator for a data analysis chatbot.

Available tools and their purposes:
1. context_enhancer - Enhances follow-up questions with previous conversation context
2. intent_classifier - Determines if query is relevant to business data analysis
3. nl_to_sql - Converts natural language to SQL queries
4. sql_executor - Executes SQL queries against Snowflake database
5. response_formatter - Formats query results into natural language

Analyze the user query and determine the optimal execution path. Consider:
- Is this a follow-up question that needs context enhancement?
- Is the query obviously relevant to data analysis (skip intent classification)?
- What's the query complexity and expected processing needs?
- Are there any shortcuts or optimizations possible?

Respond with a JSON object containing the execution plan:
{
    "execution_path": ["tool1", "tool2", "tool3"],
    "skip_tools": ["tool_name"],
    "reasoning": "Why this path was chosen",
    "complexity": "simple|moderate|complex",
    "query_type": "greeting|irrelevant|simple_data|complex_data|follow_up",
    "optimizations": ["optimization1", "optimization2"]
}

Example paths:
- Simple data query: ["intent_classifier", "nl_to_sql", "sql_executor", "response_formatter"]
- Follow-up question: ["context_enhancer", "nl_to_sql", "sql_executor", "response_formatter"]
- Greeting/irrelevant: ["intent_classifier"]
- Obviously relevant complex query: ["nl_to_sql", "sql_executor", "response_formatter"]"""

    try:
        llm_response = call_llm(
            prompt=f"Plan execution path for: '{state['user_query']}'",
            system_prompt=system_prompt
        )

        # Try to parse the JSON response
        try:
            import json
            workflow_plan = json.loads(llm_response)
            state['context']['workflow_plan'] = workflow_plan
            state['context']['execution_path'] = workflow_plan.get(
                'execution_path', [])
            state['context']['skip_tools'] = workflow_plan.get(
                'skip_tools', [])

            print(f"📋 Dynamic Workflow Plan:")
            print(
                f"   Execution Path: {workflow_plan.get('execution_path', [])}")
            print(f"   Skip Tools: {workflow_plan.get('skip_tools', [])}")
            print(
                f"   Reasoning: {workflow_plan.get('reasoning', 'No reasoning provided')}")
            print(
                f"   Complexity: {workflow_plan.get('complexity', 'unknown')}")

        except json.JSONDecodeError:
            # Fallback to default path
            default_path = ["intent_classifier", "nl_to_sql",
                            "sql_executor", "response_formatter"]
            state['context']['workflow_plan'] = {
                "execution_path": default_path,
                "skip_tools": [],
                "reasoning": "Default path due to parsing error",
                "complexity": "simple",
                "query_type": "simple_data"
            }
            state['context']['execution_path'] = default_path
            state['context']['skip_tools'] = []
            print(
                f"⚠️ Using default workflow path due to parsing error: {default_path}")

    except Exception as e:
        state['error'] = f"Error in workflow orchestration: {str(e)}"
        print(f"❌ Error in workflow orchestration: {e}")

    # Mark completion
    state['orchestration_completed'] = True
    return state


print("✅ All tool functions defined!")

# %% [markdown]
# ## 5. LangGraph Workflow Setup
#
# Now we'll create the LangGraph workflow that orchestrates our tools

# %%


def create_dynamic_workflow():
    """
    Create a dynamic LangGraph workflow that adapts based on orchestrator decisions
    """
    # Create a new state graph
    workflow = StateGraph(WorkflowState)

    # Add all available tool nodes
    workflow.add_node("context_enhancer", context_enhancer_tool)
    workflow.add_node("workflow_orchestrator", workflow_orchestrator_tool)
    workflow.add_node("intent_classifier", intent_classifier_tool)
    workflow.add_node("nl_to_sql", nl_to_sql_tool)
    workflow.add_node("sql_executor", sql_executor_tool)
    workflow.add_node("response_formatter", response_formatter_tool)

    # Always start with context enhancement, then orchestration
    workflow.set_entry_point("context_enhancer")
    workflow.add_edge("context_enhancer", "workflow_orchestrator")

    # Dynamic routing function based on orchestrator decisions
    def dynamic_router(state: WorkflowState) -> str:
        """Route to next tool based on orchestrator's execution plan"""
        if state['error']:
            return END

        execution_path = state['context'].get('execution_path', [])
        skip_tools = state['context'].get('skip_tools', [])

        if not execution_path:
            print("⚠️ No execution path found, using default path")
            execution_path = ["intent_classifier", "nl_to_sql",
                              "sql_executor", "response_formatter"]

        # Determine what tools have been completed
        completed_tools = set()
        if state.get('context_enhanced', False):
            completed_tools.add('context_enhancer')
        if state.get('orchestration_completed', False):
            completed_tools.add('workflow_orchestrator')
        if state.get('intent_classified', False):
            completed_tools.add('intent_classifier')
        if state.get('sql_generated', False):
            completed_tools.add('nl_to_sql')
        if state.get('sql_executed', False):
            completed_tools.add('sql_executor')
        if state.get('response_formatted', False):
            completed_tools.add('response_formatter')

        print(f"🔍 Completed tools: {completed_tools}")
        print(f"🎯 Planned execution path: {execution_path}")
        print(f"⏭️ Skip tools: {skip_tools}")

        # Find next tool to execute
        for tool in execution_path:
            if tool not in completed_tools and tool not in skip_tools:
                print(f"➡️ Dynamic routing to: {tool}")
                return tool

        # Special handling for irrelevant queries
        if state.get('intent_classified', False) and not state.get('is_relevant', True):
            print("🚫 Query not relevant, ending workflow")
            return END

        # If all tools completed or no more tools, end workflow
        print("✅ All planned tools completed, ending workflow")
        return END

    # Add dynamic conditional edges from orchestrator
    workflow.add_conditional_edges(
        "workflow_orchestrator",
        dynamic_router,
        {
            "context_enhancer": "context_enhancer",
            "intent_classifier": "intent_classifier",
            "nl_to_sql": "nl_to_sql",
            "sql_executor": "sql_executor",
            "response_formatter": "response_formatter",
            END: END
        }
    )

    # Add dynamic routing from each tool
    for tool_name in ["intent_classifier", "nl_to_sql", "sql_executor"]:
        workflow.add_conditional_edges(
            tool_name,
            dynamic_router,
            {
                "context_enhancer": "context_enhancer",
                "intent_classifier": "intent_classifier",
                "nl_to_sql": "nl_to_sql",
                "sql_executor": "sql_executor",
                "response_formatter": "response_formatter",
                END: END
            }
        )

    # Response formatter always ends
    workflow.add_edge("response_formatter", END)

    # Compile the workflow
    app = workflow.compile()
    return app


# Create our dynamic workflow
chatbot_workflow = create_dynamic_workflow()
print("✅ Dynamic LangGraph workflow created successfully!")

# %% [markdown]
# ## 6. Main Chatbot Function
#
# This is the main function that processes user queries

# %%


def process_query(user_query: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Main function to process a user query through our LangGraph workflow

    Args:
        user_query: The user's natural language query
        session_id: Session identifier for context management

    Returns:
        Dictionary containing the final response and execution details
    """
    print(f"\n🚀 Processing query: '{user_query}'")
    print("=" * 50)

    # Initialize state
    initial_state = initialize_state(session_id, user_query)

    try:
        # Run the workflow
        final_state = chatbot_workflow.invoke(initial_state)

        # Prepare response
        response = {
            "query": user_query,
            "original_query": final_state.get('original_query', user_query),
            "enhanced_query": final_state.get('user_query', user_query),
            "is_followup": final_state.get('is_followup', False),
            "response": final_state.get('final_response', 'No response generated'),
            "session_id": session_id,
            "success": not bool(final_state.get('error')),
            "error": final_state.get('error', ''),
            "execution_details": {
                "intent": final_state.get('intent', ''),
                "is_relevant": final_state.get('is_relevant', False),
                "sql_query": final_state.get('sql_query', ''),
                "query_result": final_state.get('query_result', {})
            }
        }

        # Save conversation turn to session history (only if successful and relevant)
        if response['success'] and final_state.get('is_relevant', False):
            conversation_turn = ConversationTurn(
                query=final_state.get('original_query', user_query),
                response=final_state.get('final_response', ''),
                sql_query=final_state.get('sql_query', ''),
                query_result=final_state.get('query_result', {}),
                timestamp=datetime.now(),
                intent=final_state.get('intent', '')
            )
            session_manager.add_turn(session_id, conversation_turn)
            print(f"💾 Saved conversation turn to session {session_id}")

        print("=" * 50)
        print(f"✅ Final Response: {response['response']}")
        if response['is_followup']:
            print(
                f"🔗 Follow-up detected - Enhanced query: {response['enhanced_query']}")

        return response

    except Exception as e:
        error_response = {
            "query": user_query,
            "response": "I encountered an error while processing your request.",
            "session_id": session_id,
            "success": False,
            "error": str(e),
            "execution_details": {}
        }

        print(f"❌ Error processing query: {e}")
        return error_response


def clear_session(session_id: str):
    """
    Clear conversation history for a session
    """
    session_manager.clear_session(session_id)
    print(f"🗑️ Cleared conversation history for session {session_id}")


def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Get conversation history for a session
    """
    if session_id not in session_manager.sessions:
        return []

    history = []
    for turn in session_manager.sessions[session_id]:
        history.append({
            "query": turn.query,
            "response": turn.response,
            "timestamp": turn.timestamp.isoformat(),
            "intent": turn.intent
        })
    return history


def demo_followup_conversation():
    """
    Demonstrate follow-up conversation capabilities
    """
    print("🎭 FOLLOW-UP CONVERSATION DEMO")
    print("=" * 50)

    session_id = f"followup_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Sequence of queries that build on each other
    demo_queries = [
        "What was the total transaction volume last month?",
        "How does that compare to the previous month?",
        "Which customer segment contributed most to those sales?",
        "What about premium customers specifically?",
        "Show me the top products they bought"
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        result = process_query(query, session_id)
        print(f"Response: {result['response']}")
        if result['is_followup']:
            print(
                f"🔗 Detected as follow-up. Enhanced to: {result['enhanced_query']}")
        print("-" * 40)

    print(f"\n📚 Session History:")
    history = get_session_history(session_id)
    for i, turn in enumerate(history, 1):
        print(f"{i}. {turn['query']} → {turn['response'][:100]}...")


print("✅ Main chatbot function with follow-up support ready!")

# %% [markdown]
# ## 7. Configuration and Setup Instructions
#
# Before running the chatbot, set up your environment variables

# %%


def setup_instructions():
    """
    Display setup instructions for Snowflake and GenAI Platform configuration
    """
    print("🔧 SETUP INSTRUCTIONS")
    print("=" * 50)
    print("\n1. Update the .env file with your actual credentials:")
    print("   - Edit the .env file in this directory")
    print("   - Replace placeholder values with your actual credentials")
    print("   - The .env file is automatically loaded by the script")
    print("\n2. Required environment variables in .env:")
    print("   CFG_USECASE_ID=your-usecase-id")
    print("   CFG_EXPERIMENT_NAME=your-experiment-name")
    print("   CFG_EXPERIMENT_DESC=your-experiment-description")
    print("   SNOWFLAKE_ACCOUNT=your-account.snowflakecomputing.com")
    print("   SNOWFLAKE_USER=your-username")
    print("   SNOWFLAKE_PASSWORD=your-password")
    print("   SNOWFLAKE_WAREHOUSE=COMPUTE_WH")
    print("   SNOWFLAKE_DATABASE=your-database")
    print("   SNOWFLAKE_SCHEMA=PUBLIC")
    print("   SNOWFLAKE_ROLE=ACCOUNTADMIN")
    print("\n3. Update the database_schema.txt file with your actual table schemas:")
    print("   - Edit database_schema.txt with your 3 main tables")
    print("   - Include table names, column names, data types, and descriptions")
    print("   - Follow the format shown in the sample file")
    print("\n4. Ensure your Snowflake tables exist and match the schema file")
    print("\n5. Test your connection by running a simple query")
    print("\n6. Run the test queries below to verify everything works")

    # Display current configuration status
    print("\n📋 CURRENT CONFIGURATION STATUS:")
    print("=" * 50)

    try:
        # Test if configuration is valid by attempting to create instances
        test_usecase_id = os.getenv("CFG_USECASE_ID")
        print(
            f"GenAI Platform Usecase ID: {'✅ Set' if test_usecase_id else '❌ Not set'}")

        # Test Snowflake config
        try:
            test_config = SnowflakeConfig()
            print("✅ Snowflake Configuration: All required fields set")
        except ValueError as e:
            print(f"❌ Snowflake Configuration: {e}")

        # Test schema file
        try:
            test_schema = DatabaseSchema()
            print(
                f"✅ Database Schema: Loaded {len(test_schema.get_table_names())} tables")
        except Exception as e:
            print(f"❌ Database Schema: {e}")

    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        print("\n⚠️  Please update your .env file and database_schema.txt before running the chatbot!")


setup_instructions()

# %% [markdown]
# ## 8. Testing and Demonstration
#
# Test the chatbot with various queries (requires proper Snowflake setup)

# %%
# Test cases - uncomment when your Snowflake connection is ready
test_queries = [
    # Simple transaction queries
    "What was the transaction volume last Friday?",
    "How does that compare to the previous Friday?",
    "What's the total transaction volume this month?",

    # Multi-table customer analysis
    "Which customer segment has the highest sales?",
    "Show me sales by customer segment for last month",
    "How many premium customers made transactions this week?",

    # Multi-table product analysis
    "What are the top 5 products by sales volume?",
    "Which product category performs best?",
    "Show me sales by product category and customer segment",

    # Complex multi-table queries
    "Compare sales performance between online and store channels",
    "Which customers bought the most expensive products?",
    "Show me monthly trends for each customer segment",

    # Irrelevant queries (should be rejected)
    "What's the weather like today?",
    "Tell me about your favorite movie",
    "How do I reset my password?"
]

# Follow-up conversation test sequence
followup_test_sequence = [
    "What was the total sales volume last month?",
    "How does that compare to the previous month?",
    "Which customer segment contributed most to those sales?",
    "What about premium customers?",
    "Show me their top 3 product categories",
    "And how does that compare to gold customers?",
    "What's the average transaction value for both segments?"
]


def run_tests():
    """
    Run test queries - only call this when Snowflake is properly configured
    """
    print("🧪 Testing the chatbot with various queries:")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        try:
            result = process_query(query, f"test_session_{i}")
            print(f"Response: {result['response']}")
            print(f"Success: {result['success']}")
            if result['error']:
                print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Error running test: {e}")
        print("-" * 40)


def run_followup_tests():
    """
    Test follow-up conversation capabilities
    """
    print("🔗 Testing follow-up conversation capabilities:")
    print("=" * 60)

    session_id = f"followup_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for i, query in enumerate(followup_test_sequence, 1):
        print(f"\n📝 Query {i}: {query}")
        try:
            result = process_query(query, session_id)
            print(f"Response: {result['response']}")
            print(f"Follow-up: {result['is_followup']}")
            if result['is_followup']:
                print(f"Enhanced: {result['enhanced_query']}")
            if result['error']:
                print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Error running test: {e}")
        print("-" * 40)

    # Show conversation history
    print(f"\n📚 Final Conversation History:")
    history = get_session_history(session_id)
    for i, turn in enumerate(history, 1):
        print(f"{i}. Q: {turn['query']}")
        print(f"   A: {turn['response'][:100]}...")

# Uncomment the line below to run tests (ensure Snowflake is configured first)
# run_tests()

# Uncomment the line below to test follow-up conversations
# run_followup_tests()


print("✅ Test functions ready!")
print("Uncomment 'run_tests()' above when your Snowflake connection is configured")
print("Uncomment 'run_followup_tests()' to test follow-up conversation capabilities")

# %% [markdown]
# ## 9. Interactive Demo
#
# Run this cell to interact with the chatbot

# %%


def interactive_demo():
    """
    Interactive demo function - uncomment and run to chat with the bot
    """
    print("🤖 LangGraph Chatbot Demo")
    print("Ask me about transaction volumes, comparisons, or data queries!")
    print("Type 'quit' to exit\n")

    session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Process the query
            result = process_query(user_input, session_id)
            print(f"Bot: {result['response']}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

# Uncomment the line below to run the interactive demo
# interactive_demo()


print("✅ Interactive demo function ready!")
print("Uncomment the last line in the cell above to run the interactive demo")

# %% [markdown]
# ## 10. Summary and Next Steps
#
# This POC demonstrates a production-ready LangGraph chatbot that:
#
# ✅ **Intent Classification**: Uses LLM to determine if queries are relevant to business data analysis
# ✅ **Follow-up Question Support**: Maintains conversation context and handles follow-up questions intelligently
# ✅ **Context Enhancement**: Automatically detects and enhances follow-up queries with previous conversation context
# ✅ **Multi-Table Schema Support**: Loads database schema from external text file
# ✅ **Natural Language to SQL**: Uses LLM to convert user queries to complex Snowflake SQL with JOINs
# ✅ **SQL Execution**: Executes queries against real Snowflake database
# ✅ **Response Formatting**: Uses LLM to convert results back to natural language
# ✅ **Workflow Orchestration**: Uses LangGraph to manage the execution flow
# ✅ **Real Database Integration**: Connects to Snowflake with proper error handling
#
# ### Key Features:
# - **Conversation Memory**: Maintains session-based conversation history for natural follow-up questions
# - **Context-Aware Processing**: Automatically detects and enhances follow-up questions like "What about premium customers?"
# - **Multi-Table Support**: Handles complex queries across multiple related tables
# - **Schema File Integration**: Loads table schemas from database_schema.txt file for easy maintenance
# - **Intelligent JOIN Generation**: Automatically creates appropriate JOIN queries based on relationships
# - **Real LLM Integration**: All tools use GenAI Platform for intelligent processing
# - **Snowflake Integration**: Direct connection to Snowflake data warehouse
# - **Session Management**: Tracks conversation history per session with configurable history limits
# - **Robust Error Handling**: Proper error handling for database and LLM failures
# - **Configurable**: Environment variable and schema file based configuration
# - **Production Ready**: Real database queries with proper connection management
#
# ### Follow-up Question Examples:
# - "What was the sales volume last month?" → "How does that compare to the previous month?"
# - "Which customer segment has highest sales?" → "What about premium customers specifically?"
# - "Show me top products" → "And how do they perform in different regions?"
#
# ### Next Steps for Production:
# 1. **Persistent Session Storage**: Add database-backed session storage for long-term memory
# 2. **Security**: Add input validation and SQL injection protection
# 3. **Caching**: Add response caching for common queries
# 4. **Monitoring**: Add logging and monitoring for production use
# 5. **Authentication**: Add user authentication and authorization
# 6. **Rate Limiting**: Add rate limiting for API calls

print("🎉 LangGraph Chatbot POC Complete!")
print("The chatbot is ready to use. Try running the test queries or interactive demo!")
