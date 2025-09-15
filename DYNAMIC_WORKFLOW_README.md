# Dynamic Workflow System

## 🎯 Overview

The LangGraph chatbot now features a **truly dynamic workflow** that adapts its execution path based on the user query analysis. The `workflow_orchestrator_tool` acts as an intelligent router that decides which tools to execute and in what order.

## 🔧 How It Works

### 1. **Dynamic Orchestrator**
The `workflow_orchestrator_tool` analyzes each query and creates an execution plan:

```json
{
    "execution_path": ["intent_classifier", "nl_to_sql", "sql_executor", "response_formatter"],
    "skip_tools": ["context_enhancer"],
    "reasoning": "Simple data query, no context needed",
    "complexity": "simple",
    "query_type": "simple_data"
}
```

### 2. **Adaptive Routing**
The workflow dynamically routes between tools based on:
- **Query type** (greeting, data query, follow-up, etc.)
- **Complexity level** (simple, moderate, complex)
- **Context needs** (follow-up questions vs new queries)
- **Optimization opportunities** (skip unnecessary steps)

### 3. **Smart Optimizations**

#### **Path Examples:**

**Simple Data Query:**
```
Query: "What was the sales volume yesterday?"
Path: intent_classifier → nl_to_sql → sql_executor → response_formatter
Skips: context_enhancer (not needed for new query)
```

**Follow-up Question:**
```
Query: "How does that compare to last month?"
Path: context_enhancer → nl_to_sql → sql_executor → response_formatter  
Skips: intent_classifier (obviously relevant)
```

**Greeting/Irrelevant:**
```
Query: "Hello, how are you?"
Path: intent_classifier → END
Skips: All data processing tools
```

**Complex Analysis:**
```
Query: "Show me sales trends by customer segment over the last 6 months"
Path: intent_classifier → nl_to_sql → sql_executor → response_formatter
Optimizations: Enhanced SQL generation for complex queries
```

## 🚀 Benefits

### **1. Performance Optimization**
- **Faster responses** for simple queries
- **Skip unnecessary tools** based on query type
- **Reduced LLM calls** for obvious cases

### **2. Better User Experience**
- **Context-aware** follow-up handling
- **Appropriate responses** for different query types
- **Intelligent routing** based on user intent

### **3. Flexibility**
- **Easy to extend** with new tools
- **Configurable paths** for different scenarios
- **LLM-driven decisions** adapt to new patterns

## 🔍 Tool Completion Tracking

Each tool marks its completion in the state:
- `context_enhanced`: Context enhancement completed
- `orchestration_completed`: Workflow planning completed  
- `intent_classified`: Intent classification completed
- `sql_generated`: SQL generation completed
- `sql_executed`: SQL execution completed
- `response_formatted`: Response formatting completed

## 🎛️ Configuration

The orchestrator can be tuned by modifying the system prompt to:
- **Add new query types**
- **Define new optimization paths**
- **Adjust routing logic**
- **Include new tools in execution plans**

## 🧪 Testing

Run the test script to see the dynamic workflow in action:

```bash
python test_dynamic_workflow.py
```

This will demonstrate different execution paths for various query types.

## 📊 Monitoring

The system provides detailed logging of:
- **Planned execution paths**
- **Tools being skipped**
- **Routing decisions**
- **Completion tracking**

Look for these log messages:
- `🎯 Dynamic routing to: [tool_name]`
- `🔍 Completed tools: [list]`
- `⏭️ Skip tools: [list]`
- `✅ All planned tools completed`

## 🔮 Future Enhancements

Potential improvements:
1. **Learning from usage patterns** to optimize paths
2. **User preference-based routing** 
3. **Performance metrics** to tune execution paths
4. **A/B testing** different routing strategies
5. **Tool-specific optimizations** based on query analysis

The dynamic workflow system makes the chatbot more intelligent, efficient, and adaptable to different user needs!