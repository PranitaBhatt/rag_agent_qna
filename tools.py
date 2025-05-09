def calculator_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Invalid expression."

def dictionary_tool(word):
    return f"{word} means: [Placeholder definition, integrate dictionary API if needed]"
