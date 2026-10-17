import ast
import sys

def analyze_code(source_code, filepath="<source>"):
    try:
        tree = ast.parse(source_code, filename=filepath)
    except Exception as e:
        return f"Error parsing {filepath}: {e}"

    output = [f"### Code Architecture Map: `{filepath}`\n"]
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            output.append(f"- **Class**: `{node.name}`")
            doc = ast.get_docstring(node)
            if doc:
                output.append(f"  - Docstring: {doc.split(chr(10))[0]}")
            
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    args = [a.arg for a in item.args.args if a.arg != 'self']
                    output.append(f"  - **Method**: `{prefix}{item.name}({', '.join(args)})`")
                    
        # Extract Top-Level Functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args = [a.arg for a in node.args.args]
            output.append(f"- **Function**: `{prefix}{node.name}({', '.join(args)})`")
            doc = ast.get_docstring(node)
            if doc:
                output.append(f"  - Docstring: {doc.split(chr(10))[0]}")
    
    return "\n".join(output)

def analyze_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""
    
    summary = analyze_code(code, filepath)
    print(summary)
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_ast.py <path_to_local_file.py>")
        sys.exit(1)
    analyze_file(sys.argv[1])