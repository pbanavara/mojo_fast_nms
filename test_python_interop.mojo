from python import Python

fn main() raises:
    print("Testing Python interop...")
    
    # Test basic Python evaluation
    var result = Python.evaluate("2 + 2")
    print("Python result:", result)
    
    # Test importing a Python module
    var math = Python.import_module("math")
    var pi = math.pi
    print("Python math.pi:", pi)
    
    # Test calling a Python function
    var sqrt_result = math.sqrt(16.0)
    print("Python math.sqrt(16):", sqrt_result)
    
    print("Python interop test completed successfully!") 