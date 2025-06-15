from gpu.host import DeviceContext, DeviceBuffer
from sys import has_accelerator

def main():
    @parameter
    if not has_accelerator():
        print("No GPU found")
        return
    var s = String("{} is {}").format("Mojo", "Fire")
    var s1 = String("Hello {0}!").format("World")  # Hello World!
    var s2 = String("{0} plus {1} equals {2}").format(1, 2, 3)  # 1 plus 2 equals 3
    var s4 = String("{!r}").format("test")  # "'test'"
    print(s, s1, s2, s4)
    ctx = DeviceContext()
    print("GPU found", ctx.name())
