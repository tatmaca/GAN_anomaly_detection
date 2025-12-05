import sys, platform

print("sys.executable:", sys.executable)
print("platform:", platform.system(), platform.machine())
print("sys.path:")
for p in sys.path:
    print("   ", p)

print("\nTrying import...")

try:
    import matplotlib # <-- change this
    print("Imported OK from:", getattr(matplotlib, "__file__", "<no __file__>"))
except Exception as e:
    print("FAILED TO IMPORT:")
    print(repr(e))
