import google.generativeai as genai
from google.generativeai import protos

print("--- genai attributes ---")
print(dir(genai))

print("\n--- protos attributes (Tool related) ---")
print([x for x in dir(protos) if 'Tool' in x or 'Search' in x or 'Retrieval' in x])

print("\n--- protos.Tool fields ---")
try:
    print(protos.Tool.DESCRIPTOR.fields_by_name.keys())
except:
    pass
