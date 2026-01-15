import os
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()

if not langfuse.auth_check():
    raise RuntimeError("Langfuse auth_check() failed: проверь LANGFUSE_* переменные")

with langfuse.start_as_current_observation(as_type="span", name="smoke-test") as span:
    span.update(input={"ping": 1}, output="ok")

langfuse.flush()
print("OK: Langfuse connected, event sent")
