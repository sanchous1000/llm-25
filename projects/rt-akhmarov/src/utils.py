import yaml
from pathlib import Path
from typing import Any, Dict
from langfuse import Langfuse

def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.absolute()}")
    
    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config if config is not None else {}
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            raise

def sync_prompts_to_langfuse(config_path: str, lf: Langfuse | None) -> None:
    config = load_config(config_path)

    prompts = config.get("prompts", [])
    if not prompts:
        return

    if lf is None or not hasattr(lf, "create_prompt"):
        print("⚠️  Langfuse client not configured; skipping prompt sync.")
        return

    for p_data in prompts:
        try:
            print(f"📦 Syncing prompt to registry: {p_data['name']}")
            lf.create_prompt(
                name=p_data["name"],
                type=p_data["type"],
                prompt=p_data["config"]["prompt"],
                labels=p_data.get("labels", ["production"]),
            )
        except AttributeError:
            print("⚠️  Langfuse client appears uninitialized; skipping remaining prompts.")
            break
        except Exception as exc:
            print(f"Failed to create prompt {p_data.get('name')}: {exc}")

    print("✅ Prompt sync complete (skipped if client uninitialized).")
    if hasattr(lf, "flush"):
        try:
            lf.flush()
        except Exception:
            pass