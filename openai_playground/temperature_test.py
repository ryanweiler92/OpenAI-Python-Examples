import os
import json
from openai import OpenAI
from typing import Dict, List, Any


def test_temperature_support():
    """Test which OpenAI models support temperature settings"""
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        return {"error": "OPENAI_KEY environment variable not set"}

    client = OpenAI(api_key=api_key)

    # Models to test
    models = ["o1", "o1-mini", "o3-mini", "gpt-4", "gpt-3.5-turbo"]

    # Temperature values to test
    temperatures = [0.0, 0.5, 1.0]

    # Results dictionary
    results = {}

    # Basic message for all tests
    messages = [{"role": "user", "content": "Tell me a short joke."}]

    for model in models:
        print(f"\nTesting temperature support for {model}...")
        results[model] = {}

        # First test with no temperature specified
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            results[model]["default"] = {
                "supported": True,
                "response": response.choices[0].message.content,
                "error": None,
            }
            print(f"  Default (no temperature): Success")
        except Exception as e:
            error_msg = str(e)
            results[model]["default"] = {
                "supported": False,
                "response": None,
                "error": error_msg,
            }
            # If model doesn't exist, skip testing temperatures
            if "does not exist" in error_msg or "not found" in error_msg:
                print(f"  Model not available: {error_msg}")
                continue

        # Test each temperature value
        for temp in temperatures:
            try:
                print(f"  Testing temperature={temp}...")
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=temp
                )
                results[model][f"temp_{temp}"] = {
                    "supported": True,
                    "response": response.choices[0].message.content,
                    "error": None,
                }
                print(f"  Temperature {temp}: Success")
            except Exception as e:
                error_msg = str(e)
                results[model][f"temp_{temp}"] = {
                    "supported": False,
                    "response": None,
                    "error": error_msg,
                }
                print(f"  Temperature {temp}: Failed - {error_msg}")

    return results


if __name__ == "__main__":
    results = test_temperature_support()

    # Print summary table
    print("\n\n=== TEMPERATURE SUPPORT SUMMARY ===")
    header = (
        "Model".ljust(15)
        + "Default".ljust(10)
        + "Temp 0.0".ljust(12)
        + "Temp 0.5".ljust(12)
        + "Temp 1.0".ljust(12)
    )
    print(header)
    print("-" * len(header))

    for model in results:
        if "error" in results:
            continue

        default = (
            "✓" if results[model].get("default", {}).get("supported", False) else "✗"
        )
        temp_0 = (
            "✓" if results[model].get("temp_0.0", {}).get("supported", False) else "✗"
        )
        temp_05 = (
            "✓" if results[model].get("temp_0.5", {}).get("supported", False) else "✗"
        )
        temp_1 = (
            "✓" if results[model].get("temp_1.0", {}).get("supported", False) else "✗"
        )

        print(
            f"{model.ljust(15)}{default.ljust(10)}{temp_0.ljust(12)}{temp_05.ljust(12)}{temp_1.ljust(12)}"
        )

    # Save detailed results to JSON file
    with open("openai_temperature_support.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to openai_temperature_support.json")
