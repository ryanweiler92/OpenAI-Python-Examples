import os
import json
from openai import OpenAI
from typing import Dict, List, Any


def test_model_role_compatibility():
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        return {"error": "OPENAI_KEY environment variable not set"}

    client = OpenAI(api_key=api_key)

    # Models to test
    models = ["o1", "o1-mini", "o3", "o3-mini"]

    # Roles to test
    roles = ["system", "developer"]

    # Results dictionary
    results = {}

    for model in models:
        results[model] = {}

        for role in roles:
            print(f"\nTesting {model} with {role} role...")

            messages = [
                {"role": role, "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, what can you do?"},
            ]

            try:
                response = client.chat.completions.create(
                    model=model, messages=messages
                )
                success = True
                response_content = response.choices[0].message.content
                error_message = None
            except Exception as e:
                success = False
                response_content = None
                error_message = str(e)
                print(f"Error: {e}")

            # Control test with user role
            if not success:
                try:
                    print(f"  Testing {model} with fallback to user role...")
                    user_messages = [
                        {
                            "role": "user",
                            "content": f"{role.upper()}: You are a helpful assistant.",
                        },
                        {"role": "user", "content": "Hello, what can you do?"},
                    ]

                    user_response = client.chat.completions.create(
                        model=model, messages=user_messages
                    )
                    user_success = True
                    user_response_content = user_response.choices[0].message.content
                except Exception as e:
                    user_success = False
                    user_response_content = None
                    print(f"  Error with user role fallback: {e}")
            else:
                user_success = None
                user_response_content = None

            results[model][role] = {
                "supported": success,
                "response": response_content,
                "error": error_message,
                "user_fallback_works": user_success,
                "user_fallback_response": user_response_content,
            }

    return results


if __name__ == "__main__":
    results = test_model_role_compatibility()

    # Print summary table
    print("\n\n=== COMPATIBILITY SUMMARY ===")
    header = "Model".ljust(10) + "System Role".ljust(15) + "Developer Role".ljust(15)
    print(header)
    print("-" * len(header))

    for model in results:
        if isinstance(results[model], dict):
            system_support = (
                "✓" if results[model].get("system", {}).get("supported", False) else "✗"
            )
            dev_support = (
                "✓"
                if results[model].get("developer", {}).get("supported", False)
                else "✗"
            )
            print(f"{model.ljust(10)}{system_support.ljust(15)}{dev_support.ljust(15)}")

    # Save detailed results to JSON file
    with open("openai_model_role_compatibility.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to openai_model_role_compatibility.json")
