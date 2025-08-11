from datetime import datetime
import json, requests, getpass


class Property:
    def __init__(self, property_id: int, location: str, type: str, price_per_night: float, features: list[str], tags: list[str]):
        self.property_id = property_id
        self.location = location
        self.type = type
        self.price_per_night = price_per_night
        self.features = features
        self.tags = tags

    def __str__(self):
        feature_list = "Features: "
        for feature in self.features:
            feature_list += f"\t{feature}\n"
        tag_list = "Tags: "
        for tag in self.tags:
            tag_list += f"\t{tag}\n"
        return (f"--------------------------------------"
                f"ID: {self.property_id}"
                f"Location: ${self.location}\n"
                f"Type: ${self.type}\n"
                f"Nightly Price: ${self.price_per_night}\n"
                f"Features: ${feature_list}\n"
                f"Tags: ${tag_list}\n"
                f"--------------------------------------")

class User:
    def __init__(self, user_id: int, name: str, group_size: int, preferred_environment: list[str],
                 budget_range: tuple[int, int], travel_date: datetime = datetime.now()):
        self.user_id = user_id
        self.name = name
        self.group_size = group_size
        self.preferred_environment = preferred_environment
        self.budget_range = budget_range
        self.travel_date = travel_date

def load_from_file() -> tuple[list[Property], list[User]]:
    with open("properties.json", "r") as file:
        temp_properties = json.load(file)
    if type(temp_properties) == list:
        property_result = [Property(prop['property_id'], prop['location'], prop['type'], prop['price_per_night'],
                                    prop['features'],prop['tags']) for prop in temp_properties]
    else:
        property_result = [Property[temp_properties['property_id'], temp_properties['location'], temp_properties['type'],
                                    temp_properties['price_per_night'],temp_properties['features'],temp_properties['tags']]]
    with open("users.json", "r") as file:
        temp_users = json.load(file)
    if type(temp_users) == list:
        user_result = [User(user['user_id'], user['name'], user['group_size'], user['preferred_environment'],
                            user['budget_range'], datetime.strptime(user['travel_date'], "%Y-%m-%d %H:%M:%S.%f"))
                       for user in temp_users]
    else:
        user_result = [User(temp_users['user_id'], temp_users['name'], temp_users['group_size'],
                            temp_users['preferred_environment'], temp_users['budget_range'],
                            datetime.strptime(temp_users['travel_date'], "%Y-%m-%d %H:%M:%S.%f"))]
    return property_result, user_result

def write_to_file(property_result: list[Property], user_result: list[User]) -> bool:
    try:
        with open("properties.json", "w") as file:
            json.dump(property_result, file)
        with open("users.json", "w") as file:
            json.dump(user_result, file)
        return True
    except FileNotFoundError:
        return False

def main():
    property_result, user_result = load_from_file()
    exit()

if  __name__ == "__main__":
    main()

    ###############################################################################################
    # API Call Code from JSON_Tutorial, to be paraphrased.
    # API Key is implemented without the requirement of user input. To be removed before submission
    ###############################################################################################

    # --- Load properties ---
    with open("properties.json", "r") as f:
        properties = json.load(f)

    # --- Get API key safely (won't echo in Colab) ---
    # API_KEY = getpass.getpass("Enter your OpenRouter API key: ").strip()
    API_KEY = "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"
    # Pick a DeepSeek model available on OpenRouter.
    # Common ones include (names can change over time):
    # - "deepseek/deepseek-chat"
    # - "deepseek/deepseek-r1"
    MODEL = "deepseek/deepseek-chat-v3-0324:free"

    URL = "https://openrouter.ai/api/v1/chat/completions"
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    SYSTEM_PROMPT = (
        "You are an assistant for an Airbnb-like vacation property search. "
        "Given a list of properties (JSON) and a user request, return either: "
        "(1) a JSON object with `tags` (list of strings) inferred from the request, and "
        "(2) optionally `property_ids` (list of integers) that might match. "
        "Only return valid JSON."
    )

    def llm_search(user_prompt: str, model: str = MODEL) -> dict:
        """Call OpenRouter and return a parsed dict. On error, return {'error': '...'}."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "PROPERTIES:\n" + json.dumps(properties) +
                        "\n\nUSER REQUEST:\n" + user_prompt +
                        "\n\nRespond with JSON: {\"tags\": [...], \"property_ids\": [...]} (property_ids optional)"
                    ),
                },
            ],
            # Safety/time controls (optional):
            "temperature": 0.2,
        }
        try:
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
            # Helpful debug if something goes wrong
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}", "details": r.text}
            data = r.json()
            # Expected shape: data["choices"][0]["message"]["content"]
            msg = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not msg:
                return {"error": "No content in response", "details": data}
            # Try to parse JSON content the model returned
            try:
                return json.loads(msg)
            except json.JSONDecodeError:
                # If the model included extra text, try to extract JSON loosely
                # (basic fallback—students can improve later)
                start = msg.find("{")
                end = msg.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(msg[start:end+1])
                    except json.JSONDecodeError:
                        return {"error": "Model returned non-JSON content", "raw": msg}
                return {"error": "Model returned non-JSON content", "raw": msg}
        except Exception as e:
            return {"error": "Request failed", "details": str(e)}

    # --- Mini chatbot loop ---
    print("Vacation Property Bot (type 'exit' to quit)")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            print("Bot: Have a great vacation!")
            break
        result = llm_search(prompt)
        if "error" in result:
            print("Bot (error):", result["error"])
            if "details" in result:
                print("Details:", result["details"])
            elif "raw" in result:
                print("Raw output:", result["raw"])
        else:
            tags = result.get("tags", [])
            prop_ids = result.get("property_ids", [])
            print("Bot: tags =", tags)
            print("Bot: property_ids =", prop_ids)
