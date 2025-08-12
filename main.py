from datetime import datetime
import json, requests, getpass


class Property:
    def __init__(self, property_id: int, location: str, loc_type: str, price_per_night: float, features_list: list[str], tags_list: list[str]):
        self.property_id = property_id
        self.location = location
        self.type = loc_type
        self.price_per_night = price_per_night
        if type(features_list) is str:
            self.features = [features_list]
        else:
            self.features = features_list
        if type(tags_list) is str:
            self.tags = [tags_list]
        else:
            self.tags = tags_list

    def __str__(self):
        feature_list = "Features: "
        for feature in self.features:
            feature_list += f"\n\t{feature}"
        tag_list = "Tags: "
        for tag in self.tags:
            tag_list += f"\n\t{tag}"
        return (f"--------------------------------------\n"
                f"ID: {self.property_id}"
                f"Location: {self.location}\n"
                f"Type: {self.type}\n"
                f"Nightly Price: {self.price_per_night}\n"
                f"{feature_list}\n"
                f"{tag_list}\n"
                f"--------------------------------------\n\n")
    def update_id(self, property_id: int):
        self.property_id = property_id
    def update_type(self, property_type: str):
        self.type = property_type
    def update_price_per_night(self, price_per_night: float):
        self.price_per_night = price_per_night
    def update_location(self, location: str):
        self.location = location
    def update_tags(self, tags: list[str]):
        self.tags = tags
    def update_features(self, features: list[str]):
        self.features = features

    def get_id(self):
        return self.property_id
    def get_type(self):
        return self.type
    def get_price_per_night(self):
        return self.price_per_night
    def get_location(self):
        return self.location
    def get_tags(self):
        return self.tags
    def get_features(self):
        return self.features


class User:
    def __init__(self, user_id: int, name: str, group_size: int, preferred_environment: list[str],
                 budget_range: tuple[int, int], travel_date: datetime = datetime.now()):
        self.user_id = user_id
        self.name = name
        self.group_size = group_size
        if type(preferred_environment) is str:
            self.preferred_environment = [preferred_environment]
        else:
            self.preferred_environment = preferred_environment
        self.budget_range = budget_range
        self.travel_date = travel_date

    def __str__(self):
        env_list = "Preferred Environments: "
        for env in self.preferred_environment:
            env_list += f"\n\t{env}"
        return (f"--------------------------------------\n"
                f"ID: {self.user_id}\n"
                f"Name: {self.name}\n"
                f"Group Size: {self.group_size}\n"
                f"{env_list}\n"
                f"Budget Range: {self.budget_range[0]} to {self.budget_range[1]}\n"
                f"Travel Date: {self.travel_date}\n"
                f"--------------------------------------\n\n")
    def update_name(self, new_name: str):
        self.name = new_name
    def update_id(self, new_id: int):
        self.user_id = new_id
    def update_preferences(self, preferences: list[str]):
        self.preferred_environment = preferences
    def update_budget_range(self, budget_range: tuple[int, int]):
        self.budget_range = budget_range
    def update_travel_date(self, travel_date: datetime):
        self.travel_date = travel_date
    def update_group_size(self, group_size: int):
        self.group_size = group_size

    def get_id(self):
        return self.user_id
    def get_name(self):
        return self.name
    def get_preferred_environment(self):
        return self.preferred_environment
    def get_budget_range(self):
        return self.budget_range
    def get_travel_date(self):
        return self.travel_date
    def get_group_size(self):
        return self.group_size

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

def create_user():
    #TODO implement try except to catch invalid input for int
    user_id = int(input("Enter User ID: "))
    name = input("Enter User name: ")
    group_size = int(input("Enter Group Size: "))

    user_input = ''
    preferred_environment = []
    while user_input != "F":
        user_input = input("Enter Preferred Environment, F to finish: ")
        if user_input != "F":
            preferred_environment.append(user_input)

    budget_range_lower = input("Enter Lower Budget Range: ")
    budget_range_upper = input("Enter Upper Budget Range: ")
    budget_range = (int(budget_range_lower), int(budget_range_upper))
    travel_date = input("Enter Travel Date (YYYY-MM-DD HH:MM:SS.FF), N for now: ")
    if travel_date == "N":
        travel_date = datetime.now()
    else:
        travel_date = datetime.strptime(travel_date, "%Y-%m-%d %H:%M:%S.%f")

    return User(user_id, name, group_size, preferred_environment, budget_range, travel_date)

def create_property():
    property_id = int(input("Enter Property ID: "))
    location = input("Enter Location: ")
    loc_type = input("Enter Type: ")
    price_per_night = int(input("Enter Price for Night: "))

    user_input = ''
    features = []
    while user_input != "F":
        user_input = input("Enter Features, F to finish: ")
        if user_input != "F":
            features.append(user_input)

    user_input = ''
    loc_tags = []
    while user_input != "F":
        user_input = input("Enter Tags, F to finish: ")
        if user_input != "F":
            loc_tags.append(user_input)

    return Property(property_id, location, loc_type, price_per_night, features, loc_tags)

def view_property(properties: list[Property]):
    for prop in properties:
        print(prop)
    return

def view_user(users: list[User]):
    for user in users:
        print(user)
    return

def edit_property(properties: list[Property]):
    return properties

def edit_user(users: list[User]):
    return users

def get_recommendation():
    return

def llm_summary():
    return

def main():
    # Main loop for CLI
    properties = []
    users = []
    while True:
        print(f"-------------------- Main Menu --------------------\n"
              f"1. Create a new user\t 2. Create a new property\n"
              f"3. View properties  \t 4. View users\n"
              f"5. Edit user        \t 6. Edit property\n"
              f"7. Load from file   \t 8. Save to file\n"
              f"9. Get recommendations \t 10. LLM summary\n"
              f"11. Exit\n")
        user_input = input("Enter your choice: ")
        match user_input:
            case "1":
                users.append(create_user())
            case "2":
                properties.append(create_property())
            case "3":
                view_property(properties)
            case "4":
                view_user(users)
            case "5":
                users = edit_user(users)
            case "6":
                properties = edit_property(properties)
            case "7":
                properties, users = load_from_file()
            case "8":
                write_to_file(properties, users)
            case "9":
                get_recommendation()
            case "10":
                llm_summary()
            case "11":
                exit()



if  __name__ == "__main__":
    main()

#
# ###############################################################################################
# # API Call Code from JSON_Tutorial, to be paraphrased.
# # API Key is implemented without the requirement of user input. To be removed before submission
# ###############################################################################################
#
# # --- Load properties ---
# with open("properties.json", "r") as f:
#     properties = json.load(f)
#
# # --- Get API key safely (won't echo in Colab) ---
# # API_KEY = getpass.getpass("Enter your OpenRouter API key: ").strip()
# API_KEY = "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"
# # Pick a DeepSeek model available on OpenRouter.
# # Common ones include (names can change over time):
# # - "deepseek/deepseek-chat"
# # - "deepseek/deepseek-r1"
# MODEL = "deepseek/deepseek-chat-v3-0324:free"
#
# URL = "https://openrouter.ai/api/v1/chat/completions"
# HEADERS = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json",
# }
#
# SYSTEM_PROMPT = (
#     "You are an assistant for an Airbnb-like vacation property search. "
#     "Given a list of properties (JSON) and a user request, return either: "
#     "(1) a JSON object with `tags` (list of strings) inferred from the request, and "
#     "(2) optionally `property_ids` (list of integers) that might match. "
#     "Only return valid JSON."
# )
#
# def llm_search(user_prompt: str, model: str = MODEL) -> dict:
#     """Call OpenRouter and return a parsed dict. On error, return {'error': '...'}."""
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {
#                 "role": "user",
#                 "content": (
#                     "PROPERTIES:\n" + json.dumps(properties) +
#                     "\n\nUSER REQUEST:\n" + user_prompt +
#                     "\n\nRespond with JSON: {\"tags\": [...], \"property_ids\": [...]} (property_ids optional)"
#                 ),
#             },
#         ],
#         # Safety/time controls (optional):
#         "temperature": 0.2,
#     }
#     try:
#         r = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
#         # Helpful debug if something goes wrong
#         if r.status_code != 200:
#             return {"error": f"HTTP {r.status_code}", "details": r.text}
#         data = r.json()
#         # Expected shape: data["choices"][0]["message"]["content"]
#         msg = (data.get("choices") or [{}])[0].get("message", {}).get("content")
#         if not msg:
#             return {"error": "No content in response", "details": data}
#         # Try to parse JSON content the model returned
#         try:
#             return json.loads(msg)
#         except json.JSONDecodeError:
#             # If the model included extra text, try to extract JSON loosely
#             # (basic fallback—students can improve later)
#             start = msg.find("{")
#             end = msg.rfind("}")
#             if start != -1 and end != -1 and end > start:
#                 try:
#                     return json.loads(msg[start:end+1])
#                 except json.JSONDecodeError:
#                     return {"error": "Model returned non-JSON content", "raw": msg}
#             return {"error": "Model returned non-JSON content", "raw": msg}
#     except Exception as e:
#         return {"error": "Request failed", "details": str(e)}
#
# # --- Mini chatbot loop ---
# print("Vacation Property Bot (type 'exit' to quit)")
# while True:
#     prompt = input("You: ").strip()
#     if prompt.lower() == "exit":
#         print("Bot: Have a great vacation!")
#         break
#     result = llm_search(prompt)
#     if "error" in result:
#         print("Bot (error):", result["error"])
#         if "details" in result:
#             print("Details:", result["details"])
#         elif "raw" in result:
#             print("Raw output:", result["raw"])
#     else:
#         tags = result.get("tags", [])
#         prop_ids = result.get("property_ids", [])
#         print("Bot: tags =", tags)
#         print("Bot: property_ids =", prop_ids)
