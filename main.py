from datetime import datetime
import json, requests, getpass
import numpy as np
import pandas as pd
from PIL.features import features
from qstylizer.descriptor import prop

ENVIRONMENTS = ("mountain","lake","beach","city","rural","suburban","desert","forest","ski","island")
PROPERTY_TYPES = ("apartment","house","cabin","villa","condo","townhome","bnb")
FEATURES = ("hot_tub","fireplace","wifi","kitchen","parking","pool","pet_friendly","ev_charger")

class Property:
    """
    A class representing a rental property listing.

    Attributes:
        property_id (int): Unique identifier for the property.
        location (str): The property's location (stored in lowercase).
        property_type (str): The type of property (e.g., cabin, condo, house).
        price_per_night (float): Nightly price for renting the property.
        features (list[str]): List of features (e.g., WiFi, hot tub, pet-friendly).
        tags (list[str]): List of descriptive tags (e.g., family-friendly, nightlife).
        max_guests (int): Maximum number of guests allowed.
        environment (str): The environment of the property (e.g., mountain, lake, beach, city).

    Raises:
        ValueError: If the property type, environment, or features are invalid.
    """

    def __init__(self, property_id: int, location: str, property_type: str, price_per_night: float, features_list: list[str],
                 tags_list: list[str], max_guests: int, environment: str):
        """
        Initialize a new Property instance.

        Args:
            property_id (int): Unique identifier for the property.
            location (str): Location of the property.
            property_type (str): Type of the property (must be in PROPERTY_TYPES).
            price_per_night (float): Price per night.
            features_list (list[str]): Features provided by the property.
            tags_list (list[str]): Tags describing the property.
            max_guests (int): Maximum allowed guests.
            environment (str): Environment (must be in ENVIRONMENTS).

        Raises:
            ValueError: If property_type, environment, or features are invalid.
        """
        self.property_id = property_id
        self.location = location.strip().lower()
        self.property_type = property_type.strip().lower()
        self.price_per_night = price_per_night
        self.max_guests = max_guests
        self.environment = environment.strip().lower()
        if isinstance(features_list, str):
            self.features = features_list.lower().strip()
        else:
            self.features = [feature.lower().strip() for feature in features_list]
        if isinstance(tags_list, str):
            self.tags = [tags_list.lower().strip()]
        else:
            self.tags = [tag.lower().strip() for tag in tags_list]

        if self.property_type not in PROPERTY_TYPES:
            raise ValueError(f"Property type {self.property_type} is not supported.")
        if self.environment not in ENVIRONMENTS:
            raise ValueError(f"Environment {self.environment} is not supported.")
        unknown = [feature for feature in self.features if feature not in FEATURES]
        if unknown:
            raise ValueError(f"Unknown features {self.features}.")


    def __str__(self):
        """
        Return a formatted string representation of the property.

        Returns:
            str: Readable details of the property, including features and tags.
        """
        feature_list = "Features: "
        for feature in self.features:
            feature_list += f"\n\t{feature}"
        tag_list = "Tags: "
        for tag in self.tags:
            tag_list += f"\n\t{tag}"
        return (f"--------------------------------------\n"
                f"ID: {self.property_id}\n"
                f"Location: {self.location}\n"
                f"Type: {self.property_type}\n"
                f"Max guests: {self.max_guests}\n"
                f"Nightly Price: {self.price_per_night}\n"
                f"{feature_list}\n"
                f"{tag_list}\n"
                f"--------------------------------------\n\n")

    def get_dict(self):
        """
        Convert the property details to a dictionary.

        Returns:
            dict: A dictionary with property attributes.
        """
        return {
            "property_id": self.property_id,
            "location": self.location,
            "property_type": self.property_type,
            "price_per_night": self.price_per_night,
            "features": self.features,
            "tags": self.tags,
            "max_guests": self.max_guests,
            "environment": self.environment
        }

    def update_id(self, property_id: int):
        """
        Update the property ID.

        Args:
            property_id (int): New unique identifier for the property.
        """
        self.property_id = property_id

    def update_type(self, property_type: str):
        """
        Update the property type.

        Args:
            property_type (str): New type of property (must be in PROPERTY_TYPES).

        Raises:
            ValueError: If property_type is invalid.
        """
        self.property_type = property_type
        if self.property_type not in PROPERTY_TYPES:
            raise ValueError(f"Property type {self.property_type} is not supported.")

    def update_price_per_night(self, price_per_night: float):
        """
        Update the nightly rental price.

        Args:
            price_per_night (float): New price per night.
        """
        self.price_per_night = price_per_night

    def update_location(self, location: str):
        """
        Update the property location.

        Args:
            location (str): New location.
        """
        self.location = location

    def update_tags(self, tags: list[str]):
        """
        Update the tags describing the property.

        Args:
            tags (list[str]): List of new tags.
        """
        self.tags = tags

    def update_max_guests(self, max_guests: int):
        """
        Update the maximum number of guests.

        Args:
            max_guests (int): New maximum guest count.
        """
        self.max_guests = max_guests

    def update_features(self, features: list[str]):
        """
        Update the list of property features.

        Args:
            features (list[str]): New list of features.

        Raises:
            ValueError: If unknown features are provided.
        """
        self.features = features
        unknown = [feature for feature in self.features if feature not in FEATURES]
        if unknown:
            raise ValueError(f"Unknown features {self.features}.")

    def update_environment(self, environment: str):
        """
        Update the property environment.

        Args:
            environment (str): New environment (must be in ENVIRONMENTS).

        Raises:
            ValueError: If the environment is invalid.
        """
        self.environment = environment
        if  self.environment not in ENVIRONMENTS:
            raise ValueError(f"Environment {self.environment} is not supported.")

    def get_id(self):
        """
        Get the property ID.

        Returns:
            int: The unique property identifier.
        """
        return self.property_id

    def get_type(self):
        """
        Get the property type.

        Returns:
            str: The property type.
        """
        return self.property_type

    def get_price_per_night(self):
        """
        Get the nightly rental price.

        Returns:
            float: The price per night.
        """
        return self.price_per_night

    def get_location(self):
        """
        Get the property location.

        Returns:
            str: The location of the property.
        """
        return self.location

    def get_tags(self):
        """
        Get the list of property tags.

        Returns:
            list[str]: Tags associated with the property.
        """
        return self.tags

    def get_features(self):
        """
        Get the list of property features.

        Returns:
            list[str]: Features of the property.
        """
        return self.features

    def get_environment(self):
        """
        Get the property environment.

        Returns:
            str: The environment type.
        """
        return self.environment

    def get_max_guests(self):
        """
        Get the maximum number of guests allowed.

        Returns:
            int: Maximum guest capacity.
        """
        return self.max_guests


class User:
    def __init__(self, user_id: int, name: str, group_size: int, preferred_environment: list[str],
                 budget_range: tuple[int, int], travel_date: datetime = datetime.now()):
        """
        Instantiate new user

        :param user_id: User ID
        :param name: Username
        :param group_size: Group Size
        :param preferred_environment: Preferred Environment
        :param budget_range: Budget Range
        :param travel_date: Travel Date
        """
        self.user_id = user_id
        self.name = name
        self.group_size = group_size
        if type(preferred_environment) is str:
            self.preferred_environment = [preferred_environment]
        else:
            self.preferred_environment = preferred_environment
        self.budget_range = min(budget_range), max(budget_range)
        self.travel_date = travel_date

        unknown_env = [env for env in self.preferred_environment if env not in ENVIRONMENTS]
        if unknown_env:
            raise ValueError(f"Unknown preferred environment {self.preferred_environment} is not supported.")

    def __str__(self):
        env_list = "Preferred Environments:\n\t" + "\n\t".join(self.preferred_environment) \
            if self.preferred_environment else "Preferred Environments: (none)"
        return (f"--------------------------------------\n"
                f"ID: {self.user_id}\n"
                f"Name: {self.name}\n"
                f"Group Size: {self.group_size}\n"
                f"{env_list}\n"
                f"Budget Range: {self.budget_range[0]} to {self.budget_range[1]}\n"
                f"Travel Date: {self.travel_date}\n"
                f"--------------------------------------\n\n")

    def get_dict(self):
        return {
            "user_id": self.user_id,
            "name": self.name,
            "group_size": self.group_size,
            "preferred_environment": self.preferred_environment,
            "budget_range": self.budget_range,
            "travel_date": self.travel_date,
        }

    def update_name(self, new_name: str):
        self.name = new_name

    def update_id(self, new_id: int):
        self.user_id = new_id

    def update_budget_range(self, budget_range: tuple[int, int]):
        self.budget_range = budget_range

    def update_travel_date(self, travel_date: datetime):
        self.travel_date = travel_date

    def update_group_size(self, group_size: int):
        self.group_size = group_size

    def update_preferred_environment(self, preferred_environment: list[str]):
        self.preferred_environment = preferred_environment
        unknown_env = [env for env in self.preferred_environment if env not in ENVIRONMENTS]
        if unknown_env:
            raise ValueError(f"Unknown preferred environment {self.preferred_environment} is not supported.")

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

    def match_property_by_feature(self, properties: list[Property]):
       return

    def score_properties(self, properties: list[Property]):
        """
        To return a list of score for properties. The rule is following:
        20% Budget Score
        30% Capacity Score
        20% Environment Matching Score
        15% Feature Abundancy
        15% LLM Score

        :param properties:
        :return:
        """
        properties_df = properties_to_df(properties)

        # Budget Score
        # 1 = mean of user's budget range
        # 0 = 1 stddev away from max/min
        # Z score normed in 0 - 1 for rest points
        budget_mean = sum(self.budget_range) / 2
        budget_std = np.std(self.budget_range)
        price_z_score = properties_df[(properties_df["Budget Score"] -budget_mean)/ budget_std]

        return

################## datetime ISO control ##################
# def iso_now():
#     return datetime.now().replace(microsecond=0).isoformat(timespec="seconds")
#
# def time_to_iso(dt: datetime):
#     return dt.replace(microsecond=0).isoformat(timespec="seconds")
#
# def string_to_iso(time_str: str):
#     if time_str.lower() == 'n':
#         return iso_now()
#
#     time_str = time_str.replace(' ', 'T')
#     try:
#         dt = datetime.fromisoformat(time_str)



def load_from_file() -> tuple[list[Property], list[User]]:
    with open("properties.json", "r") as file:
        temp_properties = json.load(file)
    if type(temp_properties) == list:
        property_result = [Property(prop['property_id'], prop['location'], prop['property_type'], prop['price_per_night'],
                                    prop['features'], prop['property_tags'], prop['max_guests'], prop['environment']) for prop in temp_properties]
    else:
        property_result = [
            Property(temp_properties['property_id'], temp_properties['location'], temp_properties['property_type'],
                     temp_properties['price_per_night'], temp_properties['features'], temp_properties['property_tags'],
                     temp_properties['max_guests'], temp_properties['environment']) for temp_properties in temp_properties]
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


def write_to_file(properties: list[Property], users: list[User]) -> bool:
    try:
        properties_list = [property.get_dict() for property in properties]
        users_list = [user.get_dict() for user in users]
        with open("properties.json", "w") as file:
            json.dump(properties_list, file, default=str, indent=4)
        with open("users.json", "w") as file:
            json.dump(users_list, file, default=str, indent=4)
        return True
    except FileNotFoundError:
        return False

def properties_to_df(properties: list[Property]) -> pd.DataFrame:
    df = pd.DataFrame([
        prop.get_dict() for prop in properties
    ])
    env_dummy = pd.get_dummies(df['environment'], prefix='environment')
    type_dummy = pd.get_dummies(df['type'], prefix='type')
    features_dummies = pd.get_dummies(df['features'].explode(), prefix='features').groupby(level=0).max()
    tags_dummies = pd.get_dummies(df['tags'].explode(), prefix='tags').groupby(level=0).max()
    return pd.concat([df.drop(columns=['property_type', 'environment','features', 'tags']), env_dummy, type_dummy, features_dummies, tags_dummies], axis=1)


def search_user():
    # TODO
    return


def search_property():
    # TODO
    return


def filter_user():
    # TODO
    return


def filter_property():
    # TODO
    return


def map_visualization():
    # TODO
    return


def get_recommendation():
    return


def llm_summary():
    return


def gui(properties: list[Property], users: list[User]):
    # TODO
    def create_user():
        return

    def create_property():
        return

    def view_property():
        return

    def view_user():
        return

    def edit_property():
        return

    def edit_user():
        return

    return


def cli(properties: list[Property], users: list[User]):
    def create_user():
        # TODO implement try except to catch invalid input for int
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
        price_per_night = float(input("Enter Price for Night: "))

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

    def view_user(users: list[User]):
        for user in users:
            print(user)

    def edit_property(properties: list[Property]):
        id_list = [prop.get_id() for prop in properties]
        print("Property IDs: ", id_list)
        user_input = input("Enter Property to be Edited, F to dismiss: ")
        if user_input == "F":
            return properties
        target_id = id_list.index(int(user_input))
        print(f"Selected property is:\n"
              f"{properties[target_id]}\n"
              f"What you want to change?\n"
              f"1. Property ID\t 2. Location\n"
              f"3. Type       \t 4. Price per Night\n"
              f"5. Features   \t 6. Tags\n"
              f"7. Dismiss\n")
        user_input = input("Enter your choice, F to dismiss: ")
        match user_input:
            case "1":
                user_input = input("Enter New Property ID: ")
                properties[target_id].update_id(int(user_input))
            case "2":
                user_input = input("Enter New Location: ")
                properties[target_id].update_location(user_input)
            case "3":
                user_input = input("Enter New Type: ")
                properties[target_id].update_type(user_input)
            case "4":
                user_input = input("Enter New Price: ")
                properties[target_id].update_price_per_night(float(user_input))
            case "5":
                user_input = ''
                features = []
                while user_input != "F":
                    user_input = input("Enter Features, F to finish: ")
                    if user_input != "F":
                        features.append(user_input)
                properties[target_id].update_features(features)
            case "6":
                user_input = ''
                loc_tags = []
                while user_input != "F":
                    user_input = input("Enter Tags, F to finish: ")
                    if user_input != "F":
                        loc_tags.append(user_input)
                properties[target_id].update_tags(loc_tags)
            case "7":
                return properties
        return properties

    def edit_user(users: list[User]):
        id_list = [user.get_id() for user in users]
        print("User IDs: ", id_list)
        user_input = input("Enter User to be Edited, F to dismiss: ")
        if user_input == "F":
            return users
        target_id = id_list.index(int(user_input))
        print(f"Selected user is:\n"
              f"{users[target_id]}\n"
              f"What you want to change?\n"
              f"1. User ID     \t 2. Name\n"
              f"3. Group Size  \t 4. Budget\n"
              f"5. Travel Date \t 6. Preferred Environment\n"
              f"7. Dismiss\n")
        user_input = input("Enter your choice, F to dismiss: ")
        match user_input:
            case "1":
                user_input = input("Enter New User ID: ")
                users[target_id].update_id(int(user_input))
            case "2":
                user_input = input("Enter New Name: ")
                users[target_id].update_name(user_input)
            case "3":
                user_input = input("Enter New Group Size: ")
                users[target_id].update_group_size(int(user_input))
            case "4":
                budget_range_lower = input("Enter New Lower Budget Range: ")
                budget_range_upper = input("Enter New Upper Budget Range: ")
                users[target_id].update_budget_range((int(budget_range_lower), int(budget_range_upper)))
            case "5":
                travel_date = input("Enter New Travel Date (YYYY-MM-DD HH:MM:SS.FF), N for now: ")
                if travel_date == "N":
                    travel_date = datetime.now()
                else:
                    travel_date = datetime.strptime(travel_date, "%Y-%m-%d %H:%M:%S.%f")
                users[target_id].update_travel_date(travel_date)
            case "6":
                user_input = ''
                preferred_environment = []
                while user_input != "F":
                    user_input = input("Enter Preferred Environment, F to finish: ")
                    if user_input != "F":
                        preferred_environment.append(user_input)
                users[target_id].update_preferred_environment(preferred_environment)
            case "7":
                return users
        return users

        return users

    # Main CLI Loop
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


def main():
    # Main loop for CLI
    properties = []
    users = []
    while True:
        print(f"------------ Display Menu ------------ \n"
              f"1. CLI \n"
              f"2. GUI \n")
        user_input = input("Enter your choice: ")
        match user_input:
            case "1":
                cli(properties, users)
            case "2":
                gui(properties, users)
        print("Invalid input")


if __name__ == "__main__":
    main()

def get_api():
    return "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"
    # return getpass.getpass(prompt="Enter API Key: ")

class Llm:
    def __init__(self, api_key: str, model: str, url: str):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def update_api_key(self, api_key: str):
        self.api_key = api_key
        self.header = {
            "Authorization": f"Bearer {api_key}",
        }

    def update_model(self, model: str):
        self.model = model

    def update_url(self, url: str):
        self.url = url

    def llm_inquiry(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.2
        }

        try:
            r = requests.post(self.url, headers=self.header, json=payload, timeout=60)
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
