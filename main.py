import os
from datetime import datetime
import json, requests, getpass

import numpy as np
import pandas as pd

ENVIRONMENTS = ["mountain","lake","beach","city","rural","suburban","desert","forest","ski","island"]
PROPERTY_TYPES = ["apartment","house","cabin","villa","condo","townhome","bnb", "chalet", "cottage", "loft"]
FEATURES = ["hot_tub","fireplace","wifi","kitchen","parking","pool","pet_friendly","ev_charger", "gym", "bbq",
            "patio", "garden", "beach access", "canoe", "kayak", "air conditioning", "washer", "dryer"]
LOCATIONS = ["Lake Muskoka", "Toronto Downtown", "Blue Mountain","Niagara-on-the-Lake", "Prince Edward County",
             "Collingwood", "Wasaga Beach", "Kingston", "Ottawa", "Halifax"]
TAGS_POOL = ["lakefront", "beachfront", "family-friendly", "pets", "luxury", "urban", "nightlife", "business",
             "mountains", "romantic", "quiet", "nature"]


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
            self.features = [features_list.lower().strip()]
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
                f"Environment: {self.environment}\n"
                f"Type: {self.property_type}\n"
                f"Max guests: {self.max_guests}\n"
                f"Nightly Price: {self.price_per_night}\n"
                f"{feature_list}\n"
                f"{tag_list}\n")

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
        if not isinstance(property_id, int):
            raise ValueError("Property ID must be an integer.")

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
        if not isinstance(price_per_night, (int, float)):
            raise ValueError("Price per night must be a number.")

    def update_location(self, location: str):
        """
        Update the property location.

        Args:
            location (str): New location.
        """
        self.location = location.strip().lower()
        if not isinstance(location, str):
            raise ValueError("Location must be a string.")

    def update_tags(self, tags: list[str]):
        """
        Update the tags describing the property.

        Args:
            tags (list[str]): List of new tags.
        """
        if isinstance(tags, str):
            self.tags = [tags.strip().lower()]
        self.tags = [tag.strip().lower() for tag in tags]
        unknown = [tag for tag in tags if not isinstance(tag, str)]
        if unknown:
            raise ValueError("Tags must be a list of strings.")

    def update_max_guests(self, max_guests: int):
        """
        Update the maximum number of guests.

        Args:
            max_guests (int): New maximum guest count.
        """
        self.max_guests = max_guests
        if  isinstance(max_guests, int):
            raise ValueError("Maximum guest count must be an integer.")
        if max_guests < 1:
            raise ValueError("Maximum guest count must be at least 1.")

    def update_features(self, features: list[str]):
        """
        Update the list of property features.

        Args:
            features (list[str]): New list of features.

        Raises:
            ValueError: If unknown features are provided.
        """
        if not isinstance(features, list):
            raise ValueError("Features must be a list.")
        self.features = [feature.strip().lower() for feature in features]
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
        self.environment = environment.strip().lower()
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
        self.name = name.strip().lower()
        self.group_size = group_size
        if isinstance(preferred_environment, str):
            self.preferred_environment = [preferred_environment.strip().lower()]
        else:
            self.preferred_environment = preferred_environment
        unknown_env = [env for env in self.preferred_environment if env not in ENVIRONMENTS]
        if unknown_env:
            raise ValueError(f"Unknown preferred environment {self.preferred_environment} is not supported.")

        self.budget_range = min(budget_range), max(budget_range)
        self.travel_date = travel_date
        self._weighted_score = {
            "budget": .20,
            "capacity": .30,
            "environment": .20,
            "features": .15,
            "llm": .15,
        }
        self._llm_api = get_api()
        self._llm_url = "https://openrouter.ai/api/v1/chat/completions"
        self._llm_model = "openai/gpt-4o"
        json_format = "[{\"property_id\": int, \"score\": float, \"recommendation\": str},]"
        self.system_prompt = (f"You are an assistant for an Airbnb-like vacation property search. "
                              f"The user has following requirements for the desired property:\n"
                              f"1. group size: {self.group_size}\n"
                              f"2. preferred environment: {self.preferred_environment}\n"
                              f"3. budget range: {self.budget_range}\n"
                              f"4. travel date: {self.travel_date}\n"
                              f"You will be provide a list of properties , return a JSON object with "
                              f"1. the property id from the provided list, 2. the score of each property from 1 to 10, "
                              f"and 3. a simple recommendation for the property to the user within 75 words.\n"
                              f"Only return valid JSON with following structure that includes all provided property: "
                              f"{json_format} Do not omit any property")
        self._scoring_llm = Llm(self._llm_api, self._llm_model, self._llm_url, self.system_prompt)

    def __str__(self):
        env_list = "Preferred Environments:\n\t" + "\n\t".join(self.preferred_environment) \
            if self.preferred_environment else "Preferred Environments: (none)"
        return (f"\n\n"
                f"ID: {self.user_id}\n"
                f"Name: {self.name}\n"
                f"Group Size: {self.group_size}\n"
                f"{env_list}\n"
                f"Budget Range: {self.budget_range[0]} to {self.budget_range[1]}\n"
                f"Travel Date: {self.travel_date}\n"
                f"--------------------------------------")

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
        self.name = new_name.strip().lower()
        if not isinstance(self.name, str):
            raise ValueError("Name must be a string.")

    def update_id(self, new_id: int):
        self.user_id = new_id
        if not isinstance(self.user_id, int):
            raise ValueError("User ID must be an integer.")

    def update_budget_range(self, budget_range: tuple[int, int]):
        self.budget_range = budget_range
        if not isinstance(self.budget_range[0], (float, int)) or not isinstance(self.budget_range[1], (float, int)):
            raise ValueError("Budget range must be a tuple of two integers or floats.")

    def update_travel_date(self, travel_date: datetime):
        self.travel_date = travel_date

    def update_group_size(self, group_size: int):
        self.group_size = group_size
        if not isinstance(self.group_size, int):
            raise ValueError("Group size must be an integer.")
        if self.group_size < 1:
            raise ValueError("Group size must be at least 1.")

    def update_preferred_environment(self, preferred_environment: list[str]):
        if isinstance(preferred_environment, str):
            self.preferred_environment = [preferred_environment.strip().lower()]
        elif isinstance(preferred_environment, list):
            self.preferred_environment = [env.strip().lower() for env in preferred_environment]
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

    def _llm_scoring(self, properties: list[Property]):

        result = self._scoring_llm.llm_inquiry([prop.get_dict() for prop in properties])
        start = result.find("[") if "[" in result else result.find("{")
        end = result.rfind("]")
        cleaned = result[start : end + 1].replace("\n", "")

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError("LLM output is not valid JSON")

        return pd.DataFrame(parsed)

    def score_properties(self, properties: list[Property]):
        """
        To return a list of score for properties.

        :param properties:
        :return:
        """
        properties_df = properties_to_df(properties)

        # Budget Score
        # 10 = mean of user's budget range
        # Z score normed in 0 - 10 for rest points
        budget_mean = sum(self.budget_range) / 2
        budget_std = max(np.std(self.budget_range), 0.01)
        properties_df["price score"] = np.clip(10 * (1 - abs((properties_df["price_per_night"] - budget_mean)/ budget_std)), 0, 10)

        # Capacity Score
        # 10 = exact party size of the user
        # for each one unsatisfied spot minus 3 points
        properties_df["capacity score"] = np.clip(10 - 3 * (self.group_size - properties_df["max_guests"]), 0, 10)

        # Environment Score
        # 10 = environment matched
        environment_token = ["environment_" + penv for penv in self.preferred_environment]
        properties_df["environment score"] = 0
        for penv in environment_token:
            properties_df["environment score"] = np.clip((10 * properties_df[penv].astype(int) + properties_df["environment score"]), 0, 10)

        # Feature Abundancy

        properties_df["feature score"] = sum(properties_df["features_" + token] for token in FEATURES)

        # LLM Score
        llm_score_df = self._llm_scoring(properties).add_prefix("llm_").rename(columns={"llm_property_id": "property_id"})
        properties_df = pd.merge(properties_df, llm_score_df, how="left", on="property_id")

        properties_df["total score"] = (
                self._weighted_score["budget"] * properties_df["price score"] +
                self._weighted_score["capacity"] * properties_df["capacity score"] +
                self._weighted_score["environment"] * properties_df["environment score"] +
                self._weighted_score["features"] * properties_df["feature score"] +
                self._weighted_score["llm"] * properties_df["llm_score"])
        return properties_df[["property_id", "total score", "llm_recommendation"]].sort_values("total score", ascending=False)


def load_from_file() -> tuple[list[Property], list[User]]:
    try:
        with open("properties.json", "r") as file:
            temp_properties = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")

    try:
        if type(temp_properties) == list:
            property_result = [Property(prop['property_id'], prop['location'], prop['property_type'], prop['price_per_night'],
                                        prop['features'], prop['tags'], prop['max_guests'], prop['environment']) for prop in temp_properties]
        else:
            property_result = [
                Property(temp_properties['property_id'], temp_properties['location'], temp_properties['property_type'],
                         temp_properties['price_per_night'], temp_properties['features'], temp_properties['tags'],
                         temp_properties['max_guests'], temp_properties['environment']) for temp_properties in temp_properties]
    except Exception as e:
        raise ValueError(f"Error loading properties from file: {e}")

    try:
        with open("users.json", "r") as file:
            temp_users = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")

    try:
        if type(temp_users) == list:
            user_result = [User(user['user_id'], user['name'], user['group_size'], user['preferred_environment'],
                                user['budget_range'], datetime.strptime(user['travel_date'], "%Y-%m-%d %H:%M:%S.%f"))
                           for user in temp_users]
        else:
            user_result = [User(temp_users['user_id'], temp_users['name'], temp_users['group_size'],
                                temp_users['preferred_environment'], temp_users['budget_range'],
                                datetime.strptime(temp_users['travel_date'], "%Y-%m-%d %H:%M:%S.%f"))]
        return property_result, user_result
    except Exception as e:
        raise ValueError(f"Error loading users from file: {e}")


def write_to_file(properties: list[Property], users: list[User]) -> bool:
    try:
        properties_list = [prop.get_dict() for prop in properties]
        users_list = [user.get_dict() for user in users]
        with open("properties.json", "w") as file:
            json.dump(properties_list, file, default=str, indent=4)
        with open("users.json", "w") as file:
            json.dump(users_list, file, default=str, indent=4)
        return True
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")


def properties_to_df(properties: list[Property]) -> pd.DataFrame:
    df = pd.DataFrame([
        prop.get_dict() for prop in properties
    ])
    env_dummy = pd.get_dummies(df['environment'], prefix='environment')
    type_dummy = pd.get_dummies(df['property_type'], prefix='type')
    features_dummies = pd.get_dummies(df['features'].explode(), prefix='features').groupby(level=0).max()
    tags_dummies = pd.get_dummies(df['tags'].explode(), prefix='tags').groupby(level=0).max()
    return pd.concat([df.drop(columns=['property_type', 'environment','features', 'tags']), env_dummy, type_dummy,
                      features_dummies, tags_dummies], axis=1)


def search_user(users: list[User], uid: int = None, name: str = None, group_size: int = None, preferred_environment: list[str] = None,
                budget: int = None, travel_date: datetime = None):
    if uid is None and name is None and group_size is None and preferred_environment is None and budget is None and travel_date is None:
        return users
    result = []
    for user in users:
        if ((uid is None or user.get_id() == uid) and (name is None or user.get_name() == name)
                and (group_size is None or user.get_group_size() == group_size)
                and (preferred_environment is None or
                     len([env for env in preferred_environment if env in user.get_preferred_environment()])>0)
                and (budget is None or user.get_budget_range()[0] <= budget <= user.get_budget_range()[1])
                and (travel_date is None or user.get_travel_date() == travel_date)):
            result.append(user)
    return result


def search_property(properties: list[Property], property_id: int = None, location: str =None, property_type: str=None, price_per_night: float=None,
                    features: list[str]=None, tags: list[str]=None, max_guests: int=None, environment: str=None):
    if (property_id is None and location is None and property_type is None and price_per_night is None and features is None and tags is None
            and max_guests is None and environment is None):
        return properties
    if isinstance(features, str):
        features = [features]
    if isinstance(tags, str):
        tags = [tags]
    result = []
    for prop in properties:
        if ((property_id is None or prop.get_id() == property_id) and (location is None or prop.get_location() == location)
                and (property_type is None or prop.get_type() == property_type)
                and (price_per_night is None or prop.get_price_per_night() <= price_per_night)
                and (features is None or len([feature for feature in features if feature in prop.get_features()]) == len(features))
                and (tags is None or len([tag for tag in tags if tag in prop.get_tags()]) == len(tags))
                and (max_guests is None or prop.get_max_guests() == max_guests)
                and (environment is None or prop.get_environment() == environment)):
            result.append(prop)
    return result


def get_recommendation(user: User, properties: list[Property]):
    result = user.score_properties(properties)
    if result.shape[0] >= 5:
        return result.nlargest(5, "total score")
    else:
        return result.nlargest(result.shape[0], "total score")


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
        max_guest = int(input("Enter Max Guests: "))
        environment = None
        while environment is None:
            environment = input(f"Enter Environment from {ENVIRONMENTS}: ")
            if environment not in ENVIRONMENTS:
                print("Invalid Environment")
                environment = None
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

        return Property(property_id, location, loc_type, price_per_night, features, loc_tags, max_guest, environment)

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
    def get_top(users: list[User], properties: list[Property]):
        id_list = [user.get_id() for user in users]
        print("User IDs: ", id_list)
        user_input = input("Enter User to Generate Recommendation, F to dismiss: ")
        if user_input == "F":
            return properties
        target_id = id_list.index(int(user_input))
        top_properties = get_recommendation(users[target_id], properties).to_dict()
        for i, property_id in enumerate(top_properties["property_id"], start=1):
            print(f"Number {i}\n"
                  f"{properties[top_properties['property_id'][property_id]]}\n"
                  f"Property Score:\t   {round(top_properties['total score'][property_id], 2)}\n"
                  f"Recommendation:\t   {top_properties['llm_recommendation'][property_id]}\n")


        return users
    # Main CLI Loop
    while True:
        print(f"-------------------- Main Menu --------------------\n"
              f"1. Create a new user\t 2. Create a new property\n"
              f"3. View properties  \t 4. View users\n"
              f"5. Edit user        \t 6. Edit property\n"
              f"7. Load from file   \t 8. Save to file\n"
              f"9. Get recommendations \t 10. Exit\n"
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
                get_top(users, properties)
            case "10":
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

def get_api():
    api_key = os.getenv("API_KEY")
    if api_key is None:
        api_key = getpass.getpass(prompt="Enter API Key: ")
    return api_key

class Llm:
    def __init__(self, api_key: str, model: str, url: str, system_prompt: str):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.system_prompt = system_prompt
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def update_api_key(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def update_model(self, model: str):
        self.model = model

    def update_url(self, url: str):
        self.url = url

    def llm_inquiry(self, user_prompt):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": str(user_prompt)
                },
            ],
            "temperature": 0.2
        }

        try:
            r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            # Helpful debug if something goes wrong
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}", "details": r.text}
            data = r.json()
            # Expected shape: data["choices"][0]["message"]["content"]
            msg = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not msg:
                raise ValueError("error: No content in response")
            return msg
        except Exception as e:
            return {"error": "Request failed", "details": str(e)}


if __name__ == "__main__":
    main()