import os
from datetime import datetime
import json, requests, getpass

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

environments_pool = ["mountain", "lake", "beach", "city", "rural", "suburban", "desert", "forest", "ski", "island"]
types_pool = ["apartment", "house", "cabin", "villa", "condo", "townhome", "bnb", "chalet", "cottage", "loft"]
features_pool = ["hot_tub", "fireplace", "wifi", "kitchen", "parking", "pool", "pet_friendly", "ev_charger", "gym", "bbq",
            "patio", "garden", "beach access", "canoe", "kayak", "air conditioning", "washer", "dryer"]
locations_pool = ["Lake Muskoka", "Toronto Downtown", "Blue Mountain", "Niagara-on-the-Lake", "Prince Edward County",
             "Collingwood", "Wasaga Beach", "Kingston", "Ottawa", "Halifax"]
tags_pool = ["lakefront", "beachfront", "family-friendly", "pets", "luxury", "urban", "nightlife", "business",
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

    def __init__(self, property_id: int, location: str, property_type: str, price_per_night: float, features: list[str],
                 tags_list: list[str], max_guests: int, environment: str):
        """
        Initialize a new Property instance.

        Args:
            property_id (int): Unique identifier for the property.
            location (str): Location of the property.
            property_type (str): Type of the property (must be in PROPERTY_TYPES).
            price_per_night (float): Price per night.
            features (list[str]): Features provided by the property.
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
        if isinstance(features, str):
            self.features = [features.lower().strip()]
        else:
            self.features = [feature.lower().strip() for feature in features]
        if isinstance(tags_list, str):
            self.tags = [tags_list.lower().strip()]
        else:
            self.tags = [tag.lower().strip() for tag in tags_list]

        update_features_pool(self.features)
        update_tags_pool(self.tags)
        update_environments_pool(self.environment)
        update_features_pool(self.features)
        update_tags_pool(self.tags)




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
        if self.property_type not in types_pool:
            types_pool.append(self.property_type)

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
        wrong_type = [tag for tag in tags if not isinstance(tag, str)]
        tags_pool.extend(wrong_type)
        if wrong_type:
            raise ValueError("Tags must be a list of strings.")
        tags_pool.extend([tag for tag in tags if tag not in tags_pool])


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
        features_pool.extend([feature for feature in self.features if feature not in features_pool])

    def update_environment(self, environment: str):
        """
        Update the property environment.

        Args:
            environment (str): New environment (must be in ENVIRONMENTS).

        Raises:
            ValueError: If the environment is invalid.
        """
        self.environment = environment.strip().lower()
        if  self.environment not in environments_pool:
            environments_pool.append(self.environment)

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
    """
    A class representing a user profile for an Airbnb-like property recommendation system.

    Attributes:
        user_id (int): Unique identifier for the user.
        name (str): Name of the user (stored in lowercase).
        group_size (int): Number of people in the group.
        preferred_environment (list[str]): Preferred environments (e.g., "beach", "city").
        budget_range (tuple[int, int]): Minimum and maximum nightly budget.
        travel_date (datetime): Desired travel date.
        _weighted_score (dict): Weights for scoring categories (budget, capacity, environment, features, LLM).
        _llm_api (str): API key for the LLM.
        _llm_url (str): Endpoint for the LLM.
        _llm_model (str): LLM model identifier.
        system_prompt (str): Prompt template used for LLM scoring.
        _scoring_llm (Llm): Instance of the Llm helper class.

    Raises:
        ValueError: If the preferred environment contains unsupported values.
    """
    def __init__(self, user_id: int, name: str, group_size: int, preferred_environment: list[str],
                 budget_range: tuple[int, int], travel_date: datetime = datetime.now()):
        """
        Instantiate a new user.

        Args:
            user_id (int): Unique identifier for the user.
            name (str): Name of the user.
            group_size (int): Number of people in the group.
            preferred_environment (list[str] | str): List of preferred environments (or a single string).
            budget_range (tuple[int, int]): Min and max nightly budget.
            travel_date (datetime, optional): Desired travel date. Defaults to now.
        """
        self.user_id = user_id
        self.name = name.strip().lower()
        self.group_size = group_size
        if isinstance(preferred_environment, str):
            self.preferred_environment = [preferred_environment.strip().lower()]
        else:
            self.preferred_environment = preferred_environment
        environments_pool.extend([env for env in self.preferred_environment if env not in environments_pool])
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
        """
        Return a readable string representation of the user profile.

        Returns:
            str: Formatted user details including group size, budget, and environments.
        """
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
        """
        Convert the user profile to a dictionary.

        Returns:
            dict: A dictionary with user attributes.
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "group_size": self.group_size,
            "preferred_environment": self.preferred_environment,
            "budget_range": self.budget_range,
            "travel_date": self.travel_date,
        }

    def update_name(self, new_name: str):
        """
        Update the user's name.

        Args:
            new_name (str): New name.

        Raises:
            ValueError: If the input is not a string.
        """
        self.name = new_name.strip().lower()
        if not isinstance(self.name, str):
            raise ValueError("Name must be a string.")

    def update_id(self, new_id: int):
        """
        Update the user's ID.

        Args:
            new_id (int): New user ID.

        Raises:
            ValueError: If the input is not an integer.
        """
        self.user_id = new_id
        if not isinstance(self.user_id, int):
            raise ValueError("User ID must be an integer.")

    def update_budget_range(self, budget_range: tuple[int, int]):
        """
        Update the user's budget range.

        Args:
            budget_range (tuple[int, int]): New budget range.

        Raises:
            ValueError: If values are not numeric.
        """
        self.budget_range = budget_range
        if not isinstance(self.budget_range[0], (float, int)) or not isinstance(self.budget_range[1], (float, int)):
            raise ValueError("Budget range must be a tuple of two integers or floats.")

    def update_travel_date(self, travel_date: datetime):
        """
        Update the user's travel date.

        Args:
            travel_date (datetime): New travel date.
        """
        self.travel_date = travel_date

    def update_group_size(self, group_size: int):
        """
        Update the group size.

        Args:
            group_size (int): New group size.

        Raises:
            ValueError: If not an integer or less than 1.
        """
        self.group_size = group_size
        if not isinstance(self.group_size, int):
            raise ValueError("Group size must be an integer.")
        if self.group_size < 1:
            raise ValueError("Group size must be at least 1.")

    def update_preferred_environment(self, preferred_environment: list[str]):
        """
        Update the user's preferred environments.

        Args:
            preferred_environment (list[str] | str): List of preferred environments.

        Raises:
            ValueError: If the environment is not supported.
        """
        if isinstance(preferred_environment, str):
            self.preferred_environment = [preferred_environment.strip().lower()]
        elif isinstance(preferred_environment, list):
            self.preferred_environment = [env.strip().lower() for env in preferred_environment]
        unknown_env = [env for env in self.preferred_environment if env not in environments_pool]
        if unknown_env:
            raise ValueError(f"Unknown preferred environment {self.preferred_environment} is not supported.")

    def get_id(self):
        """
        Return the user ID (int).
        """
        return self.user_id

    def get_name(self):
        """
        Return the user's name (str).
        """
        return self.name

    def get_preferred_environment(self):
        """
        Return the preferred environments (list[str]).
        """
        return self.preferred_environment

    def get_budget_range(self):
        """
        Return the budget range (tuple[int, int]).
        """
        return self.budget_range

    def get_travel_date(self):
        """
        Return the travel date (datetime).
        """
        return self.travel_date

    def get_group_size(self):
        """
        Return the group size (int).
        """
        return self.group_size

    def _llm_scoring(self, properties: list[Property]):
        """
        Score properties using the LLM model.

        Args:
            properties (list[Property]): List of properties to score.

        Returns:
            pd.DataFrame: DataFrame with property_id, LLM score, and recommendation.

        Raises:
            ValueError: If LLM output is not valid JSON.
        """
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
        Score properties for the user based on budget, capacity, environment, features, and LLM feedback.

        Args:
            properties (list[Property]): List of properties to score.

        Returns:
            pd.DataFrame: Ranked DataFrame with property_id, total score, and LLM recommendation.
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

        properties_df["feature score"] = sum(properties_df["features_" + token] for token in features_pool)

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
    """
    Load property and user data from JSON files.

    Returns:
        tuple[list[Property], list[User]]: Lists of Property and User objects.

    Raises:
        FileNotFoundError: If JSON files are missing.
        ValueError: If JSON is invalid or parsing fails.
    """
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
    """
    Save property and user data to JSON files.

    Args:
        properties (list[Property]): List of properties.
        users (list[User]): List of users.

    Returns:
        bool: True if successful.

    Raises:
        FileNotFoundError: If file writing fails.
    """
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
    """
    Convert a list of Property objects into a pandas DataFrame with dummy variables.

    Args:
        properties (list[Property]): List of properties.

    Returns:
        pd.DataFrame: DataFrame with expanded categorical variables for features, tags, type, and environment.
    """
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
    """
    Search users by attributes.

    Args:
        users (list[User]): List of users.
        uid (int, optional): User ID.
        name (str, optional): User name.
        group_size (int, optional): Group size.
        preferred_environment (list[str], optional): Preferred environment(s).
        budget (int, optional): Budget to check if within user's range.
        travel_date (datetime, optional): Travel date.

    Returns:
        list[User]: Filtered list of matching users.
    """
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
    """
    Search properties by attributes.

    Args:
        properties (list[Property]): List of properties.
        property_id (int, optional): Property ID.
        location (str, optional): Location filter.
        property_type (str, optional): Type filter.
        price_per_night (float, optional): Max price per night.
        features (list[str] | str, optional): Required features.
        tags (list[str] | str, optional): Required tags.
        max_guests (int, optional): Guest capacity.
        environment (str, optional): Environment filter.

    Returns:
        list[Property]: Filtered list of matching properties.
    """
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
    """
    Get top property recommendations for a user.

    Args:
        user (User): The user profile.
        properties (list[Property]): List of available properties.

    Returns:
        pd.DataFrame: Top 5 properties ranked by total score.
    """
    result = user.score_properties(properties)
    if result.shape[0] >= 5:
        return result.nlargest(5, "total score")
    else:
        return result.nlargest(result.shape[0], "total score")

def update_environments_pool(environment: list[str] | str):
    """
    Update global environments pool with new environments.
    """
    if isinstance(environment, str):
        environment = [environment.strip().lower()]
    environments_pool.extend([env for env in environment if env not in environments_pool])

def update_tags_pool(tag: list[str] | str):
    """
    Update global tags pool with new tags.
    """
    if isinstance(tag, str):
        tag = [tag.strip().lower()]
    tags_pool.extend([tag for tag in tag if tag not in tags_pool])

def update_features_pool(feature: list[str] | str):
    """
    Update global features pool with new features.
    """
    if isinstance(feature, str):
        feature = [feature.strip().lower()]
    features_pool.extend([feature for feature in feature if feature not in features_pool])

def update_locations_pool(locations: list[str] | str):
    """
    Update global locations pool with new locations.
    """
    if isinstance(locations, str):
        locations = [locations.strip().lower()]
    locations_pool.extend([location for location in locations if location not in locations_pool])

def update_types_pool(types: list[str] | str):
    """
    Update global property types pool with new types.
    """
    if isinstance(types, str):
        types = [types.strip().lower()]
    types_pool.extend([type for type in types if type not in types_pool])

class GUI:
    """
    A class representing the graphical user interface (GUI) for the Simple Airbnb application.

    Attributes:
        root (tk.Tk): The main Tkinter application window.
        users_list (list[User]): List of user profiles.
        properties_list (list[Property]): List of property listings.
        label (tk.Label): A label widget for displaying text.
        listbox (tk.Listbox): A listbox widget for displaying menu options.
        entry (tk.Entry): An entry widget for user input.
        button (tk.Button): A button widget for submitting input.
    """
    def __init__(self):
        """
        Initialize the GUI application.

        Creates the main Tkinter window, initializes user and property lists,
        and displays the main menu.
        """
        self.root = tk.Tk()
        self.root.title("Simple Airbnb") # change title
        self.root.geometry("500x500")

        self.users_list = []
        self.properties_list = []

        self.main_menu()

    def main_menu(self):
        """
        Display the main menu of the application.

        Shows available options such as creating users, creating properties,
        viewing data, editing entries, loading/saving files, and exiting.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        self.label = tk.Label(self.root, text="Welcome to Simple Airbnb! Please select an option: \n\n Main Menu")
        self.label.pack(pady=10)

        self.listbox = tk.Listbox(self.root, width=40, height=15)
        self.listbox.pack()

        list_of_options = ["1. Create a new user", "2. Create a new property", "3. View properties",
                          "4. View users", "5. Edit user", "6. Edit property", "7. Load from file",
                          "8. Save to file", "9. Get recommendations", "10. Exit"]
        for item in list_of_options:
            self.listbox.insert(tk.END, item)

        self.entry = tk.Entry(self.root, width=30)
        self.entry.pack(pady=10)

        self.button = tk.Button(self.root, text="Enter", command=self.redirect_input)
        self.button.pack(pady=20)

    def redirect_input(self):
        """
        Redirect user input from the main menu to the corresponding functionality.

        Raises:
            ValueError: If the input is not a valid integer menu option.
        """
        try:
            user_input = int(self.entry.get())
        except ValueError:
            messagebox.showinfo("Error", "Please enter a valid number.")
            self.entry.delete(0, tk.END)
            return

        if user_input == 1:
            self.create_user()
        elif user_input == 2:
            self.create_property()
        elif user_input == 3:
            self.view_property(self.properties_list)
        elif user_input == 4:
            self.view_user(self.users_list)
        elif user_input == 5:
            self.edit_user(self.users_list)
        elif user_input == 6:
            self.edit_property(self.properties_list)
        elif user_input == 7:
            self.properties_list, self.users_list = load_from_file()
            messagebox.showinfo("Simple Airbnb", "Success!")
            self.main_menu()
        elif user_input == 8:
            write_to_file(self.properties_list, self.users_list)
            messagebox.showinfo("Simple Airbnb", "Success!")
            self.main_menu()
        elif user_input == 9:
            self.get_recommendations(self.users_list, self.properties_list)
        elif user_input == 10:
            self.exit()

    def run(self):
        """
        Run the Tkinter main loop to keep the GUI application active.
        """
        self.root.mainloop()

    # collects user entries and adds label
    def user_entry(self, label):
        """
        Create a labeled entry field for user input.

        Args:
            label (str): The label text to display.

        Returns:
            tk.Entry: The entry widget for user input.
        """
        tk.Label(self.root, text=label).pack()
        user_entry = tk.Entry(self.root)
        user_entry.pack()
        return user_entry

    # user entry labels
    # it's more difficult to do the one by one entries like the cli so its all in one form rn for the gui
    def create_user(self):
        """
        Display the form for creating a new user.

        Includes fields for user ID, name, group size, preferred environment,
        budget range, and travel date.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Create User").pack(pady=5)

        self.user_id_entry = self.user_entry("User ID:")
        self.name_entry = self.user_entry("Name:")
        self.group_size_entry = self.user_entry("Group Size:")
        self.env_entry = self.user_entry("Preferred Environment (comma separated):")
        self.budget_low_entry = self.user_entry("Lower Budget Range:")
        self.budget_high_entry = self.user_entry("Upper Budget Range:")
        self.date_entry = self.user_entry("Travel Date (YYYY-MM-DD HH:MM:SS.FF), N for today:")

        tk.Button(self.root, text="Save", command=self.submit_user).pack(pady=10)

    # create a new user - for button to work (same format as CLI)
    def submit_user(self):
        """
        Collect inputs from the user creation form and save the new user.

        Raises:
            ValueError: If inputs are invalid (e.g., non-integer IDs or budgets).
        """
        user_id = int(self.user_id_entry.get())
        name = str(self.name_entry.get())
        group_size = int(self.group_size_entry.get())
        preferred_environment = [i.strip() for i in self.env_entry.get().split(",")]
        budget_range = (int(self.budget_low_entry.get()), int(self.budget_high_entry.get()))
        travel_date_input = str(self.date_entry.get())
        if travel_date_input == "N":
            travel_date = datetime.now()
        else:
            travel_date = datetime.strptime(travel_date_input, "%Y-%m-%d %H:%M:%S.%f")

        users = User(user_id, name, group_size, preferred_environment, budget_range, travel_date)
        self.users_list.append(users)

        messagebox.showinfo("Simple Airbnb", "User has been saved!")

        self.main_menu()

    def create_property(self):
        """
        Display the form for creating a new property.

        Includes fields for property ID, location, type, price, maximum guests,
        features, tags, and environment.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Create User").pack(pady=5)

        self.property_id_entry = self.user_entry("Property ID:")
        self.location_entry = self.user_entry("Location:")
        self.type_entry = self.user_entry("Type:")
        self.price_entry = self.user_entry("Price for Night:")
        self.max_guests_entry = self.user_entry("Maximum Number of Guests:")
        self.features_entry = self.user_entry("Features:")
        self.tags_entry = self.user_entry("Tags:")
        self.environment_entry = self.user_entry("Environment:")

        tk.Button(self.root, text="Enter", command=self.submit_property).pack(pady=10)

    def submit_property(self):
        """
        Collect inputs from the property creation form and save the new property.

        Raises:
            ValueError: If inputs are invalid (e.g., non-integer IDs or prices).
        """
        property_id = self.property_id_entry.get()
        location = self.location_entry.get()
        loc_type = self.type_entry.get()
        price_per_night = self.price_entry.get()
        max_guests = self.max_guests_entry.get()
        features = [i.strip() for i in self.features_entry.get().split(",")]
        loc_tags = [i.strip() for i in self.tags_entry.get().split(",")]
        env = self.environment_entry.get()

        prop = Property(property_id, location, loc_type, price_per_night, features, loc_tags, max_guests, env)
        self.properties_list.append(prop)

        messagebox.showinfo("Simple Airbnb", "Property has been saved!")

        self.main_menu()

    def view_property(self, properties):
        """
        Display property viewing options.

        Args:
            properties (list[Property]): List of property objects to display.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        properties_df = pd.DataFrame([prop.get_dict() for prop in properties])

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        one_property = tk.Button(button_frame, text="I know my Property ID", command=lambda: self.view_one_property(properties_df))
        one_property.pack(side="left", padx=10)

        all_property = tk.Button(button_frame, text="View All", command=lambda: self.view_all_property(properties_df))
        all_property.pack(side="right", padx=10)

    def view_one_property(self, properties):
        """
        Display input form to view a single property by ID.

        Args:
            properties (pd.DataFrame): DataFrame of property details.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        self.selected_property_id = self.user_entry("Please enter Property ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.get_property_id(properties)).pack(pady=10)

    def get_property_id(self, properties):
        """
        Retrieve and display details of a property by ID.

        Args:
            properties (pd.DataFrame): DataFrame of property details.
        """
        self.ID = int(self.selected_property_id.get())
        selected_property = properties[properties["property_id"]==self.ID]

        for widget in self.root.winfo_children():
            widget.destroy()

        self.tree = ttk.Treeview(self.root)
        self.tree.pack(fill="both", expand=True)

        self.tree["columns"] = list(selected_property.columns)
        self.tree["show"] = "headings"

        for col in selected_property.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in selected_property.iterrows():
            self.tree.insert("", "end", values=list(row))

        tk.Button(self.root, text="Main Menu", command=self.main_menu).pack(pady=10)

    # note: can't wrap text in treeview
    def view_all_property(self, properties):
        """
        Display all properties in a scrollable table.

        Args:
            properties (pd.DataFrame): DataFrame of property details.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(frame)
        self.tree.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=v_scroll.set)

        h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=h_scroll.set)

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.tree["columns"] = list(properties.columns)
        self.tree["show"] = "headings"

        for col in properties.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in properties.iterrows():
            self.tree.insert("", "end", values=list(row))

        tk.Button(self.root, text="Main Menu", command=self.main_menu).pack(pady=10)

    def view_user(self, users):
        """
        Display user viewing options.

        Args:
            users (list[User]): List of user objects to display.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        users_df = pd.DataFrame([user.get_dict() for user in users])

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        one_property = tk.Button(button_frame, text="I know my User ID", command=lambda: self.view_one_user(users_df))
        one_property.pack(side="left", padx=10)

        all_property = tk.Button(button_frame, text="View All", command=lambda: self.view_all_user(users_df))
        all_property.pack(side="right", padx=10)

    def view_one_user(self, users):
        """
        Display input form to view a single user by ID.

        Args:
            users (pd.DataFrame): DataFrame of user details.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        self.selected_user_id = self.user_entry("Please enter User ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.get_user_id(users)).pack(pady=10)

    def get_user_id(self, users):
        """
        Retrieve and display details of a user by ID.

        Args:
            users (pd.DataFrame): DataFrame of user details.
        """
        self.ID = int(self.selected_user_id.get())
        selected_user = users[users["user_id"] == self.ID]

        for widget in self.root.winfo_children():
            widget.destroy()

        self.tree = ttk.Treeview(self.root)
        self.tree.pack(fill="both", expand=True)

        self.tree["columns"] = list(selected_user.columns)
        self.tree["show"] = "headings"

        for col in selected_user.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in selected_user.iterrows():
            self.tree.insert("", "end", values=list(row))

        tk.Button(self.root, text="Main Menu", command=self.main_menu).pack(pady=10)

    def view_all_user(self, users):
        """
        Display all users in a scrollable table.

        Args:
            users (pd.DataFrame): DataFrame of user details.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(frame)
        self.tree.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=v_scroll.set)

        h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=h_scroll.set)

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.tree["columns"] = list(users.columns)
        self.tree["show"] = "headings"

        for col in users.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in users.iterrows():
            self.tree.insert("", "end", values=list(row))

        tk.Button(self.root, text="Main Menu", command=self.main_menu).pack(pady=10)

    def edit_property(self, properties):
        """
        Display input form to edit a property by ID.

        Args:
            properties (list[Property]): List of property objects.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        properties_df = pd.DataFrame([prop.get_dict() for prop in properties])

        tk.Label(self.root, text="Edit Property").pack(pady=5)

        self.editing_property_id = self.user_entry("Please enter Property ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.edit_property_values(properties_df)).pack(pady=10)

    def edit_property_values(self, properties):
        """
        Display editable fields for a selected property.

        Args:
            properties (pd.DataFrame): DataFrame of property details.
        """
        edit_property = properties[properties["property_id"] == int(self.editing_property_id.get())]
        property_row = edit_property.iloc[0]

        for widget in self.root.winfo_children():
            widget.destroy()

        self.property_id_entry = self.user_entry("Property ID:")
        self.location_entry = self.user_entry("Location:")
        self.type_entry = self.user_entry("Type:")
        self.price_entry = self.user_entry("Price for Night:")
        self.max_guests_entry = self.user_entry("Max Guests:")
        self.features_entry = self.user_entry("Features:")
        self.tags_entry = self.user_entry("Tags:")
        self.environment_entry = self.user_entry("Environment:")

        self.property_id_entry.insert(0, str(property_row["property_id"]))
        self.location_entry.insert(0, str(property_row["location"]))
        self.type_entry.insert(0, str(property_row["property_type"]))
        self.price_entry.insert(0, str(property_row["price_per_night"]))
        self.max_guests_entry.insert(0, str(property_row["max_guests"]))

        features = property_row["features"]
        if isinstance(features, list):
            self.features_entry.insert(0, ",".join(features))
        else:
            self.features_entry.insert(0, str(features))

        tags = property_row["tags"]
        if isinstance(tags, list):
            self.tags_entry.insert(0, ",".join(tags))
        else:
            self.tags_entry.insert(0, str(tags))

        self.environment_entry.insert(0, str(property_row["environment"]))

        old_id = property_row["property_id"]
        tk.Button(self.root, text="Save", command=lambda: self.replace_property(old_id)).pack(pady=10)

    def replace_property(self, old_id):
        """
        Replace an existing property with updated values.

        Args:
            old_id (int): ID of the property being replaced.
        """
        self.properties_list = [i for i in self.properties_list if str(i.property_id) != str(old_id)]
        self.submit_property()

    def replace_user(self, old_id):
        """
        Replace an existing user with updated values.

        Args:
            old_id (int): ID of the user being replaced.
        """
        self.users_list = [i for i in self.users_list if str(i.user_id) != str(old_id)]
        self.submit_user()

    def edit_user(self, users):
        """
        Display input form to edit a user by ID.

        Args:
            users (list[User]): List of user objects.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        users_df = pd.DataFrame([user.get_dict() for user in users])

        tk.Label(self.root, text="Edit User").pack(pady=5)

        self.editing_user_id = self.user_entry("Please enter User ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.edit_user_values(users_df)).pack(pady=10)

    def edit_user_values(self, users):
        """
        Display editable fields for a selected user.

        Args:
            users (pd.DataFrame): DataFrame of user details.
        """
        edit_user = users[users["user_id"]==int(self.editing_user_id.get())]
        user_row = edit_user.iloc[0]

        for widget in self.root.winfo_children():
            widget.destroy()

        self.user_id_entry = self.user_entry("User ID:")
        self.name_entry = self.user_entry("Name:")
        self.group_size_entry = self.user_entry("Group Size:")
        self.env_entry = self.user_entry("Preferred Environment (comma separated):")
        self.budget_low_entry = self.user_entry("Lower Budget Range:")
        self.budget_high_entry = self.user_entry("Upper Budget Range:")
        self.date_entry = self.user_entry("Travel Date (YYYY-MM-DD HH:MM:SS.FF), N for today:")

        self.user_id_entry.insert(0, str(user_row["user_id"]))
        self.name_entry.insert(0, str(user_row["name"]))
        self.group_size_entry.insert(0, str(user_row["group_size"]))

        env = user_row["preferred_environment"]
        if isinstance(env, list):
            self.env_entry.insert(0, ",".join(env))
        else:
            self.env_entry.insert(0, str(env))

        budget = user_row["budget_range"]
        self.budget_low_entry.insert(0, str(budget[0]))
        self.budget_high_entry.insert(0, str(budget[1]))

        self.date_entry.insert(0, str(user_row["travel_date"]))

        old_id = user_row["user_id"]
        tk.Button(self.root, text="Save", command=lambda: self.replace_user(old_id)).pack(pady=10)

    def get_recommendations(self, users, properties):
        """
        Display a recommendation selection menu.

        Args:
            users (list[User]): List of user objects.
            properties (list[Property]): List of property objects.
        """
        for widget in self.root.winfo_children():
            widget.destroy()

        self.tree = ttk.Treeview(self.root)
        self.tree.pack(fill="both", expand=True)

        self.tree["columns"] = ("user_id",)
        self.tree["show"] = "headings"
        self.tree.heading("user_id", text="User ID")
        self.tree.column("user_id", width=100)

        for i in users:
            self.tree.insert("", "end", values=(i.user_id,))

        self.recommend_id = self.user_entry("Please select a User ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.print_recommendations(users, properties)).pack(pady=10)

    def print_recommendations(self, users, properties):
        """
        Display property recommendations for a selected user.

        Args:
            users (list[User]): List of user objects.
            properties (list[Property]): List of property objects.
        """
        target_id = int(self.recommend_id.get())

        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        text_widget = tk.Text(frame, wrap="word", width=100, height=20)
        text_widget.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=v_scroll.set)

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        top_properties = get_recommendation(users[target_id], properties).to_dict()

        for i, property_id in enumerate(top_properties["property_id"], start=1):
            line = (f"Number {i}\n"
                  f"{properties[top_properties['property_id'][property_id]]}\n"
                  f"Property Score:\t   {round(top_properties['total score'][property_id], 2)}\n"
                  f"Recommendation:\t   {top_properties['llm_recommendation'][property_id]}\n")
            text_widget.insert(tk.END, line)

        tk.Button(self.root, text="Main Menu", command=self.main_menu).pack(pady=10)

    def exit(self):
        """
        Exit the GUI application and show a farewell message.
        """
        messagebox.showinfo("Simple Airbnb", "Thank you for using Simple Airbnb!")
        self.root.withdraw()


def cli(properties: list[Property], users: list[User]):
    """
    Run the command-line interface (CLI) for managing users and properties.

    This interactive loop allows users to create and edit users or properties,
    view details, load or save data to files, and generate recommendations.

    Args:
        properties (list[Property]): The list of property objects to manage.
        users (list[User]): The list of user objects to manage.

    Options:
        1. Create a new user
        2. Create a new property
        3. View all properties
        4. View all users
        5. Edit a user
        6. Edit a property
        7. Load data from file
        8. Save data to file
        9. Generate recommendations
        10. Exit
    """
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
            environment = input(f"Enter Environment from {environments_pool}: ")
            if environment not in environments_pool:
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
              f"9. Get recommendations \t 10. Exit\n")
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
    """
        Main entry point of the application.

        Loads initial user and property data, and prompts the user to choose
        between running the Command-Line Interface (CLI) or the Graphical User
        Interface (GUI).

        Options:
            1. CLI - Run the text-based menu system.
            2. GUI - Run the Tkinter-based graphical interface.
    """
    # Main loop for CLI
    properties, users = load_from_file()
    while True:
        print(f"------------ Display Menu ------------ \n"
              f"1. CLI \n"
              f"2. GUI \n")
        user_input = input("Enter your choice: ")
        match user_input:
            case "1":
                cli(properties, users)
            case "2":
                app = GUI()
                app.run()        
        print("Invalid input")

def get_api():
    """
    Retrieve the API key for authenticating with the LLM API.

    The API key is first read from the environment variable `API_KEY`.
    If it is not found, the user will be prompted securely for input.

    Returns:
        str: The API key for authentication.
    """
    api_key = os.getenv("API_KEY")
    if api_key is None:
        api_key = getpass.getpass(prompt="Enter API Key: ")
    return api_key

class Llm:
    """
    A class representing a client for interacting with a Large Language Model (LLM) API.

    Attributes:
        api_key (str): The API key used for authentication.
        model (str): The identifier of the LLM model.
        url (str): The endpoint URL of the API.
        system_prompt (str): The system-level instruction defining assistant behavior.
        headers (dict): HTTP request headers including authorization and content type.

    Raises:
        ValueError: If the API response does not contain content.
    """
    def __init__(self, api_key: str, model: str, url: str, system_prompt: str):
        """
        Initialize a new Llm instance.

        Args:
            api_key (str): The API key for authentication.
            model (str): The name or identifier of the LLM model.
            url (str): The API endpoint URL.
            system_prompt (str): Instruction defining assistant role or behavior.
        """
        self.api_key = api_key
        self.model = model
        self.url = url
        self.system_prompt = system_prompt
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def update_api_key(self, api_key: str):
        """
        Update the API key and refresh authorization headers.

        Args:
            api_key (str): The new API key.
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def update_model(self, model: str):
        """
        Update the LLM model.

        Args:
            model (str): The new model identifier.
        """
        self.model = model

    def update_url(self, url: str):
        """
        Update the API endpoint URL.

        Args:
            url (str): The new endpoint URL.
        """
        self.url = url

    def llm_inquiry(self, user_prompt):
        """
        Send a query to the LLM API and return the response.

        Args:
            user_prompt (str): The prompt to send to the model.

        Returns:
            str: The model's text response.
            dict: Error details if the request fails.

        Raises:
            ValueError: If no content is returned in the response.
        """
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
