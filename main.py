from datetime import datetime
import json, requests, getpass

import numpy
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

ENVIRONMENTS = ("mountain","lake","beach","city","rural","suburban","desert","forest","ski","island")
PROPERTY_TYPES = ("apartment","house","cabin","villa","condo","townhome","bnb", "chalet", "cottage", "loft")
FEATURES = ("hot_tub","fireplace","wifi","kitchen","parking","pool","pet_friendly","ev_charger", "gym", "bbq",
            "patio", "garden", "beach access", "canoe", "kayak", "air conditioning", "washer", "dryer")

LOCATIONS = (
    "Lake Muskoka", "Toronto Downtown", "Blue Mountain",
    "Niagara-on-the-Lake", "Prince Edward County", "Collingwood",
    "Wasaga Beach", "Kingston", "Ottawa", "Halifax")


TAGS_POOL = [
    "lakefront", "beachfront", "family-friendly", "pets", "luxury",
    "urban", "nightlife", "business", "mountains", "romantic", "quiet", "nature"
]


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
        # 10 = mean of user's budget range
        # Z score normed in 0 - 10 for rest points
        budget_mean = sum(self.budget_range) / 2
        budget_std = max(np.std(self.budget_range), 0.00001)
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
        properties_df["LLM score"] = llm_score()

        properties_df["total score"] = properties_df["price score"] + properties_df["capacity score"] + properties_df["environment score"] + properties_df["feature score"] + properties_df["LLM score"]
        return properties_df

def llm_score(): return 0
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
                                    prop['features'], prop['tags'], prop['max_guests'], prop['environment']) for prop in temp_properties]
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
    type_dummy = pd.get_dummies(df['property_type'], prefix='type')
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

########################################################################################################################
class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Airbnb") # change title
        self.root.geometry("500x500")

        self.main_menu()

    def main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.label = tk.Label(self.root, text="Welcome to Simple Airbnb! Please select an option: \n\n Main Menu")
        self.label.pack(pady=10)

        self.listbox = tk.Listbox(self.root, width=40, height=15)
        self.listbox.pack()

        list_of_options = ["1. Create a new user", "2. Create a new property", "3. View properties",
                          "4. View users", "5. Edit user", "6. Edit property", "7. Load from file",
                          "8. Save to file", "9. Get recommendations", "10. LLM summary", "11. Exit"]
        for item in list_of_options:
            self.listbox.insert(tk.END, item)

        self.entry = tk.Entry(self.root, width=30)
        self.entry.pack(pady=10)

        self.button = tk.Button(self.root, text="Enter", command=self.redirect_input)
        self.button.pack(pady=20)

    def redirect_input(self):
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
            self.view_property(properties_file)
        elif user_input == 4:
            self.view_user(users_file)
        elif user_input == 5:
            self.edit_user(users_file)
        elif user_input == 6: 
            self.edit_property(properties_file)
        elif user_input == 7: # load from file
            pass
        elif user_input == 8: # save from file
            pass
        elif user_input == 9: # get recommendation
            pass
        elif user_input == 10: # llm summary
            pass
        elif user_input == 11: # exit
            pass

    def run(self):
        self.root.mainloop()

    # collects user entries and adds label
    def user_entry(self, label):
        tk.Label(self.root, text=label).pack()
        user_entry = tk.Entry(self.root)
        user_entry.pack()
        return user_entry

    # user entry labels
    # it's more difficult to do the one by one entries like the cli so its all in one form rn for the gui
    def create_user(self):
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

        User(user_id, name, group_size, preferred_environment, budget_range, travel_date)

        messagebox.showinfo("Simple Airbnb", "User has been saved!")

        self.main_menu()

    def create_property(self):
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
        property_id = self.property_id_entry.get()
        location = self.location_entry.get()
        loc_type = self.type_entry.get()
        price_per_night = self.price_entry.get()
        max_guests = self.max_guests_entry.get()
        features = [i.strip() for i in self.features_entry.get().split(",")]
        loc_tags = [i.strip() for i in self.tags_entry.get().split(",")]
        env = self.environment_entry.get()

        Property(property_id, location, loc_type, price_per_night, features, loc_tags, max_guests, env)

        messagebox.showinfo("Simple Airbnb", "Property has been saved!")

        self.main_menu()

    def view_property(self, properties):
        for widget in self.root.winfo_children():
            widget.destroy()

        properties_df = pd.DataFrame(properties)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        one_property = tk.Button(button_frame, text="I know my Property ID", command=lambda: self.view_one_property(properties_df))
        one_property.pack(side="left", padx=10)

        all_property = tk.Button(button_frame, text="View All", command=lambda: self.view_all_property(properties_df))
        all_property.pack(side="right", padx=10)

    def view_one_property(self, properties):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.selected_property_id = self.user_entry("Please enter Property ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.get_property_id(properties)).pack(pady=10)

    def get_property_id(self, properties):
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
        for widget in self.root.winfo_children():
            widget.destroy()

        self.tree = ttk.Treeview(self.root)
        self.tree.pack(fill="both", expand=True)

        self.tree["columns"] = list(properties.columns)
        self.tree["show"] = "headings"

        for col in properties.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in properties.iterrows():
            self.tree.insert("", "end", values=list(row))

        tk.Button(self.root, text="Main Menu", command=self.main_menu).pack(pady=10)

    def view_user(self, users):
        for widget in self.root.winfo_children():
            widget.destroy()

        users_df = pd.DataFrame(users)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        one_property = tk.Button(button_frame, text="I know my User ID", command=lambda: self.view_one_user(users_df))
        one_property.pack(side="left", padx=10)

        all_property = tk.Button(button_frame, text="View All", command=lambda: self.view_all_user(users_df))
        all_property.pack(side="right", padx=10)

    def view_one_user(self, users):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.selected_user_id = self.user_entry("Please enter User ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.get_user_id(users)).pack(pady=10)

    def get_user_id(self, users):
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
        for widget in self.root.winfo_children():
            widget.destroy()

        self.tree = ttk.Treeview(self.root)
        self.tree.pack(fill="both", expand=True)

        self.tree["columns"] = list(users.columns)
        self.tree["show"] = "headings"

        for col in users.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in users.iterrows():
            self.tree.insert("", "end", values=list(row))

        tk.Button(self.root, text="Main Menu", command=self.main_menu).pack(pady=10)

    def edit_property(self, properties):
        for widget in self.root.winfo_children():
            widget.destroy()

        properties_df = pd.DataFrame(properties)

        tk.Label(self.root, text="Edit Property").pack(pady=5)

        self.editing_property_id = self.user_entry("Please enter Property ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.edit_property_values(properties_df)).pack(pady=10)

    def edit_property_values(self, properties):
        edit_property = properties[properties["property_id"] == int(self.editing_property_id.get())]
        property_row = edit_property.iloc[0]

        for widget in self.root.winfo_children():
            widget.destroy()

        self.property_id_entry = self.user_entry("Property ID:")
        self.location_entry = self.user_entry("Location:")
        self.type_entry = self.user_entry("Type:")
        self.price_entry = self.user_entry("Price for Night:")
        self.max_value_entry = self.user_entry("Max Value:")
        self.features_entry = self.user_entry("Features:")
        self.tags_entry = self.user_entry("Tags:")
        self.environment_entry = self.user_entry("Environment:")

        self.property_id_entry.insert(0, str(property_row["property_id"]))
        self.location_entry.insert(0, str(edit_property["location"]))
        self.type_entry.insert(0, str(edit_property["type"]))
        self.price_entry.insert(0, str(property_row["price_per_night"]))
        self.max_value_entry.insert(0, str(property_row["max_value"]))
        self.features_entry.insert(0, ",".join(property_row["features"]))
        self.tags_entry.insert(0, ",".join(property_row["tags"]))
        self.environment_entry.insert(0, ",".join(property_row["environment"]))

        tk.Button(self.root, text="Save", command=self.submit_property).pack(pady=10)

    def edit_user(self, users):
        for widget in self.root.winfo_children():
            widget.destroy()

        users_df = pd.DataFrame(users)

        tk.Label(self.root, text="Edit User").pack(pady=5)

        self.editing_user_id = self.user_entry("Please enter User ID:")
        tk.Button(self.root, text="Enter", command=lambda: self.edit_user_values(users_df)).pack(pady=10)

    def edit_user_values(self, users):
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
        self.env_entry.insert(0, ",".join(user_row["preferred_environment"]))
        self.budget_low_entry.insert(0, str(edit_user["budget_range"][0][0]))
        self.budget_high_entry.insert(0, str(edit_user["budget_range"][0][1]))
        self.date_entry.insert(0, str(user_row["travel_date"]))

        tk.Button(self.root, text="Save", command=self.submit_user).pack(pady=10)

########################################################################################################################

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
                gui = GUI()
                gui.run()
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
