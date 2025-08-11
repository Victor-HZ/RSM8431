from datetime import datetime
import json


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