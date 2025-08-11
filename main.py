from datetime import datetime


class Property:
    def __init__(self, id: int, location: str, type: str, nightly_price: float, features: list[str], tags: list[str]):
        self.id = id
        self.location = location
        self.type = type
        self.nightly_price = nightly_price
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
                f"ID: {self.id}"
                f"Location: ${self.location}\n"
                f"Type: ${self.type}\n"
                f"Nightly Price: ${self.nightly_price}\n"
                f"Features: ${feature_list}\n"
                f"Tags: ${tag_list}\n"
                f"--------------------------------------")

    # def load_from_file


class User:
    def __init__(self, user_id: int, name: str, group_size: int, preferred_environment: list[str], budget_range: tuple[int, int], travel_date: datetime = datetime.now()):
        self.user_id = user_id
        self.name = name
        self.group_size = group_size
        self.preferred_environment = preferred_environment
        self.budget_range = budget_range
        self.travel_date = travel_date


