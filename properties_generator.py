import random
import datetime

import main

def make_random_property(property_id: int):
    location = random.choice(main.LOCATIONS)
    ptype = random.choice(main.PROPERTY_TYPES)
    price = random.randint(100, 500)
    features = random.sample(main.FEATURES, k=random.randint(2, 4))
    tags = random.sample(main.TAGS_POOL, k=random.randint(2, 3))
    max_guests = random.randint(1, 10)
    penv = random.choice(main.ENVIRONMENTS)
    return main.Property(property_id, location, ptype, price, features, tags, max_guests, penv)

def make_randome_user(user_id: int):
    name = "UN: " + str(user_id)
    group_size = random.randint(1, 10)
    penv = random.choice(main.ENVIRONMENTS)
    budget = random.randint(70, 150), random.randint(120, 400)
    travel_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    return main.User(user_id, name, group_size, penv, budget, travel_time)

random.seed(42)
NUM_PROPERTIES = 25
properties = [make_random_property(i) for i in range(1, NUM_PROPERTIES + 1)]
users = [make_randome_user(i) for i in range(1, NUM_PROPERTIES + 1)]

main.write_to_file(properties, users)

p, u = main.load_from_file()

u[0].score_properties(p)

