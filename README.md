# RSM8431

# LLM-Powered Summer Home Recommender

A command-line application based on python for recommending vacation rentals.  The project supports user/property profile management, provides an interactive CLI menu, and leaves placeholders for LLM-powered travel blurbs and synthetic property generation.


## Table of Contents
- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Data Model](#data-model)  
- [Environment & Installation](#environment--installation)  
- [How to Run](#how-to-run)  
- [CLI Menu & Example Flow](#cli-menu--example-flow)
- [GUI Menu & Example Flow](#gui-menu--example-flow) 
- [Recommendation Logic](#recommendation-logic)     
- [License](#license)  

---

## Overview

According to the course spec:  
- Manage user profiles (create/view/edit/delete)  
- Manage property listings (JSON dataset)  
- **Recommender logic** (budget + preference vector matching, Top-N output)  
- **CLI application**: `create_user`, `edit_profile`, `view_properties`, `get_recommendations`, `exit` 

---

## Project Structure

- `main.py` is the main entry point (CLI, data classes, I/O)
- `users.json` is the sample user dataset  
- `properties.json` is the sample property dataset

---

## Data Model

### Property (`Property` class in `main.py`)
- `property_id: int`  
- `location: str`  
- `[property_type: str` (e.g. cabin, condo, chalet)  
- `price_per_night: float`  
- `features: list[str]`  
- `tags: list[str]`
- `max_guests: int`
- `environment: str`

Methods: `update_id / update_type / update_price_per_night / update_location / update_tags / update_features / get_dict ...`

### User (`User` class in `main.py`)
- `user_id: int`  
- `name: str`  
- `group_size: int`  
- `preferred_environment: list[str]` (lake, mountain, beach, city, etc.)  
- `budget_range: tuple[int, int]`  
- `travel_date: datetime`  

Methods: `update_name / update_id / update_budget_range / update_travel_date / update_group_size / update_preferred_environment / get_dict ...`

> File I/O: `load_from_file()` and `write_to_file()` are hardcoded to the two JSON files.

---

## Environment & Installation

- Python 3.9+ (recommended: 3.10/3.11)  
- Dependencies: `datetime`, `json`, `getpass`, `requests`, `numpy`, `pandas`, `tkinter`

```bash
# Clone
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

# (Optional) Virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# (Future) Install deps if added
pip install -r requirements.txt
```

---

## How to Run

```bash
# Fix file names
mv "properties.json" properties.json
mv "users.json" users.json

# Run
python main.py
```
You’ll see the top-level menu (main()):

------------ Display Menu ------------ 
1. CLI 
2. GUI 
Enter your choice:

- Choose 1 → CLI main menu
- Choose 2 → GUI main menu

---

## CLI Menu & Example Flow

cli() menu:

```bash
-------------------- Main Menu --------------------
1. Create a new user      2. Create a new property
3. View properties        4. View users
5. Edit user              6. Edit property
7. Load from file         8. Save to file
9. Get recommendations    10. Exit
```

#### 1) Create a new user
Prompts:
- User ID (int)
- Name (str)
- Group Size (int)
- Preferred Environment (multiple; enter F to finish)
- Budget Range (lower & upper ints)
- Travel Date (must be YYYY-MM-DD HH:MM:SS.FF with microseconds, or N for now)

#### 2) Create a new property
Prompts:
- Property ID (int)
- Location (str)
- Type (str)
- Price per Night (float)
- Features (multiple; F to finish)
- Tags (multiple; F to finish)

#### 3&4)View properties/users
Prints objects using __str__.

#### 5/6) Edit user/property
Select ID, then update fields via menu.

#### 7/8) Load/Save from file
Hardcoded JSON file read/write.

#### 9) Get recommendations
User will view a list of User IDs, and be prompted to select a User ID. User info will be input and redirected to LLM model, which will output the five top property recommendations.

#### 10) Exit

---

## GUI Menu & Example Flow

When the GUI is selected, an external tab will appear. User will view the main menu and be prompted to enter one of the following options into the text box:

```bash
-------------------- Main Menu --------------------
1. Create a new user      
2. Create a new property
3. View properties        
4. View users
5. Edit user              
6. Edit property
7. Load from file         
8. Save to file
9. Get recommendations    
10. Exit
```

#### 1) Create a new user
The user will be prompted to fill out the following textboxes, labelled above: 
- User ID (int)
- Name (str)
- Group Size (int)
- Preferred Environment (comma-separated list)
- Budget Range (lower & upper ints)
- Travel Date (must be YYYY-MM-DD HH:MM:SS.FF with microseconds, or N for now)
- Save button - user will be redirected to main menu and receive a success message.

#### 2) Create a new property
The user will be prompted to fill out the following textboxes, labelled above: 
- Property ID (int)
- Location (str)
- Type (str)
- Price per Night (float)
- Max guests (int)
- Features (comma-separated list)
- Tags (comma-separated list)
- Environment (comma-separated list)
- Save button - user will be redirected to main menu and receive a success message.

#### 3&4)View properties/users
The user will be prompted with two buttons: 
- I already know my User ID/Property ID - user will be prompted to enter User ID/Property ID, and selected User ID/Property ID will be printed out. 
- View all - all users/properties will be printed out. 

For both options, on viewing page, user will have a button which will redirect the mto the main menu.

#### 5/6) Edit user/property
Select ID, then update fields via menu. Data from existing users/properties will appear in textbox fields for editing. Save button will update fields, and user will be redirected to main menu.

#### 7/8) Load/Save from file
Hardcoded JSON file read/write. User will be redirected to main menu and receive a success message.

#### 9) Get recommendations
User will view a list of User IDs, and be prompted to select a User ID. User info will be input and redirected to LLM model, which will output the five top property recommendations. User will have the option to return to main menu after receiving recommendations. 

#### 10) Exit
User will receive a thank you message and tab will close.

---

## Recommendation Logic
1. Scoring and Filtering 
* Properties are ranked based on five points: Budget, Capacity, Environment, Features, and LLM. Filtering is done through penalizing properties which have less similarity to user preferences. 
* Budget: Properties are scored based on closeness to user's budget range. Properties that fall above the user's budget are penalized to ensure adherence to user preferences.
* Capacity: Properties are scored based on closeness to user's group size. Properties that are too small and too large for user's group size are penalized to ensure adherence to user preferences.
* Environment and Features: Properties are scored based on shared qualities to user's preferred environment/preferred features. Properties that contain more preferred environment/features are ranked higher to ensure adherence to user preferences.
* LLM: An LLM is used to score properties based on overall match to user's input preferences in all categories. The model returns a score for how much similarity each property has to user preferences. 

Each score is normalized on a scale of 1-10, and a weighted average of the five scores is computed. This weighted average is used as the overall property score for the selected user. 

2. Ranking and Output
* Sort by score
* Print top 5 properties 
