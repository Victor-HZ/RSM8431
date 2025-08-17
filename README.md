# RSM8431

# LLM-Powered Summer Home Recommender

A command-line application based on python for recommending vacation rentals.  The project supports user/property profile management, provides an interactive CLI menu, and leaves placeholders for LLM-powered travel blurbs and synthetic property generation (not yet implemented).


## Table of Contents
- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Data Model](#data-model)  
- [Environment & Installation](#environment--installation)  
- [How to Run](#how-to-run)  
- [CLI Menu & Example Flow](#cli-menu--example-flow)  
- [Recommendation Logic (Planned)](#recommendation-logic-planned)    
- [Roadmap & TODO](#roadmap--todo)  
- [Weaknesses & Improvement Checklist](#weaknesses--improvement-checklist)  
- [License](#license)  

---

## Overview

According to the course spec:  
- Manage user profiles (create/view/edit/delete)  
- Manage property listings (JSON dataset)  
- **Recommender logic** (budget + preference vector matching, Top-N output)  
- **CLI application**: `create_user`, `edit_profile`, `view_properties`, `get_recommendations`, `llm_summary`, `exit` (current code’s menu differs slightly; see below)   

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
- `type: str` (e.g. cabin, condo, chalet)  
- `price_per_night: float`  
- `features: list[str]`  
- `tags: list[str]`  

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
- Dependencies: currently only standard library (`datetime`, `json`, `getpass`, `requests`—not all used yet)  

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
- Choose 2 → GUI (currently not implemented)

---

## CLI Menu & Example Flow

cli() menu (actual formatting may differ):

```bash
-------------------- Main Menu --------------------
1. Create a new user      2. Create a new property
3. View properties        4. View users
5. Edit user              6. Edit property
7. Load from file         8. Save to file
9. Get recommendations    10. LLM summary
11. Exit
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
Not implemented (get_recommendation() is empty).

#### 10) LLM summary
Not implemented (llm_summary() is empty).

#### 11) Exit

---

## Recommendation Logic 

Per the spec:
1) Filtering 
- Budget range: price_per_night ∈ [budget_min, budget_max]
- Group size: (currently missing capacity field; can’t filter)
2) Scoring
- Price matches: Identifies properties closest to user budget
- Feature/tag matches: e.g. wifi, lakefront, hot tub - (naively) identifies properties which have the most similarities closest to users preferred environment 
- Weighted sum: 
```
score = 0.6*price_score + 0.4*feature_score
```
3) Ranking & Output
- Sort by score
- Print Top-N (e.g., top 5)

---

## Roadmap & TODO
* GUI
* GUI & CLI function sharing
* Vectorized Matching (No idea at all)
* Property Scoring
* Docstring
* Readme
* Slides

---

## Weaknesses & Improvement Checklist
1) No explainability in recommendations
- Even if the recommendation logic works, the current design doesn’t explain why a property is recommended.
- You should display scoring breakdowns (e.g., budget closeness, environment match, feature hits) so graders see the logic is meaningful.

2) No testing or data validation
- This can corrupt JSON files or produce misleading recommendations.

---

## License

MIT (recommend adding a LICENSE file at repo root)
