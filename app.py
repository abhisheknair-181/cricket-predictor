import streamlit as st
import joblib
import pandas as pd
import requests
import time
from typing import Dict, Any, List, Optional, Tuple

# --- 1. CONFIGURATION ---

# IMPORTANT: This MUST be the path to your new model trained on ONLY the 7 features.
MODEL_PATH = "live_model_7features.joblib" 
API_KEY = "2f80cbd3-73d4-4823-be9a-2af40b6ba3a8" # Your CricAPI key
BASE_URL = "https://api.cricapi.com/v1"
REFRESH_INTERVAL_SECONDS = 20 # How often to refresh live data

#
# !!! CRITICAL !!!
# You MUST replace this placeholder with your actual venue_rating mapping.
# This map should be generated from your training notebook and saved.
# e.g., VENUE_RATING_MAP = pd.read_csv('venue_ratings.csv').set_index('venue')['rating'].to_dict()
#
VENUE_RATING_MAP = {
    'Wanderers Ground': 2,
    'Wanderers Namibia': 2,
    'GMHBA Stadium, South Geelong, Victoria': 1,
    'Sabina Park': 2,
    'Eden Gardens': 2,
    'MA Chidambaram Stadium, Chepauk, Chennai': 4,
    'Dubai International Cricket Stadium': 2,
    'Gaddafi Stadium': 1,
    'Gymkhana Club Ground, Nairobi': 3,
    'Sophia Gardens': 3,
    'Harare Sports Club': 2,
    'Providence Stadium, Guyana': 3,
    'ICC Global Cricket Academy': 2,
    'United Cricket Club Ground, Windhoek': 3,
    'ICC Academy Ground No 2': 2,
    'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)': 3,
    'SuperSport Park, Centurion': 4,
    'Rangiri Dambulla International Stadium': 2,
    'R Premadasa Stadium': 3,
    'Entebbe Cricket Oval': 1,
    'Sheikh Zayed Stadium': 3,
    'National Cricket Stadium Grenada': 1,
    'Gahanga International Cricket Stadium. Rwanda': 4,
    'Eden Park': 2,
    'Shere Bangla National Stadium, Mirpur': 2,
    'Newlands': 3,
    'Edgbaston, Birmingham': 0,
    'Nassau County International Cricket Stadium, New York': 3,
    'The Village, Malahide': 2,
    'Zahur Ahmed Chowdhury Stadium': 2,
    'Grange Cricket Club Ground, Raeburn Place, Edinburgh': 2,
    'Mission Road Ground, Mong Kok, Hong Kong': 2,
    'Terdthai Cricket Ground': 2,
    'Achimota Senior Secondary School A Field, Accra': 2,
    'Pallekele International Cricket Stadium': 2,
    'Kinrara Academy Oval': 1,
    'Stadium Australia': 2,
    'Vidarbha Cricket Association Stadium, Jamtha': 1,
    'Melbourne Cricket Ground': 3,
    'Zayed Cricket Stadium, Abu Dhabi': 3,
    'Sir Vivian Richards Stadium, North Sound, Antigua': 3,
    'Tribhuvan University International Cricket Ground': 2,
    'ICC Academy': 1,
    'Windsor Park, Roseau': 2,
    'National Stadium Karachi': 1,
    'Edgbaston': 0,
    'Sportpark Westvliet': 2,
    'Mulpani Cricket Ground': 3,
    'Shere Bangla National Stadium': 2,
    'Kensington Oval': 2,
    'Warner Park, Basseterre, St Kitts': 4,
    'Central Broward Regional Park Stadium Turf Ground': 1,
    'Bellerive Oval, Hobart': 2,
    'Gahanga International Cricket Stadium, Rwanda': 2,
    'Castle Avenue, Dublin': 4,
    'Marrara Stadium, Darwin': 0,
    'Brisbane Cricket Ground, Woolloongabba, Brisbane': 0,
    'Kennington Oval': 2,
    'Bready Cricket Club, Magheramason, Bready': 0,
    'Punjab Cricket Association Stadium, Mohali': 4,
    'Headingley, Leeds': 0,
    'Willowmoore Park, Benoni': 4,
    'UKM-YSD Cricket Oval, Bangi': 4,
    'University of Doha for Science and Technology': 4,
    'Khan Shaheb Osman Ali Stadium': 2,
    'Beausejour Stadium, Gros Islet': 2,
    'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)': 2,
    'County Ground': 2,
    'Hagley Oval, Christchurch': 3,
    'MA Chidambaram Stadium, Chepauk': 2,
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium': 0,
    'Selangor Turf Club, Kuala Lumpur': 2,
    'Mangaung Oval': 0,
    'Providence Stadium': 3,
    'Westpac Stadium': 2,
    'Bay Oval': 0,
    'ICC Academy, Dubai': 3,
    'West End Park International Cricket Stadium, Doha': 2,
    'Bay Oval, Mount Maunganui': 1,
    'Indian Association Ground': 3,
    'YSD-UKM Cricket Oval, Bangi': 4,
    'Maharashtra Cricket Association Stadium, Pune': 0,
    'Old Trafford': 3,
    'Riverside Ground, Chester-le-Street': 2,
    'Mission Road Ground, Mong Kok': 2,
    'Seddon Park': 2,
    'Sharjah Cricket Stadium': 2,
    'Kensington Oval, Bridgetown': 1,
    'Kingsmead': 2,
    'Lugogo Cricket Oval': 0,
    'Perth Stadium': 2,
    'Bellerive Oval': 1,
    'Sportpark Maarschalkerweerd, Utrecht': 2,
    'Prairie View Cricket Complex': 1,
    'Sikh Union Club Ground, Nairobi': 3,
    'Wankhede Stadium': 3,
    'Warner Park, St Kitts': 3,
    'Grand Prairie Stadium, Dallas': 4,
    'The Rose Bowl, Southampton': 0,
    'Maple Leaf North-West Ground, King City': 3,
    'Carrara Oval': 2,
    'Riverside Ground': 1,
    'Sylhet Stadium': 4,
    'Adelaide Oval': 2,
    'SuperSport Park': 1,
    'Sydney Cricket Ground': 2,
    'Mombasa Sports Club Ground': 3,
    'Barsapara Cricket Stadium, Guwahati': 2,
    'Holkar Cricket Stadium': 1,
    'Arnos Vale Ground': 0,
    'Integrated Polytechnic Regional Centre': 2,
    'Rajiv Gandhi International Stadium, Uppal': 4,
    'Queens Sports Club, Bulawayo': 2,
    'Brabourne Stadium': 4,
    'Shrimant Madhavrao Scindia Cricket Stadium, Gwalior': 4,
    'Singapore National Cricket Ground': 4,
    'Brisbane Cricket Ground, Woolloongabba': 2,
    'The Rose Bowl': 2,
    'Barabati Stadium': 2,
    'Hazelaarweg': 1,
    'Bulawayo Athletic Club': 4,
    'Wanderers Cricket Ground': 4,
    'Tafawa Balewa Square Cricket Oval, Lagos': 3,
    'Trent Bridge': 2,
    'Buffalo Park': 1,
    'TCA Oval, Blantyre': 4,
    'Sky Stadium, Wellington': 4,
    'Gymkhana Club Ground': 3,
    'Coolidge Cricket Ground, Antigua': 3,
    'Daren Sammy National Cricket Stadium, Gros Islet, St Lucia': 2,
    'Sylhet International Cricket Stadium': 2,
    'Brian Lara Stadium, Tarouba, Trinidad': 2,
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 2,
    'Zhejiang University of Technology Cricket Field': 2,
    'The Village, Malahide, Dublin': 1,
    'Senwes Park': 2,
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 4,
    'VRA Ground': 2,
    'Tolerance Oval': 3,
    'Bayuemas Oval, Kuala Lumpur': 3,
    'Kyambogo Cricket Oval': 3,
    'Titwood, Glasgow': 2,
    'Central Broward Regional Park Stadium Turf Ground, Lauderhill': 2,
    'JSCA International Stadium Complex': 2,
    'Kingsmead, Durban': 2,
    'Bready': 4,
    'Malahide, Dublin': 2,
    'Arun Jaitley Stadium': 2,
    'Manuka Oval': 2,
    'Rawalpindi Cricket Stadium': 4,
    "Queen's Park Oval": 2,
    "Lord's": 1,
    'Saurashtra Cricket Association Stadium': 3,
    "St George's Park": 1,
    "Cazaly's Stadium, Cairns": 4,
    'Ruaraka Sports Club Ground, Nairobi': 2,
    'Barsapara Cricket Stadium': 4,
    'M Chinnaswamy Stadium': 3,
    'Warner Park, Basseterre': 4,
    'JSCA International Stadium Complex, Ranchi': 2,
    'Maharashtra Cricket Association Stadium': 2,
    'Jimmy Powell Oval, Cayman Islands': 4,
    'Narendra Modi Stadium': 3,
    'Namibia Cricket Ground, Windhoek': 4,
    'Greenfield International Stadium': 2,
    'Seddon Park, Hamilton': 0,
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 2,
    'Civil Service Cricket Club, Stormont, Belfast': 1,
    'Saxton Oval, Nelson': 0,
    'Tony Ireland Stadium': 0,
    "St George's Park, Gqeberha": 4,
    'Grange Cricket Club Ground, Raeburn Place': 3,
    'Moses Mabhida Stadium': 0,
    'Grange Cricket Club, Raeburn Place': 0,
    'Vidarbha Cricket Association Stadium, Jamtha, Nagpur': 4,
    'Himachal Pradesh Cricket Association Stadium': 1,
    'Green Park': 4,
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 2,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 2,
    'University Oval': 0,
    'Sky Stadium': 4,
    'Civil Service Cricket Club, Stormont': 4,
    'Old Trafford, Manchester': 2,
    'Sir Vivian Richards Stadium, North Sound': 1,
    'Bready Cricket Club, Magheramason': 3,
    'Sheikh Abu Naser Stadium': 1,
    'University Oval, Dunedin': 3,
    'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa': 1,
    'White Hill Field, Sandys Parish': 3,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 4,
    'Arun Jaitley Stadium, Delhi': 2,
    'Barabati Stadium, Cuttack': 4,
    'Queens Sports Club': 0,
    'Kennington Oval, London': 4,
    'M.Chinnaswamy Stadium': 4,
    'Udayana Cricket Ground': 4,
    'Maple Leaf North-West Ground': 4,
    'Boland Park': 4,
    'Subrata Roy Sahara Stadium': 4,
    'Western Australia Cricket Association Ground': 2,
    'Achimota Senior Secondary School B Field, Accra': 3,
    'Windsor Park, Roseau, Dominica': 0,
    'Darren Sammy National Cricket Stadium, St Lucia': 4,
    'Saxton Oval': 0,
    'P Sara Oval': 4,
    'Sawai Mansingh Stadium, Jaipur': 4,
    'Sportpark Het Schootsveld': 0,
    'Feroz Shah Kotla': 3,
    'OUTsurance Oval': 4,
    'Shaheed Veer Narayan Singh International Stadium, Raipur': 0,
    'Arnos Vale Ground, Kingstown': 2,
    'AMI Stadium': 2,
    'Narendra Modi Stadium, Ahmedabad': 0,
    'Simonds Stadium, South Geelong': 4,
    'United Cricket Club Ground': 4,
    'Saurashtra Cricket Association Stadium, Rajkot': 0,
    'University of Lagos Cricket Oval': 4,
    'Jade Stadium': 0,
    'Hagley Oval': 2,
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 4,
    'Goldenacre, Edinburgh': 4,
    'McLean Park': 2,
    'De Beers Diamond Oval': 0,
    'Jinja Cricket Ground': 2,
    'Indian Association Ground, Singapore': 4,
    'McLean Park, Napier': 4,
    'John Davies Oval, Queenstown': 4,
    'Sardar Patel Stadium, Motera': 0,
    'Nigeria Cricket Federation Oval 1, Abuja': 4,
    'M Chinnaswamy Stadium, Bengaluru': 0,

    # --- CRITICAL FALLBACK ---
    # This 'default' key is essential. It handles any new venues
    # from the live API that were not in your training data.
    # We use '2' as it is the neutral/average rating in your 0-4 scale.
    'default': 2
}

# --- 2. MODEL & DATA LOADING (Cached) ---

@st.cache_resource
def load_model(model_path):
    """Loads the trained model pipeline from disk."""
    try:
        model = joblib.load(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"FATAL: Model file '{model_path}' not found. Please train your 7-feature model and save it to this folder.")
        print(f"Error: Model file '{model_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

@st.cache_data(ttl=60) # Cache the list of matches for 60 seconds
def fetch_live_match_list() -> List[Dict[str, Any]]:
    """Fetches the list of all currently live matches."""
    url = f"{BASE_URL}/currentMatches"
    params = {"apikey": API_KEY, "offset": 0}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        
        # Filter for T20 matches that have started
        live_t20_matches = [
            match for match in data
            if match.get('matchType') == 't20' and match.get('matchStarted')
        ]
        return live_t20_matches
    except requests.exceptions.RequestException as e:
        st.error(f"API Error fetching match list: {e}")
        return []

def fetch_live_score(match_id: str) -> Optional[Dict[str, Any]]:
    """Fetches the detailed score for a single match. NOT cached."""
    # Using 'match_score' endpoint for detailed ball-by-ball (if bbbEnabled)
    # or at least detailed scorecards.
    url = f"{BASE_URL}/match_score"
    params = {"apikey": API_KEY, "id": match_id}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get('data')
    except requests.exceptions.RequestException as e:
        # Don't show an error on the UI, as it will pop up on every refresh
        print(f"Error fetching score (will retry): {e}")
        return None

# --- 3. DATA PARSING & FEATURE ENGINEERING ---

def parse_live_data(score_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the full API score data into the 7 features and UI stats.
    Returns a dictionary with:
    - 'status': "WAITING", "IN_PLAY", or "COMPLETE"
    - 'model_input': A DataFrame for the model (if in_play)
    - 'stats': A dictionary of rich stats for the UI
    """
    try:
        scores = score_data.get('score', [])
        venue = score_data.get('venue', 'Unknown Venue')
        teams = score_data.get('teams', ['Team 1', 'Team 2'])
        
        # --- Get Team Names ---
        team1_name = teams[0]
        team2_name = teams[1]
        
        # --- Check Match State ---
        if not scores or len(scores) == 0:
            return {"status": "WAITING", "stats": {"message": "Match has not started."}}
        
        if score_data.get('matchEnded'):
            return {"status": "COMPLETE", "stats": {"message": score_data.get('status', 'Match Ended')}}

        # --- Innings 1 in Progress ---
        if len(scores) == 1:
            inn1 = scores[0]
            # Figure out who is batting
            batting_team_name = team1_name if inn1['inning'].startswith(team1_name) else team2_name
            bowling_team_name = team2_name if batting_team_name == team1_name else team1_name
            
            stats = {
                "message": "Innings 1 in progress. Waiting for chase to begin.",
                "batting_team": batting_team_name,
                "bowling_team": bowling_team_name,
                "score_str": f"{inn1['r']}/{inn1['w']} ({inn1['o']})",
                "venue": venue
            }
            return {"status": "WAITING", "stats": stats}

        # --- Innings 2 in Progress (The main logic) ---
        if len(scores) >= 2:
            inn1 = scores[0]
            inn2 = scores[1]
            
            # Identify batting and bowling teams
            # The chasing team is the one *not* batting in innings 1
            team1_bat_first = inn1['inning'].startswith(team1_name)
            batting_team_name = team2_name if team1_bat_first else team1_name
            bowling_team_name = team1_name if team1_bat_first else team2_name

            # --- Calculate Model Features ---
            target = inn1['r'] + 1
            current_runs = inn2['r']
            current_wickets = inn2['w']
            current_overs_float = inn2['o']
            
            runs_required = target - current_runs
            wickets_remaining = 10 - current_wickets
            
            total_balls = 120
            overs_int = int(current_overs_float)
            balls_in_over = int(round((current_overs_float - overs_int) * 10))
            balls_bowled = (overs_int * 6) + balls_in_over
            
            balls_left = total_balls - balls_bowled
            
            if balls_left <= 0 or wickets_remaining <= 0 or runs_required <= 0:
                return {"status": "COMPLETE", "stats": {"message": score_data.get('status', 'Match Ended')}}

            current_run_rate = (current_runs * 6) / balls_bowled if balls_bowled > 0 else 0
            required_run_rate = (runs_required * 6) / balls_left if balls_left > 0 else float('inf')
            
            venue_rating = VENUE_RATING_MAP.get(venue, VENUE_RATING_MAP['default'])

            # Create the 7-feature DataFrame for the model
            model_input_df = pd.DataFrame([{
                "balls_left": balls_left,
                "runs_required": runs_required,
                "wickets_remaining": wickets_remaining,
                "current_run_rate": current_run_rate,
                "required_run_rate": required_run_rate,
                "venue_rating": venue_rating,
                "current_wickets": current_wickets
            }])
            
            # Create rich stats for the UI
            stats = {
                "message": score_data.get('status', 'Match in Progress'),
                "batting_team": batting_team_name,
                "bowling_team": bowling_team_name,
                "target": target,
                "score_str": f"{current_runs}/{current_wickets} ({current_overs_float})",
                "runs_required": runs_required,
                "balls_left": balls_left,
                "wickets_remaining": wickets_remaining,
                "current_run_rate": f"{current_run_rate:.2f}",
                "required_run_rate": f"{required_run_rate:.2f}",
                "venue": venue
            }
            
            return {"status": "IN_PLAY", "model_input": model_input_df, "stats": stats}
            
        return {"status": "WAITING", "stats": {"message": "Waiting for match data..."}}

    except Exception as e:
        print(f"Error in parse_live_data: {e}")
        return {"status": "ERROR", "stats": {"message": f"An error occurred: {e}"}}

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="T20 Win Predictor", layout="wide")
model = load_model(MODEL_PATH)

# --- VIEW 1: MATCH SELECTION (HOME PAGE) ---
# We use session_state to manage which page the user is on.
if 'selected_match_id' not in st.session_state:
    st.session_state.selected_match_id = None

if st.session_state.selected_match_id is None:
    st.title("Live T20 Match Selector")
    
    match_list = fetch_live_match_list()
    
    if not match_list:
        st.warning("No live T20 matches found. Please check API key or try again later.")
    else:
        # Create a dictionary of 'Match Name' -> 'match_id'
        match_options = {match['name']: match['id'] for match in match_list}
        
        selected_match_name = st.selectbox(
            "Select a live T20 match to track:",
            options=match_options.keys()
        )
        
        if st.button("Start Prediction Dashboard"):
            st.session_state.selected_match_id = match_options[selected_match_name]
            st.session_state.match_name = selected_match_name # Store name for display
            st.rerun() # Re-run the script to load the dashboard view

# --- VIEW 2: LIVE PREDICTION DASHBOARD ---
else:
    if not model:
        st.error("Model could not be loaded. Dashboard cannot run.")
        if st.button("Go Back"):
            st.session_state.selected_match_id = None
            st.rerun()
    else:
        # --- Dashboard Header ---
        st.title(st.session_state.get('match_name', 'Live Win Probability'))
        if st.button("Change Match"):
            st.session_state.selected_match_id = None
            st.rerun()
            
        # --- Auto-Refreshing Dashboard ---
        # Create a placeholder that the loop will overwrite
        placeholder = st.empty()
        
        # Loop indefinitely to auto-refresh
        while st.session_state.selected_match_id:
            
            # Fetch the latest score data
            score_data = fetch_live_score(st.session_state.selected_match_id)
            
            if not score_data:
                with placeholder.container():
                    st.warning("Fetching live score... (Will retry in 20s)")
                time.sleep(REFRESH_INTERVAL_SECONDS)
                continue

            # Parse the data
            parsed_data = parse_live_data(score_data)
            status = parsed_data['status']
            stats = parsed_data['stats']

            # Use the placeholder to draw the dashboard
            with placeholder.container():
                
                # --- A. Display Live Stats ---
                st.header("Live Match State")
                st.subheader(stats.get("message", "Loading..."))
                
                if "batting_team" in stats:
                    st.text(f"Venue: {stats.get('venue')}")
                    
                    # Use columns for a clean layout
                    cols = st.columns(4)
                    cols[0].metric("Chasing Team", stats.get('batting_team', 'N/A'))
                    cols[1].metric("Current Score", stats.get('score_str', 'N/A'))
                    cols[2].metric("Target", stats.get('target', 'N/A'))
                    cols[3].metric("Defending Team", stats.get('bowling_team', 'N/A'))

                # --- B. Display Prediction ---
                st.divider()
                st.header("Win Probability")

                if status == "IN_PLAY":
                    # Get model input
                    model_input_df = parsed_data['model_input']
                    
                    # Get probabilities from the model
                    probabilities = model.predict_proba(model_input_df)
                    win_prob = probabilities[0][1] # Probability of Class 1 (Win)
                    loss_prob = probabilities[0][0] # Probability of Class 0 (Loss)

                    # Create data for the horizontal bar chart
                    prob_df = pd.DataFrame({
                        "Team": [stats['batting_team'], stats['bowling_team']],
                        "Win Probability": [win_prob, loss_prob]
                    }).set_index("Team")
                    
                    # Display the bar chart
                    st.bar_chart(prob_df, horizontal=True)

                    # --- C. Display Key Predictive Factors (The "Insights") ---
                    st.subheader("Key Predictive Factors")
                    cols = st.columns(4)
                    cols[0].metric("Runs Required", stats.get('runs_required', 'N/A'))
                    cols[1].metric("Balls Left", stats.get('balls_left', 'N/A'))
                    cols[2].metric("Wickets Remaining", stats.get('wickets_remaining', 'N/A'))
                    cols[3].metric("Required Run Rate", stats.get('required_run_rate', 'N/A'))
                    
                elif status == "WAITING":
                    # This fulfills your requirement for "Chasing Team yet to bat"
                    st.info(stats.get("message", "Waiting for match to enter 2nd innings."))
                
                elif status == "COMPLETE":
                    st.success(stats.get("message", "Match has ended."))
                    st.text("Dashboard will stop auto-refreshing.")
                    st.session_state.selected_match_id = None # Stop the loop
                
                elif status == "ERROR":
                    st.error(stats.get("message", "An error occurred."))
                    st.text("Dashboard will stop.")
                    st.session_state.selected_match_id = None # Stop the loop
                
                # Only show refresh timer if the loop is still active
                if st.session_state.selected_match_id:
                    st.text(f"Refreshing in {REFRESH_INTERVAL_SECONDS} seconds...")
            
            # Wait before the next loop iteration
            time.sleep(REFRESH_INTERVAL_SECONDS)
