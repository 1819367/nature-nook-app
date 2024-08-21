from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime
from langchain_core.prompts.few_shot import FewShotPromptTemplate 
from langchain_core.prompts import (
    PromptTemplate,
) 
from langchain_openai import OpenAI

# app will run at: http://127.0.0.1:5000/

# Initialize the OpenAI language model
llm = OpenAI()

# Initialize logging
logging.basicConfig(filename="app.log", level=logging.INFO)
log = logging.getLogger("app")


# Initialize the Flask application
app = Flask(__name__)

# new function to create a PromptTemplate
# pass the user's submitted form data (form_data) as a parameter
def build_new_trip_prompt(form_data):
    # examples of prompts and responses
    examples = [
        {
        "prompt":
"""
This trip is to Zion National Park between 2024-8-22 and 2024-8-25. This person will be traveling solo and would like to  stay in campsites. They want to do the following activities: hiking.  Create a daily itinerary for this trip using this information.
""", 
        "response":
"""
Itinerary:
Day 1: August 22, 2024 (Friday)
Morning: Arrive at Zion National Park
Afternoon: Set up campsite at Watchman Campground
Evening: Explore the campground and have dinner by the campfire

Day 2: August 23, 2024 (Saturday)
Morning: Guided tour of Zion National Park
Afternoon: Picnic lunch the Grotto Picnic Area
Evening: Relax at the campsite and stargaze later

Day 3: August 24, 2024 (Sunday)
Morning: Hike to Emerald Pools
Afternoon: Enjoy lunch at Zion Lodge
Evening: Dinner at the campsite

Day 4: August 25, 2024 (Monday)
Morning:  Pack up campsite
Afternoon: Enjoy biking through the canyon
Evening:  Head home out of the park
"""
        },
        {
        "prompt":
"""
This trip is to Yellowstone National Park between 2024-7-15 and 2024-7-18. This person will be traveling with family and would like to stay in a lodge. They want to do the following activities: wildlife viewing, geyser watching, and hiking. Create a daily itinerary for this trip using this information.
""",
        "response":
"""
Itinerary:
Day 1: July 15, 2024 (Monday)
Morning: Arrive at Yellowstone National Park, check into Old Faithful Inn
Afternoon: Watch Old Faithful geyser eruption, explore nearby geyser basin
Evening: Welcome dinner at Old Faithful Inn Dining Room

Day 2: July 16, 2024 (Tuesday)
Morning: Guided wildlife tour in Lamar Valley
Afternoon: Picnic lunch, hike to Trout Lake
Evening: Relax at the lodge, attend ranger program

Day 3: July 17, 2024 (Wednesday)
Morning: Visit Grand Prismatic Spring and Midway Geyser Basin
Afternoon: Hike to Fairy Falls
Evening: Family-style dinner at Lake Yellowstone Hotel

Day 4: July 18, 2024 (Thursday)
Morning: Early breakfast, final geyser watching at Old Faithful
Afternoon: Short hike around West Thumb Geyser Basin
Evening: Depart Yellowstone National Park
"""
        }, 
        {
        "prompt":
"""
This trip is to Acadia National Park between 2024-9-10 and 2024-9-13. This person will be traveling with friends and would like to stay in a vacation rental. They want to do the following activities: biking, kayaking, and stargazing. Create a daily itinerary for this trip using this information.
""",
        "response":
"""
Itinerary:
Day 1: September 10, 2024 (Tuesday)
Morning: Arrive in Acadia National Park, check into vacation rental
Afternoon: Bike ride on carriage roads around Eagle Lake
Evening: Group dinner at vacation rental, plan stargazing session

Day 2: September 11, 2024 (Wednesday)
Morning: Kayaking tour of Frenchman Bay
Afternoon: Picnic lunch at Sand Beach, explore Thunder Hole
Evening: Stargazing at Cadillac Mountain summit

Day 3: September 12, 2024 (Thursday)
Morning: Bike the Park Loop Road
Afternoon: Hike Beehive Trail
Evening: Sunset kayaking in Somes Sound, followed by lobster dinner

Day 4: September 13, 2024 (Friday)
Morning: Final bike ride on Witch Hole Pond carriage road
Afternoon: Visit Bass Harbor Head Lighthouse
Evening: Pack up and depart Acadia National Park
"""
        },
        {
        "prompt":
"""
This trip is to Grand Canyon National Park between 2024-6-5 and 2024-6-8. This person will be traveling with a partner and would like to stay in a hotel. They want to do the following activities: hiking, mule riding, and river rafting. Create a daily itinerary for this trip using this information.
""",
        "response":
"""
Itinerary:
Day 1: June 5, 2024 (Wednesday)
Morning: Arrive at Grand Canyon National Park, check into El Tovar Hotel
Afternoon: Rim Trail hike for panoramic views
Evening: Dinner at Arizona Room, watch sunset at Hopi Point

Day 2: June 6, 2024 (Thursday)
Morning: Mule ride along the rim to Abyss Overlook
Afternoon: Visit Grand Canyon Village and Geology Museum
Evening: Attend ranger-led talk on Grand Canyon geology

Day 3: June 7, 2024 (Friday)
Morning: Early breakfast at hotel
All Day: Full-day whitewater rafting trip on the Colorado River
Evening: Relaxing dinner at El Tovar Dining Room

Day 4: June 8, 2024 (Saturday)
Morning: Sunrise hike on South Kaibab Trail to Ooh-Aah Point
Afternoon: Final exploration of visitor center and gift shops
Evening: Depart Grand Canyon National Park
"""
    },
    ]
    # create a prompt template that defines the structure of the response
    example_prompt = PromptTemplate.from_template(
        template = 
"""
{prompt}\n{response}
"""
    )
    # log info to check prompt template.  removed after checking
    # log.info(example_prompt.format(**examples[1]))

    # create the few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples = examples, #these are the prompt examples
        example_prompt = example_prompt, #this is the example prompt template
        suffix = "{input}", #sets the real prmpt with the user's form data, it's appended at the end of the example prompts
        input_variables = ["input"], #passes in the real prompt with teh user's form data
    )

    # to check few_shot_prompt.format in the log
    # log.info(few_shot_prompt.format)

    #few_shot_prompt formatted & returned
    return few_shot_prompt.format(input = "This trip is to " + form_data["location"] + " between " + form_data["trip_start"] + " and " + form_data["trip_end"] + ".  This person will travel " + form_data["traveling_with_list"] + " and wants to stay in " + form_data["lodging_list"] + ".  They want to do the following activities: " + form_data["adventure_list"] + ".  Create a daily itinerary for this trip using this information."
    ) 

# Define the route for the home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Define the route for the plan trip page
@app.route("/plan_trip", methods=["GET"])
def plan_trip():
    return render_template("plan-trip.html")


# Define the route for view trip page with the generated trip itinerary
@app.route("/view_trip", methods=["POST"])
def view_trip():

    # create a comma-separated list for the multi-select fields
    traveling_with_list = ", ".join(request.form.getlist("traveling-with"))
    lodging_list = ", ".join(request.form.getlist("lodging"))
    adventure_list = ", ".join(request.form.getlist("adventure"))
    # create a dictionary with the cleaned form data
    cleaned_form_data = {
        "location": request.form["location-search"],
        "trip_start": request.form["trip-start"],
        "trip_end": request.form["trip-end"],
        "traveling_with_list": traveling_with_list,
        "lodging_list": lodging_list,
        "adventure_list": adventure_list,
        #  "trip_name": request.form["trip-name"] removed from the dictionary
    }

    # call build_new_trip_prompt and pass it the cleaned data dictionary
    prompt = build_new_trip_prompt(cleaned_form_data)

    # send the "cleaned" trip prompt to OpenAI (llm model)
    response = llm.invoke(prompt)

    # to see the response from OpenAI
    log.info(response) 

    return render_template("view-trip.html")


# Run the flask server
if __name__ == "__main__":
    app.run()
