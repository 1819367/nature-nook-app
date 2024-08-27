from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime
from langchain_core.prompts.few_shot import FewShotPromptTemplate 
from langchain_core.prompts import (
    PromptTemplate,
) 
from langchain_openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser 

# app will run at: http://127.0.0.1:5000/

# Initialize the OpenAI language model
# set the token limit to -1 to avoid output parcer exception errors for large files
# there is a risk doing this, NO limit on the token usage
llm = OpenAI(
    max_tokens = -1
)

# create an instance of the JSONOutputParserclass
parser = JsonOutputParser()

# Initialize logging
logging.basicConfig(filename="app.log", level=logging.INFO)
log = logging.getLogger("app")


# Initialize the Flask application
app = Flask(__name__)

# updated to return a prompt template not a formatted prompt
def build_new_trip_prompt_template():
    # examples of prompts and responses
    examples = [
        {
        "prompt":
"""
This trip is to Zion National Park between 2024-8-22 and 2024-8-25. This person will be traveling solo and would like to  stay in campsites. They want to do the following activities: hiking.  Create a daily itinerary for this trip using this information.
""", 
        "response":
"""
{{"trip_name":"My amazing trip to Zion National Park 2024","location":"Zion National Park","trip_start":"2024-08-22","trip_end":"2024-08-25","num_days":"4","traveling_with":"solo","lodging":"campsites","adventure":"hiking","itinerary":[{{"day":"1","date":"2024-08-22","morning":"Arrive at Zion National Park","afternoon":"Set up campsite at Watchman Campground","evening":"Explore the campground and have dinner by the campfire"}},{{"day":"2","date":"2024-08-23","morning":"Guided tour of Zion National Park","afternoon":"Picnic lunch the Grotto Picnic Area","evening":"Relax at the campsite and stargaze later"}},{{"day":"3","date":"2024-08-24","morning":"Hike to Emerald Pools","afternoon":"Enjoy lunch at Zion Lodge","evening":"Dinner at the campsite"}},{{"day":"4","date":"2024-08-25","morning":"Pack up campsite","afternoon":"Enjoy biking through the canyon","evening":"Head home out of the park"}}]}}

"""
        },
        {
        "prompt":
"""
This trip is to Yellowstone National Park between 2024-7-15 and 2024-7-18. This person will be traveling with family and would like to stay in a lodge. They want to do the following activities: wildlife viewing, geyser watching, and hiking. Create a daily itinerary for this trip using this information.
""",
        "response":
"""
{{"trip_name":"My amazing trip to Yellowstone National Park 2024","location":"Yellowstone National Park","trip_start":"2024-08-15","trip_end":"2024-08-18","num_days":"4","traveling_with":"solo, with the kids","lodging":"lodge","adventure":"wildlife viewing, geyser watching, hiking","itinerary":[{{"day":"1","date":"2024-08-15","morning":"Arrive at Yellowstone National Park, check into Old Faithful Inn","afternoon":"Watch Old Faithful geyser eruption, explore nearby geyser basin","evening":"Welcome dinner at Old Faithful Inn Dining Room"}},{{"day":"2","date":"2024-08-16","morning":"Guided wildlife tour in Lamar Valley","afternoon":"Picnic lunch, hike to Trout Lake","evening":"Relax at the lodge, attend ranger program"}},{{"day":"3","date":"2024-08-17","morning":"Visit Grand Prismatic Spring and Midway Geyser Basin","afternoon":"Hike to Fairy Falls","evening":"Family-style dinner at Lake Yellowstone Hotel"}},{{"day":"4","date":"2024-08-18","morning":"Early breakfast, final geyser watching at Old Faithful","afternoon":"Short hike around West Thumb Geyser Basin","evening":"Depart Yellowstone National Park"}}]}}
"""
        }, 
        {
        "prompt":
"""
This trip is to Acadia National Park between 2024-9-10 and 2024-9-13. This person will be traveling with friends and would like to stay in a vacation rental. They want to do the following activities: biking, kayaking, and stargazing. Create a daily itinerary for this trip using this information.
""",
        "response":
"""
{{"trip_name":"My amazing trip to Acadia National Park 2024","location":"Acadia National Park","trip_start":"2024-09-10","trip_end":"2024-09-13","num_days":"4","traveling_with":"solo, with friends","lodging":"vacation rental","adventure":"biking, kayaking, stargazing","itinerary":[{{"day":"1","date":"2024-09-10","morning":"Arrive in Acadia National Park, check into vacation rental","afternoon":"Bike ride on carriage roads around Eagle Lake","evening":"Group dinner at vacation rental, plan stargazing session"}},{{"day":"2","date":"2024-09-11","morning":"Kayaking tour of Frenchman Bay","afternoon":"Picnic lunch at Sand Beach, explore Thunder Hole","evening":"Stargazing at Cadillac Mountain summit"}},{{"day":"3","date":"2024-09-12","morning":"Bike the Park Loop Road","afternoon":"Hike Beehive Trail","evening":"Sunset kayaking in Somes Sound, followed by lobster dinner"}},{{"day":"4","date":"2024-09-13","morning":"Final bike ride on Witch Hole Pond carriage road","afternoon":"Visit Bass Harbor Head Lighthouse","evening":"Pack up and depart Acadia National Park"}}]}}
"""
        },
        {
        "prompt":
"""
This trip is to Grand Canyon National Park between 2024-6-5 and 2024-6-8. This person will be traveling with a partner and would like to stay in a hotel. They want to do the following activities: hiking, mule riding, and river rafting. Create a daily itinerary for this trip using this information.
""",
        "response":
"""
{{"trip_name":"My amazing trip to Grand Canyon National Park 2024","location":"Grand Canyon National Park","trip_start":"2024-06-05","trip_end":"2024-06-08","num_days":"4","traveling_with":"solo, with partner","lodging":"hotel","adventure":"hiking, mule riding, river rafting","itinerary":[{{"day":"1","date":"2024-06-05","morning":"Arrive at Grand Canyon National Park, check into El Tovar Hotel","afternoon":"Rim Trail hike for panoramic views","evening":"Dinner at Arizona Room, watch sunset at Hopi Point"}},{{"day":"2","date":"2024-06-06","morning":"Mule ride along the rim to Abyss Overlook","afternoon":"Visit Grand Canyon Village and Geology Museum","evening":"Attend ranger-led talk on Grand Canyon geology"}},{{"day":"3","date":"2024-06-07","morning":"Early breakfast at hotel","afternoon":"Full-day whitewater rafting trip on the Colorado River","evening":"Relaxing dinner at El Tovar Dining Room"}},{{"day":"4","date":"2024-06-08","morning":"Sunrise hike on South Kaibab Trail to Ooh-Aah Point","afternoon":"Final exploration of visitor center and gift shops","evening":"Depart Grand Canyon National Park"}}]}}
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

    # update for first chain
    # create the few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples = examples, #these are the prompt examples
        example_prompt = example_prompt, #this is the example prompt template
        suffix = "This trip is to {location} between {trip_start} and {trip_end}.  This person will travel {traveling_with} and wants to stay in {lodging}.  They want to {adventure}.  Create a daily itinerary for this trip using this information. You are a backend data processor of our app's programmatic workflow.  Output the itinerary as only JSON with no text before or after the JSON.", 
        input_variables = ["location", "trip_start", "trip_end", "traveling_with", "lodging", "adventure"], 
    )

    # to check few_shot_prompt.format in the log
    # log.info(few_shot_prompt.format)

    # new line added, return few_shot_prompt
    return few_shot_prompt

  
    # commented out to update for first chain
    #few_shot_prompt formatted & returned
    # return few_shot_prompt.format(input = "This trip is to " + form_data["location"] + " between " + form_data["trip_start"] + " and " + form_data["trip_end"] + ".  This person will travel " + form_data["traveling_with_list"] + " and wants to stay in " + form_data["lodging"] + ".  They want to " + form_data["adventure"] + ".  Create a daily itinerary for this trip using this information. You are a backend data processor of our app's programmatic workflow.  Output the itinerary as only JSON with no text before or after the JSON."
    # ) 

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
# updated to use a chain
def view_trip():

    # create a comma-separated list for the multi-select fields
    traveling_with_list = ", ".join(request.form.getlist("traveling-with"))
    lodging_list = ", ".join(request.form.getlist("lodging"))
    adventure_list = ", ".join(request.form.getlist("adventure"))
    
    # removed the argument and updated the function name
    prompt = build_new_trip_prompt_template()
    
    # build the chain
    chain = prompt | llm | parser

    # updated to invoke the chain and pass the user's submitted from data
    output = chain.invoke({
        "location": request.form["location-search"],
        "trip_start": request.form["trip-start"],
        "trip_end": request.form["trip-end"],
        "traveling_with": traveling_with_list,
        "lodging": lodging_list, #changed key name from lodging_list
        "adventure": adventure_list, #changed key name from adventure_list
        "trip_name": request.form["trip-name"] # added back into dictionary
    })

   # log output
    log.info(output)

    return render_template("view-trip.html", output = output)

    # code removed since llm is invoked in the chain
    # send the "cleaned" trip prompt to OpenAI (llm model)
    # response = llm.invoke(prompt)

    # to see the response from OpenAI
    # commented out to parse
    # log.info(response) 

    # code removed since parsing is now part of the chain
    # parse the model's response using parser, set to new variable called output
    # output = parser.parse(response)

# Run the flask server
if __name__ == "__main__":
    app.run()
