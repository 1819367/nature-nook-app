from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime
from langchain_core.prompts import (
    PromptTemplate,
)  # import LangChain's PromptTemplate class

# app will run at: http://127.0.0.1:5000/

# Initialize logging
logging.basicConfig(filename="app.log", level=logging.INFO)
log = logging.getLogger("app")


# new function to create a PromptTemplate
# pass the user's submitted form data (form_data) as a parameter
def build_new_trip_prompt(form_data):
    prompt_template = PromptTemplate.from_template(
        "This trip is to {location} between {trip_start} and {trip_end}. This person will be traveling {traveling_with_list} and would like to  stay in {lodging_list}. They want to do the following activities: {adventure_list}.   Create a daily itinerary for this trip using this information."
    )
    # removed from prompt template:  This trip information is saved as {trip_name}.

    #  use the format method on the template to pass all the form data as arguments
    # return it by the function
    return prompt_template.format(
        location=form_data["location"],
        trip_start=form_data["trip_start"],
        trip_end=form_data["trip_end"],
        traveling_with_list=form_data["traveling_with_list"],
        lodging_list=form_data["lodging_list"],
        adventure_list=form_data["adventure_list"],
        # trip_name = form_data["trip_name"] removed from the list
    )


# Initialize the Flask application
app = Flask(__name__)


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
    # removed log of request.form after reviewing the form object
    # log the request form object
    # log.info(request.form)

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
    # removed after checking object
    # log the request form object
    # log.info(cleaned_form_data)

    # call build_new_trip_prompt and pass it the cleaned data dictionary
    prompt = build_new_trip_prompt(cleaned_form_data)
    # removed after checking 
    # log.info(prompt)

    return render_template("view-trip.html")


# Run the flask server
if __name__ == "__main__":
    app.run()
