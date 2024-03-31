# save this as app.py
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Needed for code to waterfall plots to generate properly

application = Flask(__name__)

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- Code For Web Pages --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

# The about me, resume, and projects pages are basic funtions that simplt render those HTML templates as they have no dynamic code.

@application.route('/')
@application.route('/about')
def about():

    return render_template("about.html")

@application.route('/resume')
def resume():

    return render_template("resume.html")

@application.route('/projects')
def projects():

    return render_template("projects.html")

# The Hospitalization Model consists of various web pages

# 1. The first page has a static explanation of the model and a form to select a state

@application.route('/hospitalizationPredictor')
def hospitalizationPredictor():
     # open file
    file = open("provider_data.pkl", "rb")
    # load state list
    provider_data = joblib.load(file)
    states = np.sort(provider_data['State'].unique())
    return render_template("hospitalizationPredictor.html", states = states)

# 2. After selecting the state, another page opens prompting the user to select a city from the list of cities filtered by state

@application.route('/city', methods=['GET', 'POST'])
def city():
    if request.method == "POST":
         # open file
        file = open("provider_data.pkl", "rb")
        # load state list
        provider_data = joblib.load(file)
        # get form data
        state = request.form.get('state')
        # filter for cities in the state
        cities = np.sort(provider_data.loc[provider_data['State'] == state, 'City/Town'].unique())
        return render_template("city.html", cities = cities)

 # 3. After selecting the city, another page opens prompting the user to select a provider from the list of providers within the selected city

@application.route('/provider', methods=['GET', 'POST'])
def proider():
    if request.method == "POST":
         # open file
        file = open("provider_data.pkl", "rb")
        # load state list
        provider_data = joblib.load(file)
        # get form data
        city = request.form.get('city')
        # filter for providers in the City
        providers = np.sort(provider_data.loc[provider_data['City/Town'] == city, 'Provider Name'].unique())
        return render_template("provider.html", providers = providers)
    
 # 4. As sometimes there are duplicate nursing homes with the same provider name, the user is then presented a list of CCN numbers for that provider.
 #    Often there is only one CCN. The CCN uniquely identifies one nursing home.


@application.route('/ccn', methods=['GET', 'POST'])
def ccn():
    if request.method == "POST":
         # open file
        file = open("provider_data.pkl", "rb")
        # load provider list
        provider_data = joblib.load(file)
            # get form data
        provider = request.form.get('provider')
            # filter for providers in the City
        ccns = np.sort(provider_data.loc[provider_data['Provider Name'] == provider, 'CMS Certification Number (CCN)'].unique())
        return render_template("ccn.html", ccns = ccns)

# 5. Results Page for Selected Nursing Home

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
         # open file
        file = open("provider_data.pkl", "rb")
        # load provider list
        provider_data = joblib.load(file)
        # get form data
        ccn = request.form.get('ccn')
        # Obtain the index corresponding to the SHAP values in the model (provider data has matching index to the shap values)
        shap_index = provider_data[provider_data['CMS Certification Number (CCN)'] == ccn].index[0]
        # Create transposed data frame of provider information
        data_all  = provider_data[provider_data['CMS Certification Number (CCN)'] == ccn].set_index('Provider Name').T
        # Exclude rate from DataFrame to create an HTML talbe displaying address information
        data = data_all.loc[['CMS Certification Number (CCN)', 'Provider Address', 'City/Town', 'State', 'ZIP Code'], :]
        # Seperate the predicted rate to display seperately
        actual_rate = round(data_all.loc['Adjusted Hospitalization Rate',:].values[0], 2)
        # open shap values file
        file = open("shap_values.pkl", "rb")
        # load shap values
        shap_values = joblib.load(file)
        # Create waterfall plot with custom size and labels
        plt.figure(figsize=(25,25))
        shap.plots.waterfall(shap_values[shap_index], show = False)
        plt.xlabel('Adjusted Hospitalization Rate per 1,000 Resident Days', fontsize = 'x-large', weight = 'bold', labelpad = 15)
        # Save plot
        plt.savefig('static/waterfall.png', bbox_inches='tight')
        return render_template("predict.html", tables=[data.to_html(classes='data', header = False)], titles=data.columns.values, actual_rate = actual_rate)


# Run on Correct Port
if __name__ == '__main__':
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    #application.debug = True
    #application.run(host="localhost", port=5000, debug=True) # Line for hosting locally
    application.run()
