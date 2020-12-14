import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import numpy as np
import pandas as pd
import scipy.stats.distributions as dist
from statsmodels.stats.proportion import proportions_ztest
import researchpy as rp
import scipy.stats as stats
from statsmodels.stats import weightstats as stests
from vega_datasets import data

#################################################################################
## This app is a great way to visualize data and combine it with story telling,##
## hence most of the comments are in form of the text used in the app. To run  ##
## this app type: $streamlit run Q3_app.py #######################################
#################################################################################

#Data Sources: https://www.epa.gov/climate-indicators/climate-change-indicators-high-and-low-temperatures
#https://www.epa.gov/climate-indicators/climate-change-indicators-high-and-low-temperatures


st.title("Shift in Weather Pattern across US")
st.markdown(
"""
We are tasked with identifying regions in US with the largest shifts in the weather pattern
in recent times
""")



st.markdown("""The Earth's climate is undoubtely changing. Temperatures are on the rise, similarly, snow and
rainfall patterns are shifting, and much more extreme climate events – like heavy rainstorms
and record high temperatures – have been witnessed on a much regular basis.

My efforts here are to identify the effects of these changes across regions in USA

In order to put together my analysis, I will be analyzing various dataset and will focus  on:

1. Overall change in temperatures over time in United States
2. Hot and Cold Weather changes in specific areas
3. States with highest effect of hurricanes
4. Wild Fires in California State""")


st.markdown("________________________________________")

st.header("Historical Climate Change: 1950 to 2013")


st.markdown("""In this spirit, I used data from various sources to help me see the differences in climate
over time.

My first dataset is taken from a kaggle competition:
https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data

The specific dataset that I will utilize this data to begin my initial analysis is:
Global Average Land Temperature by State""")
df= pd.read_csv("GlobalLandTemperaturesByState.csv")

st.write(df.head(15))


st.markdown("""Since this data is from across the world, I will only limit it to United States, this
data is in Celcicus""")

df_usa =df[df['Country']=='United States']
df_usa['year'] = pd.DatetimeIndex(df_usa['dt']).year
df_usa['month'] = pd.DatetimeIndex(df_usa['dt']).month
df_usa['day'] = pd.DatetimeIndex(df_usa['dt']).dayofweek
df_usa =df_usa[df_usa['year']>=1950]

st.markdown("In the interest of more recent data, I will limit my data to 1950 and beyond")

st.write(df_usa.head(10))

st.markdown("""Now that we have our data, we can utilize it to make various plots, one of them can be
the average temperature each year,followed by a moving average(I calculated a 10 year moving average)""")

g=df_usa.groupby(['year'])['AverageTemperature'].mean()
g = pd.DataFrame(g)
g=g.reset_index()
g['moving_avg']= g['AverageTemperature'].shift(1).rolling(window=10).mean()
source = g.melt('year', var_name='category', value_name='y')

line = alt.Chart(source).mark_line().encode(
    x='year:O',

    y='y:Q',
     color='category:N',


    tooltip=['category:N','y:Q', 'year:O']
).properties(
    width=700,
    height=400
).interactive()

st.altair_chart(line, use_container_width=True)

st.markdown("""We can  immediately see an upward trend of temperature, even compared to the moving Average
and from 1950 average the temperature is almost 3 Degrees higher!! """)

st.markdown("""Since this data is at State level, we can drill down to  the change in temperatures
across each state""")

g=df_usa.groupby(['year','State'])['AverageTemperature'].mean()
g = pd.DataFrame(g)
g=g.reset_index()
source = g

b=alt.Chart(source).mark_boxplot().encode(
    x='State:O',
    y='AverageTemperature:Q',
    color = 'State'
).properties(
    width=700,
    height=400
).interactive()

st.altair_chart(b, use_container_width=True)

st.markdown("""An interesting observation here is the outliers observed in states which have been
cold primarily, for e.g. Minnesota, Idaho, Alaska and North Dakota""")

def std(x):
    return np.std(x)



g=df_usa.groupby(['State'])['AverageTemperature'].std()
g = pd.DataFrame(g)
g=g.reset_index()
g.columns = ['State','std']

st.write(g)

st.markdown("""The table above and the chart below both explain the change in temperature changes,
confirming about North Dakota and Alaska, interestingly, Hawaii has the least fluctuations in temperature""")

b=alt.Chart(g).mark_bar().encode(
    x="State:N",
    y="std:Q",
    color= "State:N",
    tooltip=[ 'State','std'],
).properties(height=400,width=600).interactive()

st.altair_chart(b, use_container_width=True)




st.markdown("________________________________________")

st.header("Changes in Highs and Lows acorss US: 1948-2015")


df_warm= pd.read_csv("weather_change.csv")
df_cold = pd.read_csv("weather_cold.csv")

st.markdown("""To further deepen our analysis, I will introduce two new datasets taken from the webiste:
https://www.epa.gov/climate-indicators/climate-change-indicators-high-and-low-temperatures

We can analyse pattern for hot and cold days here.

The trends show unusually hot temperatures at individual weather stations that have operated
consistently since 1948. In this case, the term “unusually hot” refers to a daily maximum
temperature that is hotter than the 95th percentile temperature during the 1948–2015 period.
Thus, the maximum temperature on a particular day at a particular station would be
considered “unusually hot” if it falls within the warmest 5 percent of measurements at
that station during the 1948–2015 period. The map shows changes in the total number of
days per year that were hotter than the 95th percentile. Red color show
where these unusually hot days are becoming more common. A lighter shade indicates
show where unusually hot days are becoming less common.

Using the slider we can find out places which have now began to get much warmer days. Most of those
are located in  Florida and California state. Interestingly, places in Minnesota are also getting
warmer""")

sliders = {
    "change": st.slider(
        "change", min_value=-40.0, max_value=20.0, value=(-40.0, 55.0), step=10.0
    ),

}

#filter = np.full(n_rows, True)  # Initialize filter as only True

for feature_name, slider in sliders.items():
    # Here we update the filter to take into account the value of each slider
    filter = (
        (df_warm['change'] >= slider[0]) &
        (df_warm['change'] <= slider[1])

    )



airports = df_warm[filter]
states = alt.topo_feature(data.us_10m.url, feature='states')

# US states background.mark_geoshape().encode(
background = alt.Chart(states).mark_geoshape(
   fill='lightgray',
    stroke='white'
).properties(
    width=700,
    height=400
).project('albersUsa')

# airport positions on background
points = alt.Chart(airports).mark_circle(
    size=70
).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    tooltip=[ 'state','change','latitude','longitude'],
    color=alt.Color('change:Q', scale=alt.Scale(scheme='yelloworangered'))
)



st.altair_chart(background + points, use_container_width=True)

st.text("""Map: Change in Unusually Hot Temperatures in the Contiguous 48 States, 1948-2015",,,
Source: EPA's Climate Change Indicators in the United States:
www.epa.gov/climate-indicators,,,
"Data source: NOAA, 2016"

Units: Change in number of days hotter than 95th percentile""")




st.markdown(""" The map below points to trends in unusually cold temperatures at individual
weather stations that have operated consistently since 1948. In this case, the term
“unusually cold” refers to a daily minimum temperature that is colder than the 5th percentile
temperature during the 1948–2015 period. Thus, the minimum temperature on a particular day
at a particular station would be considered “unusually cold” if it falls within the
coldest 5 percent of measurements at that station during the 1948–2015 period.
The map shows changes in the total number of days per year that were colder than the 5th percentile.
Light green shade show where these unusually cold days are becoming more common.
The darker shades show where unusually cold days are becoming less common.""")

sliders_2= {
    "change_cold": st.slider(
        "change_cold", min_value=-40.0, max_value=20.0, value=(-40.0, 20.0), step=5.0
    ),

}

#filter = np.full(n_rows, True)  # Initialize filter as only True

for feature_name, slider in sliders_2.items():
    # Here we update the filter to take into account the value of each slider
    filter = (
        (df_cold['change_cold'] >= slider[0]) &
        (df_cold['change_cold'] <= slider[1])

    )

airports = df_cold[filter]
states = alt.topo_feature(data.us_10m.url, feature='states')

# US states background.mark_geoshape().encode(
background = alt.Chart(states).mark_geoshape(
   fill='lightgray',
    stroke='white'
).properties(
    width=700,
    height=400
).project('albersUsa')

# airport positions on background
points = alt.Chart(airports).mark_point(
    size=40,
    color='red'
).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    tooltip=[ 'state','change_cold','latitude','longitude'],
    color=alt.Color('change_cold:Q', scale=alt.Scale(scheme='viridis'))


)

st.altair_chart(background + points, use_container_width=True)


st.text("""Map: Change in Unusually Cold Temperatures in the Contiguous 48 States, 1948-2015"
Source: EPA's Climate Change Indicators in the United States:
www.epa.gov/climate-indicators
"Data source: NOAA, 2016"
Units: Change in number of days colder than 5th percentile""")







st.markdown("________________________________________")

st.header("U.S. Mainland Hurricane Strikes by State: 1851-2004")

st.markdown("""What is the average number of hurricanes per year?
What year(s) had the most and least hurricanes? What hurricane had the
longest life? On what date did the earliest and latest hurricane occur? What was the most intense
Atlantic hurricane? What was the largest number of hurricanes in existence on the same day? When
was the last time a major hurricane or any hurricane hit a given community direct?

These are some of the overarching question one may have about hurricanes.

In the data below, taken from National Hurrican Center and Central Pacific Hurricane center,
we can see how hurricanes have effected us.

Florida and Texas have experienced the most!""")




states = alt.topo_feature(data.us_10m.url, 'states')
source = data.population_engineers_hurricanes.url
variable_list = ['hurricanes']

a=alt.Chart(states).mark_geoshape().encode(
    alt.Color(alt.repeat('row'), type='quantitative')
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(source, 'id', variable_list)
).properties(
    width=500,
    height=300
).project(
    type='albersUsa'
).repeat(
    row=variable_list
).resolve_scale(
    color='independent'
)
st.altair_chart(a, use_container_width=True)




st.markdown("________________________________________")




st.header("Wild Fires across California in the 7 years")

st.markdown("""Wild Fires have become  a common phenomenon for the past  few years, notably in the CA state.
I used data available at: https://www.kaggle.com/ananthu017/california-wildfire-incidents-20132020
to generate a visualization on some of the recent fires since 2013 and the  acres of land that was burned due to that""")
df = pd.read_csv("California_Fire_Incidents.csv")



st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": 37.857,
        "longitude": -120.086,
        "zoom": 5 ,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=df,
            get_position=["long", "lat"],
            radius=7000,

            getElevationWeight="AcresBurned",
            elevation_scale=1000,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            auto_highlight=True,
        ),

        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["long", "lat"],
            radius=50,
            getweight="AcresBurned",
            get_radius=100,
            cell_size = "AcresBurned",      # Radius is given in meters
            get_fill_color=[180, 0, 200, 140],
auto_highlight=True,
            pickable=True,
            extruded=True,
        ),
    ],
     tooltip={"html": "<b>Acres Burned:</b> {elevationValue} <br> <b>Lat/Long:</b> {position} ", "style": {"color": "white"}},
))





st.markdown("________________________________________")




st.header("Summary")

st.markdown("""In a nutshell, quite a few regions have had a shift in weather, places which were primarliy warmer
have become more warm and some of the cold ones have gotten warmer too. Not a single State has gotten colder
than it was before. Most effected regions from the data are:

1. California
2. Texas
3. Florida
4. Alaska""")
