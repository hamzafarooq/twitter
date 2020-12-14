import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import numpy as np
import pandas as pd
import scipy.stats.distributions as dist
import plotly.figure_factory as ff
import plotly.express as px
from statsmodels.stats.proportion import proportions_ztest
import researchpy as rp
import scipy.stats as stats
from statsmodels.stats import weightstats as stests

#################################################################################
## This app is a great way to visualize data and combine it with story telling,##
## hence most of the comments are in form of the text used in the app. To run  ##
## this app type: $streamlit run Q2_app.py #######################################
#################################################################################


df = pd.read_csv("experiment.csv")
#df.head()

st.title("Q2: Experiment Outcome [A/B Testing]")
st.markdown(
"""
We are tasked with analyzing the outcome of an experiment recently conducted on a group of users to test the
hypothesis on the effectiveness of a new feature. The feature will help in reducing campaign overspending
""")
st.markdown("________________________________________")

st.header("Exploratory Data Analysis")


st.text('We will start with looking at the data..')

st.dataframe(df)

st.markdown(
"""
From a cursory look, the data has two groups, treatment and control and there are three company sizes, small, meidum and elevation_range.
Each is followed by the campaign budget and spend, respectively """)

a= pd.crosstab(df.treatment,df.company_size,margins=True)
a

st.text('A quick overview of the data summary')
st.text(df.describe())

st.markdown(
"""
75th percentile of both campaign spend and budget is in 236 and 252 dollars, however,
their maxium values are way in the multi-millions!! For the purpose of visualization, we can remove some of the higher percentile values """)
a=np.percentile(df['campaign_spend'].values,95)
b=np.percentile(df['campaign_budget'].values,95)
st.text("95th percentile for campaign spend:" )
a
st.text("95th percentile for campaign budget:" )
b


st.text("Checking for na")
st.text(df.isna().sum())


st.text("""A good visualization on the spread of data for campaign spend and budget
** data reduced to 75th percentile for better visualization **""")
sliders = {
    "campaign_spend": st.slider(
        "Campaign Spend", min_value=0.0, max_value=400.0, value=(0.0, 400.0), step=10.0
    ),
    "campaign_budget": st.slider(
        "Campaign Budget", min_value=0.0, max_value=400.0, value=(0.0, 400.0), step=10.0
    ),
}

#filter = np.full(n_rows, True)  # Initialize filter as only True

for feature_name, slider in sliders.items():
    # Here we update the filter to take into account the value of each slider
    filter = (
        (df['campaign_spend'] >= slider[0])
        & (df['campaign_budget'] <= slider[1])
    )

#st.write(df[filter])



c = alt.Chart(df[filter]).mark_circle().encode(
x='campaign_spend', y='campaign_budget', size='campaign_spend', color='treatment',
tooltip=['campaign_spend', 'campaign_budget', 'treatment','company_size']).interactive()

st.altair_chart(c, use_container_width=True)

st.text('Adding a histogram - it updates with the filter above')

# Create distplot with custom bin_size
c=alt.Chart(df[filter]).mark_boxplot().encode(
    x='company_size:O',
    y='campaign_spend:Q',
    color=alt.Color('company_size')
).interactive()
st.altair_chart(c, use_container_width=True)

st.markdown(
"""
The data is consistent with the inutuition that the company size follows the budget and spend-
although there are quite a few instances when the spend is quite high for smaller companies """)


st.text("""Lets take a look at some of the mean campaign spend and budget
** Greater than 95 Percentile data removed, to reflect a better picture**""")
bars = alt.Chart(df[df['campaign_spend']<a]).mark_bar(cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    x=alt.X('mean(campaign_spend):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    color=alt.Color('treatment')
).interactive()

text = alt.Chart(df[df['campaign_budget']<a]).mark_text(dx=-50, dy=3, color='white').encode(
    x=alt.X('mean(campaign_spend):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    detail='treatment:N',
    text=alt.Text('mean(campaign_spend):Q', format='.1f')
)



st.altair_chart(bars + text, use_container_width=True)






bars = alt.Chart(df[df['campaign_spend']<b]).mark_bar(cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    x=alt.X('mean(campaign_budget):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    color=alt.Color('treatment')
)


text = alt.Chart(df[df['campaign_spend']<b]).mark_text(dx=-50, dy=3, color='white').encode(
    x=alt.X('mean(campaign_budget):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    detail='treatment:N',
    text=alt.Text('mean(campaign_budget):Q', format='.1f')
)


st.altair_chart(bars + text, use_container_width=True)


###adding new dataframe

st.markdown("________________________________________")


st.header("Adding new variables for over spend")

st.markdown("""**Assumption: Since we are considering if overspend happened and not by how much,
we are going to include all the data at this time ** """)

st.text("""We can introduce two new variables,
percentage overspend and assign 1 to it if crossed 1% of the budget""")

df = pd.read_csv("experiment.csv")
df['percentage'] = (df['campaign_spend'] - df['campaign_budget'])*100/df['campaign_budget']
df.loc[df['percentage'] <= 1, 'over_spend'] = 0
df.loc[df['percentage'] > 1, 'over_spend'] =  1

st.text(df.head(10))

st.markdown("""**Note: It is important to remember that over spend is 1 when the spend is
1% greater than the budget** """)

st.text("The table below shows the proportion of over spend across groups and overall")

a= pd.crosstab(df.treatment,df.over_spend).apply(lambda r:r/r.sum(),axis=0)
a
a= pd.crosstab([df.company_size,df.over_spend],df.treatment).apply(lambda r:r/r.sum(),axis=0)
a



st.text('The barplots will give us a better breakdown of total number overall and for each group')



bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    x=alt.X('sum(over_spend):Q', stack='zero'),
    y=alt.Y('treatment:N'),
    color=alt.Color('treatment')
)

text = alt.Chart(df).mark_text(dx=-25, dy=3, color='white').encode(
    x=alt.X('sum(over_spend):Q', stack='zero'),
    y=alt.Y('treatment:N'),
    detail='treatment:N',
    text=alt.Text('sum(over_spend):Q', format='.0f')
)


st.altair_chart(bars + text, use_container_width=True)



bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    x=alt.X('sum(over_spend):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    color=alt.Color('treatment')
)

text = alt.Chart(df).mark_text(dx=-19, dy=3, color='white').encode(
    x=alt.X('sum(over_spend):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    detail='treatment:N',
    text=alt.Text('sum(over_spend):Q', format='.0f')
)


st.altair_chart(bars + text, use_container_width=True)




st.markdown("""**From this data it is clear that there are definitely more over spend occurences
in the control group overall and across all the companies
At an overall level, the number of over spend campaigns are 5716 vs 5180 in control vs treatment,respectively** """)


st.text("We can also look at the mean percentage for each group in overspending")
bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    x=alt.X('mean(percentage):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    color=alt.Color('treatment')
)

text = alt.Chart(df).mark_text(dx=-19, dy=3, color='white').encode(
    x=alt.X('mean(percentage):Q', stack='zero'),
    y=alt.Y('company_size:N'),
    detail='treatment:N',
    text=alt.Text('mean(percentage):Q', format='.0f')
)

st.altair_chart(bars + text, use_container_width=True)
st.text("""...definitely a reduction in percentage of overspend (except for middle size)
but it is too early to tell""")
st.markdown("________________________________________")
st.header("Hypothesis Testing for the new feature")

st.markdown("""We will begin by looking at an overall effectiveness and will then delve in to
specific groups, a.k.a company size  """)

st.markdown("""For this purpose, we will first consider the overall propotion in the control
and test for the campaigns which had an over spend in the budget""")


st.markdown("""We will be using the z-test for this since we meet the conditions of it:

1. Your sample size is greater than 30.

2. Data points should be independent from each other. In other words, one data point isn’t related
or doesn’t affect another data point.

3. Your data should be normally distributed. However, for large sample sizes (over 30) this
doesn’t always matter.

4. Your data should be randomly selected from a population, where each item has an equal
chance of being selected.

5. Sample sizes should be equal or close if at all possible.""")
st.markdown("""This will be a one-tail z-test since we are only concerned with control if it is
larger or not than test""")

st.markdown("Setting the null and alternative Hypothesis--")
st.text ("""H[null] = There is no difference between the propotions of the two groups, i.e.
any change in proportions is due to chance whereas the alternative hypothesis is that
control group has a higher propotion than test""")



st.latex(r''' H_{0} : proportion_{control} = proportion_{test}''')

st.latex(r'''H_{a} : proportion_{control} > proportion_{test}''')

st.text('Setting a significance level to 0.05')
significance = 0.05
# our samples - 82% are good in one, and ~79% are good in the other
# note - the samples do not need to be the same size
st.text("""overspend_control, sample_size_control = (5716, 7733)
overspend_test, sample_size_test = (5180, 7741)""")
sample_success_a, sample_size_a = (5716, 7733)
sample_success_b, sample_size_b = (5180, 7741)
# check our sample against Ho for Ha != Ho
successes = np.array([sample_success_a, sample_success_b])
samples = np.array([sample_size_a, sample_size_b])
# note, no need for a Ho value here - it's derived from the other parameters
stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='larger')
# report
st.text('Running the z-test...')
st.text('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
if p_value > significance:
   st.write ("Fail to reject the null hypothesis - we have nothing else to say")
else:
   st.text ("Reject the null hypothesis - suggest the alternative hypothesis is true")

st.markdown("""**The p-value suggests that the we can reject the null hypothesis, we can
infer that the change in proportions of over spending in test is not due to chance and
the new feature is definitely working** """)

st.markdown("""Similar to our above methodology, we can use z-test for each company size
to highlight the effectiveness of the new feature  """)





st.markdown("""     ________________________________________



**_Setting up z-test for small companies_**""")

st.markdown("Setting the null and alternative Hypothesis--")
st.text ("""H[null] = There is no difference between the propotions of the two groups
of small companies, i.e.
any change in proportions is due to chance whereas the alternative hypothesis is that
control group has a higher propotion than test""")


st.latex(r''' H_{0} : proportion_{control} = proportion_{test}''')

st.latex(r'''H_{a} : proportion_{control} > proportion_{test}''')

st.text('Setting a significance level to 0.05')
significance = 0.05
# our samples - 82% are good in one, and ~79% are good in the other
# note - the samples do not need to be the same size

sample_success_a, sample_size_a = (df[(df['company_size']=="small")& (df['over_spend']==1) & (df['treatment']==0)].shape[0],
 df[(df['company_size']=="small") & (df['treatment']==0)].shape[0])
sample_success_b, sample_size_b = (df[(df['company_size']=="small")& (df['over_spend']==1) & (df['treatment']==1)].shape[0],
 df[(df['company_size']=="small") & (df['treatment']==1)].shape[0])
# check our sample against Ho for Ha != Ho
successes = np.array([sample_success_a, sample_success_b])
samples = np.array([sample_size_a, sample_size_b])
# note, no need for a Ho value here - it's derived from the other parameters
stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='larger')
# report
st.text('Running the z-test...')
st.text('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
if p_value > significance:
   st.write ("Fail to reject the null hypothesis - we have nothing else to say")
else:
   st.text ("Reject the null hypothesis - suggest the alternative hypothesis is true")

st.markdown("""**The p-value suggests that the we can reject the null hypothesis, we can
infer that the change in proportions of over spending in test is not due to chance and
the new feature is definitely working** """)





st.markdown("""     ________________________________________



**_Setting up z-test for medium size companies_**""")

st.markdown("Setting the null and alternative Hypothesis--")
st.text ("""H[null] = There is no difference between the propotions of the two groups
of medium size companies, i.e.
any change in proportions is due to chance whereas the alternative hypothesis is that
control group has a higher propotion than test""")


st.latex(r''' H_{0} : proportion_{control} = proportion_{test}''')

st.latex(r'''H_{a} : proportion_{control} > proportion_{test}''')

st.text('Setting a significance level to 0.05')
significance = 0.05
# our samples - 82% are good in one, and ~79% are good in the other
# note - the samples do not need to be the same size

sample_success_a, sample_size_a = (df[(df['company_size']=="medium")& (df['over_spend']==1) & (df['treatment']==0)].shape[0],
 df[(df['company_size']=="medium") & (df['treatment']==0)].shape[0])
sample_success_b, sample_size_b = (df[(df['company_size']=="medium")& (df['over_spend']==1) & (df['treatment']==1)].shape[0],
 df[(df['company_size']=="medium") & (df['treatment']==1)].shape[0])
# check our sample against Ho for Ha != Ho
successes = np.array([sample_success_a, sample_success_b])
samples = np.array([sample_size_a, sample_size_b])
# note, no need for a Ho value here - it's derived from the other parameters
stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='larger')
# report
st.text('Running the z-test...')
st.text('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
if p_value > significance:
   st.write ("Fail to reject the null hypothesis - we have nothing else to say")
else:
   st.text ("Reject the null hypothesis - suggest the alternative hypothesis is true")

st.markdown("""**The p-value suggests that the we cannot reject the null hypothesis, we can
infer that any change in proportions of over spending in the test is due to chance and
the new feature is not working for medium size companies** """)





st.markdown("""     ________________________________________



**_Setting up z-test for large size companies_**""")

st.markdown("Setting the null and alternative Hypothesis--")
st.text ("""H[null] = There is no difference between the propotions of the two groups
of large companies, i.e.
any change in proportions is due to chance whereas the alternative hypothesis is that
control group has a higher propotion than test""")


st.latex(r''' H_{0} : proportion_{control} = proportion_{test}''')

st.latex(r'''H_{a} : proportion_{control} > proportion_{test}''')

st.text('Setting a significance level to 0.05')
significance = 0.05
# our samples - 82% are good in one, and ~79% are good in the other
# note - the samples do not need to be the same size

sample_success_a, sample_size_a = (df[(df['company_size']=="large")& (df['over_spend']==1) & (df['treatment']==0)].shape[0],
 df[(df['company_size']=="large") & (df['treatment']==0)].shape[0])
sample_success_b, sample_size_b = (df[(df['company_size']=="large")& (df['over_spend']==1) & (df['treatment']==1)].shape[0],
 df[(df['company_size']=="large") & (df['treatment']==1)].shape[0])
# check our sample against Ho for Ha != Ho
successes = np.array([sample_success_a, sample_success_b])
samples = np.array([sample_size_a, sample_size_b])
# note, no need for a Ho value here - it's derived from the other parameters
stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='larger')
# report
st.text('Running the z-test...')
st.text('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
if p_value > significance:
   st.write ("Fail to reject the null hypothesis - we have nothing else to say")
else:
   st.text ("Reject the null hypothesis - suggest the alternative hypothesis is true")

st.markdown("""**The p-value suggests that the we can reject the null hypothesis, we can
infer that the change in proportions of over spending in test is not due to chance and
the new feature is definitely working** """)

st.markdown("________________________________________")
st.header("Checking for any changes in budget in the Experiment ")

st.text('Here we will try to determine if there is intentional lowering of Budgets')



st.markdown("""For this purpose, we will analyze the campaign budgets set by teams in overspend_control
and test group""")


st.markdown("""We will again be using the z-test for this since we meet the conditions.""")
st.markdown("""This will be a one-tail z-test since we are only concerned with if budget in contol is
larger than test""")

st.markdown("Setting the null and alternative Hypothesis--")
st.text ("""H[null] = There is no difference between the mean of budgets of the two groups, i.e.
any change in budget is due to chance, whereas the alternative hypothesis is that
control group has a higher budget than test [intentional lowering for the latter]""")



st.latex(r''' H_{0} :meanofbudget_{control} = meanofbudget_{test}''')

st.latex(r'''H_{a} : meanofbudget_{control} > meanofbudget_{test}''')

st.text('Setting a significance level to 0.05')
significance = 0.05



# report
st.text('Running the z-test...')

ztest ,pval1 = stests.ztest(df[df['treatment']==0].campaign_budget.values, x2=df[df['treatment']==1].campaign_budget.values,
value=0,alternative='larger')
st.text("ztest and p-value:")
st.write(ztest,float(pval1))
if pval1<0.05:
    st.write("reject null hypothesis")
else:
    st.write("accept null hypothesis")

st.markdown("""**The p-value suggests that the we cannot reject the null hypothesis, we can
refer that any change in setting up campaign budge in test is  due to chance and intentional
lowering of budget has been done** """)


st.markdown("________________________________________")
