# Machine Learning - Group Project

shhshs





# Rearach 




Exemple de screenshoot

<img width="240" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/6fba0e37-b8cc-49d0-8cfb-7bd43373a791">





## Sub-section





# Analysis on annual reports:

In this section of the report, our objective is to conduct an analysis of greenwashing detection using annual reports from diverse banks and financial institutions. To ensure the utmost accuracy in our analysis, we will leverage the most advanced model developed thus far, namely gpt3 with davinci. As demonstrated in the previous chapter, this model has exhibited superior performance metrics, making it the most suitable choice for our investigation.

However, it is important to clarify that our text classifier is designed to determine whether a sentence qualifies as an environmental claim or not, rather than specifically identifying instances of greenwashing. Thus, the primary significance of our model lies in its ability to quantify the occurrence of "environmental claims" within annual reports. However, drawing conclusions about greenwashing practices based solely on this information is not feasible. To gain meaningful insights, it is necessary to compare this data with relevant benchmarks or references such as CO2 metrics or historical trends. Before delving into further details, we would like to provide a brief explanation for our choice to focus on banks in our analysis.


For years, banks have been central to the challenges posed by greenwashing. Despite showcasing impressive ESG web sections, both Swiss and international banks persist in financing fossil fuels at levels far exceeding scientific recommendations. Over the course of six years following the Paris Agreement, [Banking on Climate Chaos](https://www.bankingonclimatechaos.org) identified that the 60 largest banks worldwide have channeled a staggering USD $4.6 trillion into fossil fuel investments, with a striking USD $742 billion invested in 2021 alone [1]. However, research conducted by the [International Energy Agency](https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf) reveals that the expansion of new oil and gas fields, as well as the continued development of coal mines, squarely lie outside the climate budget of 1.5°C required to achieve carbon neutrality by 2050 [2]. This disconnect between the actions of the financial sector and the urgent climate crisis is evident. Therefore, it becomes intriguing to explore the degree to which various financial institutions communicate their climate commitments within their annual reports and how well these align with their actual actions, as reflected by environmental metrics. Any divergence between the two could potentially indicate the presence of greenwashing practices. In order to explore this further, we will employ our text classifier model to conduct two distinct analyses on a selected group of banks:


- In the initial analysis, we will examine each selected bank individually and compare the changes in their number of environmental claims over two consecutive years with the corresponding changes in their CO2 emissions during the same period. This comparative assessment aims to identify any alignment or disparity between the bank's increased communication on climate change through environmental claims in their annual reports and the actual trajectory of their CO2 emissions. If we observe a significant increase in environmental claims alongside a simultaneous rise in CO2 emissions, it could raise concerns regarding potential greenwashing practices.

	
- In the next phase, we will extend our analysis to the industry level, taking into account the varying sizes of banks to address differences in emissions. To achieve this, we will introduce a measure called CO2 intensity for each bank, which normalizes the individual CO2 emissions by the company's revenues. This normalization helps mitigate the impact of size variations and enables meaningful comparisons across different companies within the same industry. By leveraging this metric, we will establish a ranking of banks based on their CO2 intensity and compare it with the ranking of their number of environmental claims. This comparison aims to identify instances where a bank, despite being ranked poorly in terms of CO2 intensity, extensively communicates its climate ambitions. Such a discrepancy could serve as an additional indication of potential greenwashing practices. This analysis will be conducted over a two-year period as well.

The selected banks for analysis include:

- UBS
- Credit Suisse
- Banque Cantonale Vaudoise
- J.P. Morgan
- Goldman Sachs


Our focus initially lies on the two largest Swiss banks (soon one), UBS and Credit Suisse. These banks have been at the forefront of discussions surrounding brown financing in recent years. Additionally, we have chosen Banque Cantonale Vaudoise (BCV) for its local significance within the canton of Vaud. Lastly, our analysis extends beyond Swiss borders to include prominent American banks. For the analysis, we will be examining their respective annual reports for the years 2019 and 2020.

The data related to the climate performance of each bank was obtained from the Trucost database, which was graciously provided by Dr. Boris Thurm. These data served as the basis for calculating the following metrics for each bank:

- Carbon-Scope 1  (tonnes CO2e)
- Carbon-Scope 2  (tonnes CO2e)
- Carbon-Scope 3 (tonnes CO2e)
- Carbon Intensity-Scope 1 (tonnes CO2e/USD mn)
- Carbon Intensity-Scope 2 (tonnes CO2e/USD mn)
- Carbon Intensity-Scope 3 (tonnes CO2e/USD mn)
 

Finally, during the extraction of text from the various annual reports, we focused on retaining the specific pages and chapters where environmental claims could potentially be found. Not all sections of the annual reports were pertinent to our analysis (e.g., financial statements, audit reports, etc.). Hence, each report was exanimated to identify the relevant portions and our predictive model was exclusively applied to these specific segments of the report.



## Result
For all the results that follow these have been created and taken from the following file: Fina_Openai_CO2.ipynb

__Analysis 1__

As mentioned previously, in the preliminary analysis, we will assess each chosen bank separately by comparing the variations in their environmental claims over two consecutive years with the corresponding changes in CO2 emissions. This evaluation aims to determine the consistency or disparity between the bank's emphasis on climate change in their annual reports through environmental claims and the actual trend of their CO2 emissions. The table below presents the findings of this analysis.

![image](https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/9ffc1cc3-707f-4e11-ab17-6cfd42e1a8f7)

The first thing to notice is that for all of the selected banks, the number of environmental claims has significantly increased between 2019 and 2020, indicating a growing emphasis on environmental issues. This observed trend could potentially be attributed to various factors, such as external pressure from stakeholders urging the bank to address environmental concerns in response to their growing societal significance. Alternatively, it may stem from the bank's genuine commitment to align its climate actions. To gain further insights, let us examine the evolution of CO2 emissions.

Out of the five actors examined, only two experienced a decrease in their CO2 emissions during the analyzed period—Crédi Suisse and JP Morgan. Consequently, the remaining three players demonstrated an increase in their environmental communication while simultaneously increasing their CO2 emissions. The case of UBS is particularly noteworthy, as our model identified 43 additional instances of environmental claims in 2020, while there was an increase of 335,550 in CO2 emissions over the same period. BCV and Goldman Sachs also exhibited a similar but less significant trend, with a smaller increase in emissions.

For instance, let's consider a sentence from UBS's 2020 annual report, which our model predicts as an environmental claim: "*We’re one of only 5% of the 5,800+ companies scored  that are A-listed for environmental transparency and action to  cut emissions, mitigate climate risks and develop the low-carbon  economy.*"

However, does this imply that UBS is engaging in greenwashing? As discussed later in the analysis, it is challenging to draw a definitive conclusion based solely on this evidence. To ascertain the presence of greenwashing, a more extended analysis over a longer timeframe would be necessary. Nevertheless, the observations made regarding UBS could be considered a warning sign of potential greenwashing activities. Hence, an investor (e.g., a pension fund) could employ the developed model to identify such disparities between claims and CO2 emissions, enabling them to delve deeper into their analysis by requesting explanations from UBS regarding the reasons behind this discrepancy.


Furthermore, it is worth highlighting the contrast in the number of claims detected by our model between the chosen Swiss banks and American banks. Remarkably, our findings indicate that the American banks fall significantly short in terms of publicly disclosing their climate objectives compared to their Swiss counterparts. Despite Goldman Sachs' considerably larger balance sheet and extensive operations, our model predicts a mere two environmental claims made by the bank in 2019, while BCV, a Swiss bank, stands out with 18 claims during the same period. This disparity may be attributed, in part, to potential factors like public pressure that could lower climate-related disclosures in the US context.

In addition, from the data provided in the table, it is evident that there exists a notable difference among the banks in terms of their CO2 emissions. This discrepancy can be attributed to variations in their size and operations, resulting in significantly differing levels of CO2 emissions. This observation brings us to our second analysis, wherein we propose the utilization of a metric to standardize the measurement of CO2 emissions: CO2 intensity.


__Analysis 2__

