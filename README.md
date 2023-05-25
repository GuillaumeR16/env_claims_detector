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

	
- In the next phase, we will extend our analysis to the industry level, taking into account the varying sizes of banks to address differences in emissions. To achieve this, we will introduce a measure called CO2 intensity for each bank, which normalizes the individual CO2 emissions by the company's revenues. This normalization helps mitigate the impact of size variations and enables meaningful comparisons across different companies within the same industry. By leveraging this metric, we will establish a ranking of banks based on their CO2 intensity and compare it with the ranking of their number of environmental claims. This comparison aims to identify instances where a bank, despite being ranked poorly in terms of CO2 intensity, extensively communicates its climate ambitions. Such a discrepancy could serve as an additional indication of potential greenwashing practices. . This analysis will be conducted over a two-year period as well.


