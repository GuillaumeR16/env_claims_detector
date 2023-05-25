# Machine Learning - Group Project




# Research



-ahhahshshds
- shhshshs
- 	- hahahah



# Performance of models
For all the results that follow, these have been created and taken from the following file: XXXX.ipynb

Now that the models have been created, it is time to evaluate their performance. The performance of each model is assessed based on three key metrics: accuracy, precision, and recall. Before delving into a detailed analysis of these metrics, let's recap their significance in the context of environmental claim detection:

__Accuracy__: The accuracy metric evaluates how well the model correctly predicts whether a sentences as either environmental claims or non-environmental claims throughout the entire dataset. It measures the overall correctness of the model's predictions, indicating the proportion of correctly classified sentences (0 or 1). Therefore, a higher accuracy score signifies a more reliable model in accurately identifying environmental claims and non-environmental claims.

__Precision__: Precision is a performance metric that focuses on the accuracy of positive predictions made by the model. In the context of environmental claim detection, precision evaluates the model's ability to correctly identify sentences as environmental claims when it predicts them to be so. It measures the proportion of true positive predictions (i.e., sentences correctly classified as environmental claims) out of all the positive predictions made by the model. A higher precision score indicates that the model is making fewer false positive predictions and is more accurate in identifying genuine environmental claims.

__Recall__: Recall focuses on how effectively the model captures and detects actual environmental claims from the entire set of environmental claims present in the dataset. It measures the proportion of true positives (i.e., actual environmental claims predicted as environmental claims) relative to the total number of actual environmental claims. A higher recall score indicates that the model has a greater ability to identify and include actual environmental claims in its predictions, minimizing the number of false negatives.

In our further comprehensive analysis of different bank reports, each of the three metrics holds significant importance. Simply relying on accuracy alone can be misleading as it fails to directly capture the model's performance in detecting environmental claims. For instance, a model may excel at accurately recognizing non-environmental claims, resulting in a high accuracy score if the text to is big. However, if the text corpus only contains a few numbers of environmental claims, such model may struggle to identify and classify environmental claims accurately. This scenario raises potential risks as it can falsely portray a company as lacking transparency in its climate ambitions when, in reality, it may actively communicate them, but in a concise manner. In such cases, a high recall metric becomes relevant as it ensures the correct detection of all environmental claims in the reports. Precision is equally crucial in this context, as a low precision could attribute a company with excessive communication (assigning many incorrect "1" labels), potentially raising concerns in greenwashing assessments, even if the company did not disclose anything on their environmental ambitions.

To sum up, when we examine which models would be the most suitable for analyzing the annual reports of various banks, it is crucial to consider these metrics collectively in a comprehensive manner. This approach allows us to accurately assess the strengths and weaknesses of each model's ability to identify environmental claims.


Now let’s look at the numbers!

Disclaimer: among the different models used for predicting environmental claims, the one implemented with the GPT3 davinci model stands out as the best performer in terms of accuracy, precision, and recall, with respectively 90%, 73,61% and 82%. However, the analysis in this model and the ada model will be deferred to a later stage, as they were not constructed from scratch but rather sourced from diverse resources towards the end of our project. In the beginning we want assess and comment the performances of more “in-house” that we have built and fine-tuned. Hence, our initial focus will be directed towards the examination of the following models: 
- BOW
-  TF-IDF
-  Doc2Vec
-  Word2Vec
-  and DistilBERT. 
  
Furthermore, all of these models have been evaluated using the following classification methods: 
- Logistic Regression
- KNN
- Decision Tree 
- and Random Forest


In the end, we have a total of 20 models to analyze, each with three performance metrics. All of the models have undergone fine-tuning. You can find the code and details of the fine-tuning process for all the models in the following notebook: XXXX.ipynb.

The tables presented below display the performance scores for each model.


<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/d5b30941-81c9-4cab-b7c1-740f2d48f6c4"> <br>



First, upon analyzing the results, an intriguing observation emerges: models based on words embedding , such as Doc2Vec, Word2Vec, do not consistently outperform models that treat words individually without considering their order, such as BOW and TF-IDF. This pattern is observed across the three considered metrics. Indeed, among the four different vectorizers explored, it is consistently observed that either BOW or TF-IDF yields the best performance across accuracy, precision, and recall metrics. Notably, the combination of the TF-IDF vectorizer with the Random Forest classifier achieves the highest accuracy (not considering DistilBERT) at 84.67%. This result implies that 84.67% of the predictions made by this model, either 0 or 1, were correct. Such a performance represents a substantial improvement of 11.79% compared to the baseline accuracy of 78.67%. Without considering the remaining models (i.e., DistilBERT, Ada, and Davinci), we can already establish a notable improvement in performance compared to the baseline and naive model that predicts only 0s. However, it remains interesting to explore the possible factors contributing to the comparatively lower performance of word embedding models, namely Doc2Vec and Word2Vec, in relation to simpler models such as Bag-of-Words (BOW) and TF-IDF.

-	In comparison to word embedding methods, the BOW vectorizer operates by simply tallying the number of words present in a sentence. Through exploratory data analysis (EDA), we have observed certain words known as buzzwords that exhibit strong association with environmental claims. These buzzwords include terms like "emissions," "impact," and "sustainable." Consequently, it is plausible that a simpler approach such as BOW, which focuses solely on word count without considering word order or context, could be more effective than more intricate techniques. For example, if the test set contains sentences featuring these buzzwords, the model is more likely to predict them as environmental claims.

-	If we look specifically at TF-IDF, this model may yield better results compared to word embeddings due to its ability to highlight specific words such as specific environmental terms or jargon. Indeed, TF-IDF assigns higher weights to words that are rare across the corpus, thus emphasizing important terms that potentially indicate environmental claims. Therefore, if the testing data set contains sentences with such specific terms, the model will be better to predict these sentences as environmental claims.

-	Focusing solely on the results from the Word2vec vectorizer, we can see that the results are very low compared to the other vectorizers, and this is true across all classification methods. Even the Word2Vec-Decision Tree combination yields accuracy equivalent to the default rate, meaning that the model predicts all sentences as non-environmental claims. This outcome is reflected in the confusion matrix presented below: 


<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/4767f97b-d688-4b65-9bad-3983990e7f35">

The precision and recall metrics are calculated based on the number of True Positives predicted by the model. Since all sentences are labeled as 0, it is expected to observe both these metrics equal to 0. 

Why does this model exhibit inferior performance compared to the others? Our analysis suggests two potential explanations.

1. Word2vec, through its vectorization technique, generates a vector for each word and then calculates the average of all these vectors to position the entire sentence in space. Consequently, when a sentence contains words typically associated with environmental claims, their influence on the final placement is significantly diminished since the average of all words in the sentence is taken. As a result, the sentence's position remains anchored at an average level and fails to gravitate towards that of environmental claims.
2. The second possible reason, as discussed with Prof. Valchos, is that this method may not capture and vectorize all the words effectively. In such cases, if the identified and vectorized words in the sentence happen to be stopwords, there is no opportunity for the entire sentence to be associated with an environmental claim. <br>

- Moreover, the dataset's size plays a significant role in the effectiveness of Word embedding models like Word2Vec and Doc2Vec. It seems that these models require a substantial amount of training data to accurately represent words. If the dataset used for training the word embeddings is relatively limited, the resulting embeddings might not adequately capture the underlying semantic connections between words. In such situations, simpler models like BOW, which depend on straightforward word occurrence statistics, may yield better results. In the case of a Word2Vec model for instance, with a smaller dataset, words like 'climate' may not be adequately represented in the vector space. This may occur because the model has not been trained on a sufficient amount of data to capture the nuanced meanings and associations of such words. Therefore, when using this model on the testing dataset, some sentences containing the word “climate” might not be correctly labelled. This is why, in a subsequent phase of the report, we plan to conduct a more extensive analysis where we enhance our dataset by incorporating new environmental claims

__DistilBERT__

Subsequently, we incorporated a DistilBERT -based model. As shown in the previously presented table, this model outperforms the previous analyses overall. Specifically, the combination of DistilBERT with a logistic regression classifier exhibits an accuracy rate of 87.67%, which surpasses our previous best result for that metric. Despite trailing the TF-IDF and Random Forest combination in precision, the DistilBERT-based model excels in terms of recall metrics, achieving the highest score of 70.31%, again with its combination with logistic regression as a classifier. 

A brief exlaination of DistilBERT: 
DistilBERT, a variant of the Transformer model, offers valuable advantages over its predecessor, BERT base. According to information shared on the [Hugging Face](https://huggingface.co/docs/transformers/model_doc/distilbert) page, it reduces parameter count by about 40%, resulting in significant gains in computational efficiency. Achieving a processing speed that is 60% faster, while maintaining over 95% of BERT's performance levels, DistilBERT therefore has a balance between resource efficiency and model effectiveness. In our comparisons with other groups using more classical versions of Bert, we observed that DistilBERT can sometimes outperformed them in performance metrics. As mentioned during our discussion with Prof. Vlachos, this superiority can be attributed to its capacity to reduce overfitting using a smaller number of parameters. From what we understood, it is more plausible that models with a higher parameter count are more susceptible to overfitting the training data, resulting in inadequate performance when applied to new data.



In addition, DistilBERT stands out from the other vectorization methods we have analyzed so far due to its ability to grasp the overall context of a sentence. This means that within the same dataset, a word can be characterized differently based on the broader context of the sentence. In our analysis, this aspect proves particularly beneficial, as illustrated by the word "environment." Depending on the general context, this word can refer to the climate-related environment or the working environment, which is unrelated to environmental claims. Similarly, the verb "reduce," which emerged as a buzzword in the EDA, can have multiple interpretations. A company may reduce its ecological footprint, but it could also reduce its balance sheet debt ratio. Consequently, the capacity to capture sentence context enhances the predictive performance of our model. This increase in performance can be observed in all three metrics.




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


<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/9ffc1cc3-707f-4e11-ab17-6cfd42e1a8f7">

The first thing to notice is that for all of the selected banks, the number of environmental claims has significantly increased between 2019 and 2020, indicating a growing emphasis on environmental issues. This observed trend could potentially be attributed to various factors, such as external pressure from stakeholders urging the bank to address environmental concerns in response to their growing societal significance. Alternatively, it may stem from the bank's genuine commitment to align its climate actions. To gain further insights, let us examine the evolution of CO2 emissions.

Out of the five actors examined, only two experienced a decrease in their CO2 emissions during the analyzed period—Crédi Suisse and JP Morgan. Consequently, the remaining three players demonstrated an increase in their environmental communication while simultaneously increasing their CO2 emissions. The case of UBS is particularly noteworthy, as our model identified 43 additional instances of environmental claims in 2020, while there was an increase of 335,550 in CO2 emissions over the same period. BCV and Goldman Sachs also exhibited a similar but less significant trend, with a smaller increase in emissions.

For instance, let's consider a sentence from UBS's 2020 annual report, which our model predicts as an environmental claim: "*We’re one of only 5% of the 5,800+ companies scored  that are A-listed for environmental transparency and action to  cut emissions, mitigate climate risks and develop the low-carbon  economy.*"

However, does this imply that UBS is engaging in greenwashing? As discussed later in the analysis, it is challenging to draw a definitive conclusion based solely on this evidence. To ascertain the presence of greenwashing, a more extended analysis over a longer timeframe would be necessary. Nevertheless, the observations made regarding UBS could be considered a warning sign of potential greenwashing activities. Hence, an investor (e.g., a pension fund) could employ the developed model to identify such disparities between claims and CO2 emissions, enabling them to delve deeper into their analysis by requesting explanations from UBS regarding the reasons behind this discrepancy.


Furthermore, it is worth highlighting the contrast in the number of claims detected by our model between the chosen Swiss banks and American banks. Remarkably, our findings indicate that the American banks fall significantly short in terms of publicly disclosing their climate objectives compared to their Swiss counterparts. Despite Goldman Sachs' considerably larger balance sheet and extensive operations, our model predicts a mere two environmental claims made by the bank in 2019, while BCV, a Swiss bank, stands out with 18 claims during the same period. This disparity may be attributed, in part, to potential factors like public pressure that could lower climate-related disclosures in the US context.

In addition, from the data provided in the table, it is evident that there exists a notable difference among the banks in terms of their CO2 emissions. This discrepancy can be attributed to variations in their size and operations, resulting in significantly differing levels of CO2 emissions. This observation brings us to our second analysis, wherein we propose the utilization of a metric to standardize the measurement of CO2 emissions: CO2 intensity.


__Analysis 2__

The CO2 intensity measure, as previously mentioned, evaluates a company's CO2 emissions in relation to the revenue it generates. This measure enables the comparison of companies of varying sizes and impacts, as it is reasonable to assume that companies with higher revenue will be larger and have a bigger societal influence. For each bank, a CO2 intensity measure will be established to standardize their CO2 emissions based on their respective revenues. This normalization facilitates fair comparisons among banks operating in the same industry but differing in size. By utilizing this metric, we will rank banks according to their CO2 intensity and compare it with the number of environmental claims they have. This analysis aims to identify cases where a bank, despite having a low CO2 intensity ranking, extensively promotes its climate ambitions.


<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/3a963d94-c692-40a9-a9eb-081f493a3251">

For the reader information: in the provided table, companies with the lowest values in the "Rank - CO2 Intensity" column have the best CO2 emissions metrics. However, companies with the lowest values in the "Rank - Env. Claims" column are the ones that make the most environmental claims in their reports.

First, compared to the table in Analysis 1, where only global CO2 emissions were reported, the range of values for different banks in relation to their CO2 intensity is narrower. This allows for a better comparison between the companies as desired.

From analyzing both rankings, it becomes apparent that the correlation between the two is not perfect. Just because a company communicates extensively about its environmental ambitions does not necessarily make it the best environmental performer. Although this result was expected, it is still interesting to note. Furthermore, two notable observations can be made based on this table, concerning Goldman Sachs and UBS for the year 2020.

In the year 2020, Goldman Sachs obtained the worst ranking in terms of CO2 intensity, scoring 35.95. However, it also ranked poorly in the number of environmental claims compared to other players. This indicates a situation contrary to greenwashing, where the company has a significant negative impact on the environment without attempting to conceal it through positive ESG language in its annual report. Although this is unfavorable from an environmental impact perspective, at least the bank does not deceive its investors and other stakeholders.


In contrast, UBS is facing another challenging situation in 2020. According to its annual report, it ranks first in terms of the number of environmental claims, but lags behind its competitors in CO2 intensity. Again, this raises concerns about potential greenwashing practices by the largest Swiss bank. However, it is worth noting that UBS's CO2 intensity has significantly increased between 2019 and 2020.

When comparing Swiss banks to their US counterparts, it can be seen that US banks tend to rank lower in terms of CO2 intensity. Specifically, they occupy positions 6, 8, 9, and 10 in the ranking. This finding may suggest that Swiss banks are more advanced in their commitment to creating a more sustainable world.

Again, it is important to note that this conclusion is drawn from a limited sample and should not be solely relied upon for making greenwashing accusations. To obtain more accurate results, it is crucial to expand the study to include a larger number of banks and conduct research over a longer period. Despite this limitation, we have confidence in the effectiveness of our model in identifying potential instances of greenwashing. With an impressive recall rate of 82.81%, our model has proven its ability to detect environmental claims across various text sources. This makes it a valuable tool, which can be shared with portfolio managers, for instance, enabling them to “red-flag” companies that may be engaging in greenwashing practices. By simply inputting a URL, our model can analyze the number of claims made and facilitate necessary comparisons with environmental metrics. 


## Limitation

- **Transparency**: The C02 data provided by Trucost relies on self-reporting. Therefore, it should be noted that during the years under analysis (2019 and 2020), the sustainability reports of the banks, from which the C02 emission figures were primarily obtained, may not have undergone external audits. Consequently, there is a possibility that these figures may not precisely reflect the actual practices of these companies. 

- **Change takes time**: Implementing changes within the financial sector is a time-consuming process. Drawing conclusions about any greenwashing activities based on a two-year analysis is difficult, and to be fully transparent not doable. Transforming processes and investment decisions significantly necessitates internal changes, such as risk management and reporting. Therefore, it would be valuable to conduct a similar analysis over a longer timeframe to demonstrate the progression of environmental claims and key environmental metrics. This approach would enable a more objective analysis. Similarly, exploring a time-lagged analysis would be interesting. This entails examining the claims made by a bank in 2021 and assessing their correlation with emissions not in the same year, but in 2022.


- **The accuracy of scope 3**: The accuracy of scope 3 remains a challenge for estimating emissions from financial institutions. However, can we accurately determine the CO2 emissions impact of BCV's financing loan to a local SME involved in piano construction? he task of obtaining precise figures in this regard remains difficult. Consequently, relying solely on numerical metrics is inadequate when determining whether a company is greenwashed. Instead, it should be important to consider other factors, such as their involvement in financing new O&G projects for example, even if the investment amount is minimal. Although such projects may not significantly inflate CO2 emissions relative to the limited capital invested, they provide insights into the bank's overall policy and the sectors in which the bank want to allocate its resources.


- **Greenwashing ?** : In summary, this model proves highly effective at assessing environmental claims but cannot directly determine greenwashing. To address this, the user has to conduct a comprehensive comparison across various environmental metrics such as C02 emissions, CO2 emissions, and others. Although this presents challenges, it would be intriguing to develop a model specifically designed to detect greenwashing directly. One approach could involve researchers concentrating on companies known to engage in greenwashing and labeling specific “greenwashing claims” in their reports. Indeed, it is possible that certain words regularly appear in the annual reports of companies involved in greenwashing practices. However, creating such a model would also carry legal risks, as accusing a company of greenwashing is a serious matter, as demonstrated by the DWS case last year. Hence, this model would need human verification given the fine line between greenwashing and genuine climate ambitions, which makes the detection task even more challenging.

## Conclusion



