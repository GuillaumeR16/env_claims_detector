# Final Project Machine Learning (MGT-502) - Towards Sustainability or Greenwashing

__Contributors:__
- Guillaume Rico
- Charlotte Ahrens
- Noé Lopez

# Task attribution
As part of the present analysis, each team member has made significant and consistent contributions across various aspects of the project.

Charlotte played a proactive role in the development of the research question and conducted an extensive literature review. Additionally, she was responsible for the entire exploratory data analysis (EDA) phase, which provided valuable insights for our research direction and endpoint analysis.

Noé took the lead in building the foundation for the text classifier models, which served as a starting point for the team. With his strong coding skills, Guillaume then enhanced and improved the existing models and incorporated GPT-3 into our project. Guillaume also undertook the task of code cleanup, implementing various functions to streamline our notebooks and optimize the overall process, making it more efficient and comprehensible.

When it came to analyzing the annual reports of the bank, Noé formulated the overall strategy for this task and implemented the best model discovered thus far. Noé also constructed the tables used for comparisons between companies within the industry. Guillaume also contributed to this aspect by simplifying and optimizing the code.

In general, tasks were allocated based on each person's expertise and preferences, ensuring an equal distribution of responsibilities. However, it's important to note that all the conclusions drawn from our research were based on group discussions and input from all members.


# Introduction
With increasing global warming, companies’ pressure to adapt to new social norms enhances. Most countries signed the Paris Agreement, back in 2015, which includes National Determined Contributions (NDCs), which outline the countries climate actions, mitigation targets and adaptation measures. To fulfill those targets, the industrial sector has a crucial role to play within each country. Therefore, public, governmental, and legal demand plays a key role for companies to comply with the set targets. This has pushed companies to adapt and apply sustainable practices within their firm. While some companies have been first mover within adoption of sustainable and ethical business practices following the Environmental, Social, and Governance (ESG) guidelines, others are trying to take the shortcut and avoid any measurements towards these needs.


# Research question
This is where our research comes into play. Sustainability is a rising star in terms of customer needs, and business reputation. As a company, being associated with sustainable practices increases the company’s competitive advantage and therefore attracts an increasing number of customers, especially the younger generation. In industries as far from energy and food to manufacturing and banking, the demand for green practices has increased tremendously. This incentivizes companies to build a “green” reputation around their business, which often comes with lies, false statements, and unachievable goal settings. To understand the extend of actual implementation of green practices versus false environmental claims our research is exploring the following question: “To what extent are companies actually implementing sustainable practices, and to what extent do sustainability reports reflect genuine efforts, as opposed to being mere instances of greenwashing?” In more detail, we will be training a model which differentiates environmental claims from other business statements of companies, and then comparing the number of claims to carbon emissions of this company. How do they develop within each company? How do they compare between companies? These questions will be answered within our research. Since we are focusing on sustainable practices, it goes hand in hand with looking at the 2030 Agenda for Sustainable Development, containing 17 Goals (SDGs), developed by the United Nations an adopted by all United Nations Member States in 2015. The goals provide a shared blueprint for peace and prosperity for people and the planet, now and into the future. This research will focus on the global banking industry and therefore touches three main [SDGs](https://sdgs.un.org/goals):

- Goal 9: Industry, Innovation, and Infrastructure (Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation)
- Goal 12: Responsible consumption and production: This goal indicates the need to ensure sustainable consumption and production patterns. The banking and finance industry influences consumption and production patterns through responsible lending and investment practices that promote sustainability, circular economy principles, and environmentally friendly initiatives.
- Goal 13: Climate Action: Banks and financial institutions have a critical role in funding climate-related projects. They support renewable energy investments and integrate climate risk management into their operations in order to take urgent action to combat climate change and its impacts.
- Goal 17: Partnership for the goals: The banking and finance sector can foster partnerships and collaborations with various stakeholders, including governments, communities, NGOs, and many more. It has the power to address sustainable development challenges collectively and strive for global partnership for sustainable development.

# Literature review
Before diving right into our research, we looked at the current research environment that focuses on environmental claims, possible greenwashing attempts through companies and methods to uncover those. One key research paper within this field is [“An Integrated framework to Assess Greenwashing.”](https://www.mdpi.com/2071-1050/14/8/4431) With 12 citations, and a publishing date of 2022, it is the most updated and thorough framework to analyze the quality and truthfulness of environmental claims, up to date. The methodology that it uses and are qualitative analyses of the academic literature to explore varieties of greenwashing, as well finding typologies of greenwashing, and detecting and analyzing greenwashing claims. Additionally, the paper looked at non-academic and practitioner sources an analyzed their greenwashing clauses. This included monitoring organizations and analyzing them based on Greenpeace given greenwashing detection criteria. Based on that, dirty companies, ad buster, political spin, and non-law compliant environmental claims are categorized as greenwashing related company statements. The paper concludes that the term “greenwashing” is far from clearly defined as the research in this field in conceptualized by several influencers including the field of business, media, communication, environmental studies, law, social sciences, and many others. The main results of the paper boil down to the definition of greenwashing which is as follows. “Greenwashing is an umbrella term for a variety of misleading communications and practices that intentionally or not, induce false positive perceptions of an organization’s environmental performance. It can be conducted by companies, governments, politicians, research organizations, international organizations, banks, and NGOs too and it can range from slight exaggeration to full fabrication, thus there are different shades of greenwashing.” To analyze environmental claims, the paper concludes that the definition requires continuous assessment. Moreover, is offers an integrated framework that helps to find a claim that is potentially a greenwash and check it against the list of indicator questions in the framework. The paper concludes that this framework should help to avoid greenwash practices by the mentioned organizations.


Another recent development and research within this field has come from a collaborative research team based at University College Dublin. A team of impact-driven sustainable finance experts from academia and industry have been developing [GreenWatch](https://greenwatch.ai), a tool for the financial sector to assess and monitor the authenticity of green claims. GreenWatch came to live 2022 and was a seed-phase team in Science Foundation Ireland’s AI for Societal Good Challenge and co-founded by the Department of Foreign Affairs and Trade. GreenWatch was created with the mission to empower investors with data to assess the credibility of green claims made by companies to accelerate the transition to a climate-neutral economy. To do so it determines greenwashing statements of the corporate sector through 3 steps. First, it uses a trained algorithm to detect “green statements”. These statements are rated and ranked based on their audacity. The next step is the verification process by a sustainable finance professional within the team to confirm or reject the ratings given by the AI model. Finally, the rating of the claim is compared to greenhouse gas performance of companies, which are publicly reported by each firm. Important to know about the model that decides over the categorization of green claims, is the way that it has been set up. Looking at it from a decision tree perspective, the first node of the tool decides the firm’s stance on climate change. Following this, the next decision node is based on the sufficient GHG emission performance of the company, with a benchmark being at 7% of GHG reduction year on year until. The tool is currently employed in the market, used by various investors to follow their impact investment strategies, and achieve their goals. According to Georgiana Ifrim, computer science professor at the University College Dublin says that GreenWatch showed that a large number of companies make absolute claims that just do not stand up when contrasted with the United Nations Environment Programs that set the benchmark of 7.6% emission reduction every year until 2050. GreenWatch focuses just as our research on the financial sector working towards the realization of four SDGs, including affordable and clean energy, decent work and economy growth, industry and innovation, and just as our focus, on positive climate actions as a whole.


Lastly, we take a look at the research that gives the foundation to our research methods. In 2022, the ETH Zürich’s center for Law & Economics department, in 2022 published the paper [“A Dataset for Detecting Real-World Environmental Claims”](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/568978/CLE_WP_2022_07.pdf?sequence=1), including a dataset of annotated environmental claims. These claims were made by listed companies, mostly in the financial domain, which implies the research is focusing on the financial sector. A claim was labelled to be an environmental claim when it “referred to the practice of suggesting or otherwise creating the impression (in the context of a commercial communication, marketing or advertising) that a product or a service is environmentally friendly (i.e., it has a positive impact on the environment) or is less damaging to the environment than competing goods or services.” In addition to providing the annotated dataset, a case study on corporate earning calls was made. In order detect greenwashing practices, the first few models were trained on the test set including a total of 2400 samples. Training the model and testing it on a sample size of 300 claims the model RoBERTa(large) with a F1 score of 90.7, slightly outperforms ClimateBert (86.7), RoBERTa(base)(85.7), as well as DistilBERT (84.7). After training the model on the annotated dataset, it was applied in a field study where 12 million sentences from corporate earning calls of 3361 unique companies between 2012 and 2020 were analyzed and categorized in environmental / non-environmental claims. Using the ClimateBert model, the development showed the exponential increase in claims since the Paris Agreement in 2015.

Overall, the literature review has shown that many academic papers tackle the issue of greenwashing. To understand the dimension of false claims, these papers mainly focus on the term definition of greenwashing. As there is no clear statement of what practices and wording includes, this term must be consistently adapted due to changing social and technical norms. On the other hand, industry example has shown that there is an increasing demand in order detect and use those claims for the greater good of society when banking decisions investments are made.

# EDA
For all the results that follow, these have been created and taken from the following file: EDA.ipynb

In this section, we present the key takeaways obtained from our EDA.

1. As can be observed in the image below, the training data set is quite unbalanced in terms of labelled=0 and labelled=1 data. In other words, there is a significant disparity between the number of environmental and non-environmental claims. The same observations can be made for the testing data set. In additon, the default rate in the testing data set is 78.67%. 

<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/f2ae0a7b-e461-4fa4-82f9-d60a5d5e6879"> <br>

2. Through a targeted analysis of environmental claims and the use of a selective tokenizer to remove insignificant words, we identified recurring buzzwords associated with environmental claims. The top three “buzzwords”, namely "energy," "environmental," and "reduce," are the most present words in these sentences. As a result, it is plausible that sentences containing these buzzwords in our testing set have a higher likelihood of being classified as environmental claims by our further predictive models.

<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/0ba6d398-6507-4b43-b90d-33f319387fa3"> <br>

3. When looking at informations about years within environmental claims, the main observation are: 

- 2015, 2019 and 2020 show a relative peak in environmental claims. 2015 is likely to be due to thh year of the Paris Agreement, meaning several claims where either made in that time or refer to that year within their statement. In terms of the rise in the year 2019 and 2020, it is likely to be mentioned more often due to the time period when the dataset was fed with claims.
- Years that lay in the future are likely to come from claims that state information about future targets. Such claims may outline plans, strategies, or commitments that organizations or individuals aim to achieve in the future (e.g,, 2030 emissions target)

<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/b25c01e0-6bc9-419c-82de-22e4cb624e11"> <br>


4. When selecting the claims that are coming from a specific company, the following observations can be made: 
- The company is not determining if a claim is labels as environmental or not as we find both labels within claims from the same company.
- We only have the company's name for a minority of the claims (125/2400). 







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


<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/d579bbd1-f7eb-4445-bfd4-e1e2d0c55768"> <br>


First, upon analyzing the results, an intriguing observation emerges: models based on words embedding , such as Doc2Vec, Word2Vec, do not outperform models that treat words individually without considering their order, such as BOW and TF-IDF. This pattern is observed across the three considered metrics. Indeed, among the four different vectorizers explored, it is most of the time observed that either BOW or TF-IDF yields the best performance across accuracy, precision, and recall metrics. Notably, the combination of the TF-IDF vectorizer with the Decision Tree classifier achieves the highest accuracy (not considering DistilBERT) at 84.00%. This result implies that 84.00% of the predictions made by this model, either 0 or 1, were correct. Such a performance represents a substantial improvement of 5.33% compared to the baseline accuracy of 78.67%. Without considering the remaining models (i.e., DistilBERT, Ada, and Davinci), we can already establish a notable improvement in performance compared to the baseline and naive model that predicts only 0s. However, it remains interesting to explore the possible factors contributing to the comparatively lower performance of word embedding models, namely Doc2Vec and Word2Vec, in relation to simpler models such as Bag-of-Words (BOW) and TF-IDF.

-	In comparison to word embedding methods, the BOW vectorizer operates by simply tallying the number of words present in a sentence. Through exploratory data analysis (EDA), we have observed certain words known as buzzwords that exhibit strong association with environmental claims. These buzzwords include terms like "emissions," "impact," and "sustainable." Consequently, it is plausible that a simpler approach such as BOW, which focuses solely on word count without considering word order or context, could be more effective than more intricate techniques. For example, if the test set contains sentences featuring these buzzwords, the model is more likely to predict them as environmental claims.

-	If we look specifically at TF-IDF, this model may yield better results compared to word embeddings due to its ability to highlight specific words such as specific environmental terms or jargon. Indeed, TF-IDF assigns higher weights to words that are rare across the corpus, thus emphasizing important terms that potentially indicate environmental claims. Therefore, if the testing data set contains sentences with such specific terms, the model will be better to predict these sentences as environmental claims.

-	Focusing solely on the results from the Word2vec vectorizer, we can see that the results are very low compared to the other vectorizers, and this is true across all classification methods. Even the Word2Vec-Decision Tree combination yields accuracy equivalent to the default rate, meaning that the model predicts all sentences as non-environmental claims. This outcome is reflected in the confusion matrix presented below: 


<img width="900" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/30b8b58d-5840-4a5f-b857-8aa8fadb75a4">

The precision and recall metrics are calculated based on the number of True Positives predicted by the model. Since all sentences are labeled as 0, it is expected to observe both these metrics equal to 0. 

Why does this model exhibit inferior performance compared to the others? Our analysis suggests two potential explanations.

1. Word2vec, through its vectorization technique, generates a vector for each word and then calculates the average of all these vectors to position the entire sentence in space. Consequently, when a sentence contains words typically associated with environmental claims, their influence on the final placement is significantly diminished since the average of all words in the sentence is taken. As a result, the sentence's position remains anchored at an average level and fails to gravitate towards that of environmental claims. For instance, let's consider the environmental claim example discussed during the EDA: "The goal to reduce relative emissions by 20% by 2020 was achieved in 2013." In the case of Word2Vec, word vectors are generated for each word within the sentence. These word vectors are then averaged to obtain a sentence vector. However, this averaging process will lead to the word "reduce," identified as a buzzword, being averaged with unrelated words like "20%" or "2020." Consequently, the resulting sentence vector's placement will be significantly influenced by these unrelated words, hindering the accurate prediction of the sentence as green.
2. The second possible reason, as discussed with Prof. Valchos, is that this method may not capture and vectorize all the words effectively. In such cases, if the identified and vectorized words in the sentence happen to be stopwords, there is no opportunity for the entire sentence to be associated with an environmental claim. <br>

- Moreover, the dataset's size plays a significant role in the effectiveness of Word embedding models like Word2Vec and Doc2Vec. It seems that these models require a substantial amount of training data to accurately represent words. If the dataset used for training the word embeddings is relatively limited, the resulting embeddings might not adequately capture the underlying semantic connections between words. In such situations, simpler models like BOW, which depend on straightforward word occurrence statistics, may yield better results. In the case of a Word2Vec model for instance, with a smaller dataset, words like 'climate' may not be adequately represented in the vector space. This may occur because the model has not been trained on a sufficient amount of data to capture the nuanced meanings and associations of such words. Therefore, when using this model on the testing dataset, some sentences containing the word “climate” might not be correctly labelled. This is why, in a subsequent phase of the report, we plan to conduct a more extensive analysis where we enhance our dataset by incorporating new environmental claims

__DistilBERT__

Subsequently, we incorporated a DistilBERT -based model. As shown in the previously presented table, this model outperforms the previous analyses overall. Specifically, the combination of DistilBERT with a logistic regression classifier exhibits an accuracy rate of 87.67%, which surpasses our previous best result for that metric.In addition, the DistilBERT-based model excels in terms of recall metrics, achieving the highest score of 70.31%, again with its combination with logistic regression as a classifier. In terms of precision, its combination with logistic regression finishes tied with the Word2Vec and logistic regression combination at 71.43%. This makes it the best model so far

A brief exlaination of DistilBERT: 
DistilBERT, a variant of the Transformer model, offers valuable advantages over its predecessor, BERT base. According to information shared on the [Hugging Face](https://huggingface.co/docs/transformers/model_doc/distilbert) page, it reduces parameter count by about 40%, resulting in significant gains in computational efficiency. Achieving a processing speed that is 60% faster, while maintaining over 95% of BERT's performance levels, DistilBERT therefore has a balance between resource efficiency and model effectiveness. In our comparisons with other groups using more classical versions of Bert, we observed that DistilBERT can sometimes outperformed them in performance metrics. As mentioned during our discussion with Prof. Vlachos, this superiority can be attributed to its capacity to reduce overfitting using a smaller number of parameters. From what we understood, it is more plausible that models with a higher parameter count are more susceptible to overfitting the training data, resulting in inadequate performance when applied to new data.


In addition, DistilBERT stands out from the other vectorization methods we have analyzed so far due to its ability to grasp the overall context of a sentence. This means that within the same dataset, a word can be characterized differently based on the broader context of the sentence. In our analysis, this aspect proves particularly beneficial, as illustrated by the word "environment." Depending on the general context, this word can refer to the climate-related environment or the working environment, which is unrelated to environmental claims. Similarly, the verb "reduce," which emerged as a buzzword in the EDA, can have multiple interpretations. A company may reduce its ecological footprint, but it could also reduce its balance sheet debt ratio. Consequently, the capacity to capture sentence context enhances the predictive performance of our model. This increase in performance can be observed in all three metrics.


## Augmented dataset
For all the results that follow, these have been created and taken from the following file: models_prediction_augmented.ipynb

After conducting a thorough analysis on enhancing our models through the incorporation of diverse classifiers and vectorizers, as well as fine-tuning each model individually, we aimed to investigate the impact of improving the initial dataset on performance metrics. In light of this, leveraging chatGPT's assistance, we supplemented our testing dataset with an additional 415 environmental claims. Once the size of our training data set was increased, we proceeded to run the same models as before, but this time on the newly expanded dataset. 

But before examining the outcomes, in what way can the inclusion of additional data improve the performance of a model?
- More learning examples: A larger dataset provides more sentences labeled as environmental claims for the model to learn from, helping it understand the patterns and characteristics of such claims. Therefore, our different models may improve at recognizing what truely differentiate environmental claims from non-environmental ones.
- Better feature representation: With an increased number of environmental claims in the dataset, the model becomes familiar with a wider range of sentence structures, wording, and contextual information associated with environmental claims. Hence, when faced with new unseen sentences, our different models may be better as labelleing them as environmental claims or not accuratly.

Now, let's examine the results of our models using this expanded dataset and determine if it has genuinely improved the performance metrics, as suggested by the reasons above.


<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/2db8c196-079e-472e-8825-99c3a4655e7b"> <br>

In comparison to the previous table, there are no significant improvements observed. The DistilBert and Logistic combination still achieves the highest accuracy score of 87.67%. This combination also stands out as the best model in terms of precision and recall metrics.

Similar to the previous findings, the recall measurements remain quite low, around 50%. This is concerning for of our study, as it indicates that our model can only identify 50% of the environmental claims present in the sentence corpus. This limitation poses a risk in drawing false conclusions from our analysis of bank annual reports, as the model may miss a substantial number of actual environmental claims documented in these reports. Consequently, using such models could potentially result in a significant underestimation of the environmental ambitions communicated by the banks.

To further improve our performance metrics via dataset modification, we are taking the next step by using not only an expanded dataset but also ensuring its balance (i.e., same observation per class).

## Augmented & Balanced dataset
For all the results that follow, these have been created and taken from the following file: models_prediction_balanced.ipynb

As mentioned privately, this sections will now use a balanced dataset. To do this, we ensured that the number of environmental and non-environmental claims was equal within our training set. On this basis, we removed 901 non-environmental claims to obtain a balanced dataset with 957 sentences per class. 

As we have seen in class with Prof. Michalis Vlachos, better performance metrics can be achieved through a balanced dataset thanks to an avoidance of biais. Indeed, when the we have an unequal number of observations per class, the model might become biased towards the majority class. Our our case, sinces they were more non-environmental claims than environmental claims in the beginning, the model may have focusedd more on learning patterns related to non-environmental claims and struggle to accurately identify environmental claims. 

Now, let's see if having a balanced daataset is improving the performance metrics !

<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/e39ae73a-831f-4429-8930-51dbf0590894">

Unfortunately, once again, we did not observe a consistent improvement across all models.

However, two notable observations stand out:

- Firstly, focusing on the metric of accuracy alone, particularly with the TF-IDF vectorizer, we observed a decrease compared to our initial results. In our opinion, this could be attributed to the reduction in the document size, resulting in a smaller training set. With fewer sentences available, the TF-IDF model may struggle to effectively identify the most frequent words in the data. Therefore, if such words appear in the testing set, their weight would not be reduced as it should.
- Secondly, when combining BOW with Random Forest, we observed an exceptionally high recall rate of 85.94%. This represents a significant improvement compared to the initial table's recall value of 45.31% and the augmented dataset's value of 57.81%. As mentioned earlier, our model may suffer from less bias, particularly when using a simple vectorizer like BOW that counts the occurrence of words. The balanced dataset exposed the model to an equal number of environmental and non-environmental claims, enabling it to focus more equally than before on detecting both types of claims. This is reflected in the high recall metric. In the opposite way, the model makes more errors in the correct identification of all labels (0 and 1), which can be seen in particular with a slight drop in accuracy.


## GPT-3 Text Classifier 
For all the results that follow, these have been created and taken from the following file: GPT_3_finetuning_ada.ipynb & GPT_3_finetuning_davinci.ipynb


As previously stated in this report, the analysis of two additional models has been postponed until the end of the study. These models, although implemented by us on python, originate more from external sources and utilize GPT-3 Text Classifier with davinci and ada as the chosen models. We did not wish to directly compare these models with previously analyzed models because we did not believe we have significantly contributed to the establishment of these models in the context of our analysis (e.g., no direct fine-tuning and parameter search). Nonetheless, as shown in the confusion matrix below, these two models typically produce the greatest results. This is precisely why we incorporate them into the report, as their inclusion enhances the performance of our model, which is the primary objective of our study/assignement.

Given that an augmented or balanced dataset did not substantially improve results overall, but rather in isolation, the following analysis does not incorporate these new aspects. However, for this final analysis, we have added the development dataset to our training set, which contains both environmental and non-environmental claims. 

__ada__
<img width="700" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/8da79c48-9dc7-43b7-9268-052ca1859ca8">


__davinci__
<img width="700" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/d498e3a1-31ad-46bc-8d98-ce039f32915c">


For the first time in our analysis, the utilization of fine-tuning GPT-3 with the Davinci model has allowed us to achieve an accuracy exceeding 90%. This represents an improvement of 11.33% compared to our default rate and positions this particular model as our new top performer. When comparing it to our previous best model from the initial table, which combined Logistic regression and DistilBert, our new model surpasses it by 2.33% in accuracy, 2.18% in precision, and 12.5% in recall.

This outcome is not unexpected considering the fundamental structure of this model. As disscused in class with Prof. Michalis Vlachos, GPT-3 possesses an extensive parameter capacity, enabling it to effectively capture and comprehend complex patterns and relationships present in the data. This naturally leads to better results. However, it is worth noting that our new best model does not achieve the highest recall observed thus far. The model trained on a balanced tranined demonstrated a recall of 85.94%. The Ada model also delivered promising results, albeit slightly inferior to the Davinci model.

Based on the consistent performance across all three metrics, we can conclude that the last two models are currently the top performers. Moving forward with our analysis of annual reports from different banks, we have opted to use the ada model instead of davinci due to financial considerations. The davinci model incurs additional costs due to the extensive lines of code that need to be executed.



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

__Examples__

Before proceeding further, let's take a moment to examine the phrases that our model has identified as environmental claims in annual reports. Below are some examples:

Environmental claims:
- BCV 2020: "It oﬀers a broad, transparent view of what we are doing to fulﬁll our commitment to promoting economically, socially, and environmentally sustainable development." 
- Crédit Suisse 2020: "We launched a new Sustainability, Research & Investment Solutions (SRI) function at the Executive Board level, underlining the sharpened focus on sustainability."
- Goldman Sachs 2019: "We are  focused on ensuring our efforts in this area are aligned  with and accretive to our overall sustainability objectives."


Non-environmental claims:
- BCV 2019: "The Board of Directors is recommending that shareholders approve a 10-for-1 stock split, in order to  enhance the liquidity of BCV’s share."
- UBS 2020: "We also provide a combined annual report for  UBS Group AG and UBS AG consolidated, which additionally  includes the consolidated financial statements of UBS AG as  well as supplemental disclosures required under SEC  regulations and is the basis for our SEC Form 20-F filing."

These examples demonstrate the effectiveness of our model. Moreover, we can observe that the phrases categorized as environmental claims are generally specific to the environment, with a few exceptions, and do not encompass the entirety of ESG vocabulary. For instance, the sentence from J.P. Morgan in 2020 stating, "As you know, we have long championed the essential role of banking in a community - its potential for bringing people together, for enabling companies and individuals to reach for their dreams." has not been classified as an environmental claim. This further reinforces our confidence in the quality of our model.

However, it is important to note that our model is not perfect. In the case of J.M. Morgan in 2019, the following sentence has been identified as an environmental claim: "Over the last five years, for example, we've used technology and machine learning to reduce fraud losses in the credit card business by 50%." This demonstrates that false predictions can occur for environmental claims, indicating that our model's is not perfect (e.g., not a a 100% precision). It is possible that in this particular sentence, the presence of the word "reduce" influenced its classification as environmentally related.

__Analysis 1__

As mentioned previously, in the preliminary analysis, we will assess each chosen bank separately by comparing the variations in their environmental claims over two consecutive years with the corresponding changes in CO2 emissions. This evaluation aims to determine the consistency or disparity between the bank's emphasis on climate change in their annual reports through environmental claims and the actual trend of their CO2 emissions. The table below presents the findings of this analysis.


<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/293af090-bf35-4012-89a3-50a150c290ef"> <br>


The first thing to notice is that for all of the selected banks, the number of environmental claims has significantly increased between 2019 and 2020, indicating a growing emphasis on environmental-relatated topics by banks. This observed trend could potentially be attributed to various factors, such as external pressure from stakeholders urging the bank to address environmental concerns in response to their growing societal significance. Alternatively, it may stem from the bank's genuine commitment to align its climate actions. To gain further insights, let us examine the evolution of CO2 emissions.

Out of the five actors examined, only two experienced a decrease in their CO2 emissions during the analyzed period—Crédi Suisse and JP Morgan. Consequently, the remaining three players demonstrated an increase in their environmental communication while simultaneously increasing their CO2 emissions. The case of UBS is particularly noteworthy, as our model identified 19 additional instances of environmental claims in 2020, while there was an increase of 335,550 in CO2 emissions over the same period. BCV and Goldman Sachs also exhibited a similar but less significant trend, with a smaller increase in emissions.

For instance, let's consider a sentence from UBS's 2020 annual report, which our model predicts as an environmental claim: "*We delivered the best of UBS to our clients and  extended our leadership in sustainability.*"

However, does this imply that UBS is engaging in greenwashing? As discussed later in the analysis, it is challenging to draw a definitive conclusion based solely on this evidence. To ascertain the presence of greenwashing, a more extended analysis over a longer timeframe would be necessary. Nevertheless, the observations made regarding UBS could be considered a warning sign of potential greenwashing activities. Hence, an investor (e.g., a pension fund) could employ the developed model to identify such disparities between claims and CO2 emissions, enabling them to delve deeper into their analysis by requesting explanations from UBS regarding the reasons behind this discrepancy.


Furthermore, it is worth highlighting the contrast in the number of claims detected by our model between the chosen Swiss banks and American banks. Remarkably, our findings indicate that the American banks fall significantly short in terms of publicly disclosing their climate objectives compared to their Swiss counterparts. Despite Goldman Sachs' considerably larger balance sheet and extensive operations, our model predicts a mere two environmental claims made by the bank in 2019, while BCV, a Swiss bank, stands out with 16 claims during the same period. This disparity may be attributed, in part, to potential factors like public pressure that could lower climate-related disclosures in the US context.

In addition, from the data provided in the table, it is evident that there exists a notable difference among the banks in terms of their CO2 emissions. This discrepancy can be attributed to variations in their size and operations, resulting in significantly differing levels of CO2 emissions. This observation brings us to our second analysis, wherein we propose the use of a metric to standardize the measurement of CO2 emissions: CO2 intensity.


__Analysis 2__

The CO2 intensity measure, as previously mentioned, evaluates a company's CO2 emissions in relation to the revenue it generates. This measure enables the comparison of companies of varying sizes and impacts, as it is reasonable to assume that companies with higher revenue will be larger and have a bigger societal influence. For each bank, a CO2 intensity measure will be established to standardize their CO2 emissions based on their respective revenues. This normalization facilitates fair comparisons among banks operating in the same industry but differing in size. By utilizing this metric, we will rank banks according to their CO2 intensity and compare it with the number of environmental claims they have. This analysis aims to identify cases where a bank, despite having a low CO2 intensity ranking, extensively promotes its climate ambitions.



<img width="800" alt="image" src="https://github.com/noelopez-E4S/env_claims_detector/assets/114017894/f5fdeaa2-3eda-4fd4-9bc2-b32e47e69245"> <br>



For the reader information: in the provided table, companies with the lowest values in the "Rank - CO2 Intensity" column have the best CO2 intensity metrics. However, companies with the lowest values in the "Rank - Env. Claims" column are the ones that make the most environmental claims in their reports.

First, compared to the table in Analysis 1, where only global CO2 emissions were reported, the range of values for different banks in relation to their CO2 intensity is narrower. This allows for a better comparison between the companies as desired.

From analyzing both rankings, it becomes apparent that the correlation between the two is not perfect. Just because a company communicates extensively about its environmental ambitions does not necessarily make it the best environmental performer. Although this result was expected, it is still interesting to note. Furthermore, two notable observations can be made based on this table, concerning Goldman Sachs and UBS for the year 2020.

In the year 2020, Goldman Sachs ranked eight in terms of CO2 intensity, scoring 35.95. However, it also ranked poorly in the number of environmental claims compared to other players. This indicates a situation contrary to greenwashing, where the company has a significant negative impact on the environment without attempting to conceal it through positive ESG language in its annual report. Although this is unfavorable from an environmental impact perspective, at least the bank does not deceive its investors and other stakeholders (i.e., does not appear as greenwashing).

In contrast, UBS is facing another challenging situation in 2020. According to its annual report, it ranks first in terms of the number of environmental claims, but lags behind its competitors in CO2 intensity (9 position). Again, this raises concerns about potential greenwashing practices by the largest Swiss bank. In addition, the same observation can be observed for 2019, where the swiss bank rank second in environmental claims, but last in terms of CO2 intensity.

When comparing the other Swiss banks to their US counterparts, it can be seen that US banks tend to rank lower in terms of CO2 intensity. Specifically, they occupy positions 5, 6, 7, and 8 in the ranking. This finding may suggest that Swiss banks, with the exception of UBD, are more advanced in their commitment to creating a more sustainable world.

Again, it is important to note that this conclusion is drawn from a limited sample and should not be solely relied upon for making greenwashing accusations. To obtain more accurate results, it is crucial to expand the study to include a larger number of banks and conduct research over a longer period. Despite this limitation, we have confidence in the effectiveness of our model in identifying potential instances of greenwashing. With an impressive recall rate of 82.81%, our model has proven its ability to detect environmental claims across various text sources. This makes it a valuable tool, which can be shared with portfolio managers, for instance, enabling them to “red-flag” companies that may be engaging in greenwashing practices. By simply inputting a URL, our model can analyze the number of claims made and facilitate necessary comparisons with environmental metrics. 


## Limitation

- **Transparency**: The C02 data provided by Trucost relies on self-reporting. Therefore, it should be noted that during the years under analysis (2019 and 2020), the sustainability reports of the banks, from which the C02 emission figures were primarily obtained, may not have undergone external audits. Consequently, there is a possibility that these figures may not precisely reflect the actual practices of these companies. 

- **Change takes time**: Implementing changes within the financial sector is a time-consuming process. Drawing conclusions about any greenwashing activities based on a two-year analysis is difficult, and to be fully transparent not doable. Transforming processes and investment decisions significantly necessitates internal changes, such as risk management and reporting. Therefore, it would be valuable to conduct a similar analysis over a longer timeframe to demonstrate the progression of environmental claims and key environmental metrics. This approach would enable a more objective analysis. Similarly, exploring a time-lagged analysis would be interesting. This entails examining the claims made by a bank in 2021 and assessing their correlation with emissions not in the same year, but in 2022.


- **The accuracy of scope 3**: The accuracy of scope 3 remains a challenge for estimating emissions from financial institutions. However, can we accurately determine the CO2 emissions impact of BCV's financing loan to a local SME involved in piano construction? he task of obtaining precise figures in this regard remains difficult. Consequently, relying solely on numerical metrics is inadequate when determining whether a company is greenwashed. Instead, it should be important to consider other factors, such as their involvement in financing new O&G projects for example, even if the investment amount is minimal. Although such projects may not significantly inflate CO2 emissions relative to the limited capital invested, they provide insights into the bank's overall policy and the sectors in which the bank want to allocate its resources.


- **Greenwashing ?** : In summary, this model proves highly effective at assessing environmental claims but cannot directly determine greenwashing. To address this, the user has to conduct a comprehensive comparison across various environmental metrics such as C02 emissions, CO2 emissions, and others. Although this presents challenges, it would be intriguing to develop a model specifically designed to detect greenwashing directly. One approach could involve researchers concentrating on companies known to engage in greenwashing and labeling specific “greenwashing claims” in their reports. Indeed, it is possible that certain words regularly appear in the annual reports of companies involved in greenwashing practices. However, creating such a model would also carry legal risks, as accusing a company of greenwashing is a serious matter, as demonstrated by the [DWS](https://www.reuters.com/business/finance/deutsche-banks-dws-sued-by-consumer-group-over-alleged-greenwashing-2022-10-24/) case last year. Hence, this model would need human verification given the fine line between greenwashing and genuine climate ambitions, which makes the detection task even more challenging.

## Conclusion



