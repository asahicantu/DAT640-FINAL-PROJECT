# DAT640-FINAL-PROJECT MS-MARCO Document Re-Ranking
Final project for the Information Retrieval and Text Mining. Project Name: MS-MARCO Document Reranking
* **Asahi Cantu - 253964**
* **Shaon Rahman - 253965**

## Project Description:
Microsoft MAchine Reading COmprehension Dataset  is a copmilation of queries and documents retrieved from Microsoft Bing Platform. It contains a big dataset ~ 22GB of documents and queries

This project aims to contribute to the analysis of big data sets and the information retrieval models in the field of "document-re-ranking" and "Learning To Rank (LTR)", provide a detailed analysis of the retrieved data and how by employing machine learning and deep learning algorithms for document re-ranking it is possible to be more accurate towards the intended information to be retrieved. The main purpose of such task is to create systems intelligent enough by helping in real world tasks to deliver the right information by a previous analysis of human written questions and return the most relevant information.

## The analysis takes place in two different tasks:
* Document re-ranking with a base method. Common machine learning regression models are applied to known retrieved documents, and then compared against original ranking BM25 method to prove the efficiency of such models towards expected values.
    
* Document re-ranking with an advanced method. A deep learning method (BERT) is used to re-rank the given documents and compared against the developed models.


In each task the results are displayed, interpreted and compared one against the other. Based on those results discussion and conclusions are presented towards the efficiency and usability of such models for real life applications.

Implement a baseline and advance methods for document re-ranking for a corpus dataset called MS-MARCO provided by Microsoft where a set of 100,000 real questions were processed by Microsoft's search engine Bingfootnote{Microsoft Bing is a web search engine owned and operated by Microsoft. The service has its origins in Microsoft's previous search engines: MSN Search, Windows Live Search and later Live Search. Bing provides a variety of search services, including web, video, image and map search product. The dataset brings already a top-100 retrieved document set per question, being each set the most relevant documents found by the search engine and a set of labeled documents, where each query contains only one most relevant document. The accuracy of the algorithms will be translated by the rank in which such relevant document is shown once the information is retrievedcraswell2020overview.

## How to use this repository

### Running main model
#### Model Evaluation

Execute [main.py](main.py) to visualize the results obtained.

```bash
|   python main.py 
```

#### Full Project development
Open the jupyter notebook [Development.ipynb](Development.ipynb)


## Repository Contents

A full working version of the same notebook is available in Kaggle [here](https://www.kaggle.com/asahicantu/dat640-final-project-ms-marco/edit)

```bash
|   Development.ipynb                   # This Readme File
|   main.py                             # Python Scrypt. Runnable code.  opens every file in the  'pickles' directory to perform model evaluation
|   notebook.ipynb                      # Descriptive information of the python script, enable to modify and understand the re-ranking process.
|   Report.pdf                          # Contains a comprehensive description and documentation for this project                                    
|
\---pickles                             # Containes serialized information for the queries and rankngs used in this project
        advanced_rankings.pkl           # Contains the document rankings used in the advanced model.
        ADV_QRELS.pkl                   # Provides Q-rels and queries used in the dadvanced methodology
        basic_rankings.pkl              # Contains the document rankings used in the basic model.
        corpus_embeddings.pkl           # BERT encoded embeddings for each summarized document
        QRELS.pkl                       # Qrels used ini this project
        QUERIES.pkl                     # Queries file used for this project
        query_embeddings.pkl            # BERT encoded embeddings for each query used in this project
        training_data.pkl               # Trained data models for the base method approeach
```
