# Description:
In Part 1 of this assessment item, you need to form a group (minimum of TWO students and a maximum FOUR students) and co-create:

* a one-page project proposal, and
* a preliminary data investigation report (multiple pages).

Task:
You must include the following components in your submission:

1. **List members of your group with signatures**.
2. **Project Title. (5 points)**
3. **A brief description of the problem you will solve or the question you will investigate.** You should have a paragraph describing the background and motivation of your project. (20 points)
4. **What data will you use and where will you get it? (10 points)**
   * Small datasets (such as Iris dataset) and synthetic datasets (simulated and not real-world datasets) are NOT acceptable.
   * The dataset should have at least 1,000 data samples, each of which has more than ten attributes.
5. **Which algorithm/techniques do you plan to use for different learning tasks? (20 points)**
   * You will need to implement the algorithm in Part 2 of the assessment item, so please carefully choose your targeting tasks and algorithms (at least one algorithm **per team member**).
   * The algorithms, for example, can be:
     * Apriori algorithm for frequent rules mining, or
     * KNN / Decision Tree / SVM classifier for a multi-class or a binary-class problem
   * Please give a short introduction to the algorithm you choose.
6. **What measurements will you use to evaluate the results?** The same as the chosen algorithms, a short description of the measurements should be included. NOTE: The measurements should correspond to your targeted problem in point 3 and are suitable for the algorithms you choose in point 5. (10 points)
7. **Some preliminary results MUST be reported as the supplementary material together with your one-page proposal, including:**
        * Data exploration: the number of the data samples, the types of attributes, and the statistical information of each attribute (five-number summary for the numerical attribute). (10 points)
        * Choose various data visualisation tools (such as boxplots, histogram, scatter box, etc.) to visualise all the attributes. (10 points)
        * Data pre-processing: identify if there are missing data and noisy data. Discuss how you will handle them. NOTE: there is ZERO mark for this part if your dataset does not have a data quality problem (missing data and noisy data). (10 points)

Individually, you must add one paragraph at the end, after the Preliminary Report and References, to highlight what 
you have contributed to the proposal. (5 points)

All other parts should be the same among group members.

## Resources:
Candidate datasets can be obtained from the online open data repository (such as UCI data repository) or other websites.

## Marking criteria:
You will be marked out of 100 points. Marks will be allocated as indicated in the task section above. Marks will be awarded depending on the proportion of the task that is successfully completed and the quality of the solution.
Please review the task thoroughly before you commence work on this assessment task. Ensure that you have addressed the relevant criteria when completing the assessment task.

## NOTE: Working with Pickles
The pickles in this project (.pickle files) are stored as compressed pickles. To load the data within, use the following:
```python
import bz2
import pickle

def decompress_pickle(filename):
    bz2_file = bz2.BZ2File(filename, "r")
    data = pickle.load(bz2_file)
    return data
```