
Only you can see this message
This story is not part of the metered paywall. Learn more

Exploratory Data Analysis(beginner) , Univariate , Bivariate and Multivariate — Habberman dataset.
Go to the profile of purnasai gudikandula
purnasai gudikandula
Nov 10, 2018
Those who are new to data science and machine learning and if you are looking for some guidence and resources to prepare, then this blog is so great one that it had give some nice advices and guidence and also nearly given 2000 resources to prepare mostly for free. this blog has reached 25000views in just 5 days after it published.

please check out that blog about “Ultimate guide and resources to datascience 2019”

This blog is written imaging a newbie to Data science (with some knowledge on python and its packages like numpy,pandas,matplotlib,seaborn) in mind. so i will take you to clear explanation of everything that i can. after reading this blog you will get an idea over Exploratory Data Analysis(EDA) using Univariate visualisations, bivariate visualisation, and multivariate visualisation and some plots. Dont worry if these terms sound alien to you. will tell you clearly what every term is used for.

Data science life cycle:
Every Data science Beginner, working professional, student or practitioner follows few steps while doing. i will tell you about all these steps in simple tems for your understanding.

1.Hypothesis definition:- A proposed explanation as starting point for further investigation.

Ex:- A(company) wants to release a Raincoat(product) in Summer. now company is in dilemma whether to release the product or not.(i know its a bad idea, but for understanding, lets think this.)

2. Data Acquistion:- collecting required data.

Ex:- collecting last 10 years of data in a certain region.

3.Exploratory Data Analysis(EDA):- Analysing colleted data using some concepts(will see them below).

Ex: on colleted data(existing data)data scientists will perform some analysis and decide, what are the features/metrics to consider for model building.

4.Model building:- this is where Machine learning comes into light.

Ex:- by using metrics(outputs of EDA), they will predict(using ML )whether the product will be successful or not, if it goes into market.

5.Result report:- after doing EDA and Model building, it generates results.

Ex: as a result of all above steps we get some results, which decides whether to start production or not.

6.final Product:- based on the result, we will get a product.

Ex:- if the result generated is positive, A(company) can start production. if result is negative, A wont start production.

Data science life cycle
Exploratory Data Analysis:-
By definition, exploratory data analysis is an approach to analysing data to summarise their main characteristics, often with visual methods.

in other words, we perform analysis on data that we collected, to find important metrics/features by using some nice and pretty visualisatitons.

every person takes some decisions in their life considering few points at some situations. to be accurate at these decisions data scientist do some EDA on data.

For Ex: if you want to join in a graduation school, what do you do ?

you collects some opinions(data) from alumni, students, friends, family. now from those opinions you will find some key points(metrics/features), lets say the points like placement rate, reputation, faculty to student ratio, labs and infrastructure. if you are happy with these points, only then you will join in it.

this is what exactly that happens with EDA from data scientist point of view. if you want to more about Exploratory data analysis please check here.

Some theory:

Exploratory Data Analysis is majorly performed using the following methods:

Univariate analysis:- provides summary statistics for each field in the raw data set (or) summary only on one variable. Ex:- CDF,PDF,Box plot, Violin plot.(dont worry, will see below what each of them is)
Bivariate analysis:- is performed to find the relationship between each variable in the dataset and the target variable of interest (or) using 2 variables and finding realtionship between them.Ex:-Box plot,Voilin plot.
Multivariate analysis:- is performed to understand interactions between different fields in the dataset (or) finding interactions between variables more than 2. Ex:- Pair plot and 3D scatter plot.
more than saying all these concepts theoretically, lets see them by doing some excercise. lets download a data set from kaggle(home for Data scientists) , you can download and know more about it here →Habberman dataset.

Haberman's Survival Data Set

Survival of patients who had undergone surgery for breast cancer
www.kaggle.com	
I am using ubuntu 18.04

my ubuntu version
and python 3.6(know your version by typing python in command prompt)

checking python version
with anaconda 5.2.0(know your version by typing conda list anaconda).

checking anaconda version
before you get your hands on please make sure that you installed some libraries like numpy, pandas, matplotlib, seaborn.

lets start. open your command prompt and type “jupyter notebook”, this will take you to jupyter environment where you can visualise graphs.

opening jupyter notebook
lets start. if you are done with all the above process, then run your notebooks paralell to mine.

1.importing libraries:
%matplotlib inline in above code snippet allows us to view our graphs in jupyter notebook itself.

Load dataset: import the dataset to a variable of your own convention. i am going with Cancer_sur by using pandas function pd.read_csv().


loaind data set to a varaible
you can see top 5 lines of data by using Cancer_sur.head().

checking top5 rows of data
if you look at it, you can see top 5 rows, but not able to make sense, because there is no column labels to it. lets add columns to it.


adding column labels and loading dataset again to Cancer_sur variable.
in the above snippet, header = None removes its headers, names =[] adds column names to the dataset as “Age”, “Operation_year, “axil_nodes_det”, “Surv_status”.

2.Some Basic analysis:
lets see top 5 rows after updating labels using Cancer_sur.head().

image after labelling
lets see last 5 rows using Cancer_sur.tail().

last 5 rows
3.High level statistics:
u can see count(gives total rows),Mean(average),std(standard deviation from one point to another),min,max and total coulmns of dataset and its rows, its data types by using .describe() and .info().

observations:

observations.
.shape gives no of rows and columns.


total rows and columns of dataset
Surv_status is a target column where it gives 2 values 1(means survived) and 2(not survived). lets see them. in the entire dataset we have 225 rows(people) with value 1(survived) and 81rows(people) with value 2(not survived).

now lets see some univariate analysis.

Univariate analysis(PDF, CDF, Boxplot, Voilin plots,Distribution plots):-
Analysis done based only on one variable. we are not going to math behind these concepts, for now lets see what these are in graphs. (please have some basic idea on these concepts if you dont get them by seeing graphs).

Distribution plot:
people follow their own ways of coding that gives the similar results. Distribution plot gives the density of distributions from point to point in general terms.

we draw this using seaborn as sns, Facetgrid gives grid layout, Cancer_sur is variable that we loaded data into.Hue colors the value/columnname that you give to it. Size is graph size and mapping all these to sns.distplot on “Age” column.

from the above graph, you can observe that people with age 50 to 60 have more survival rate.

now lets draw the same with another column “Operation_year”

from the above graph we can say that people who had operation in the year from 58 to 66 had more survival rate.

CDF(cummulative distributive function), PDF(probability denstiy funtion):
we draw this using univariable “age”, and drawn cummulative distribution function and probability density funtion.

if we draw a straight line from Age value at 70, then it intersects the curve Cummulative distribution funtion(yellow) at value approximately equal to 0.8 i.e there are 80% people from cummulative sum of 30 to 70 age.

Bivariate analysis:
this gives the relation ship between the two variables , hence its called bivariate analysis.

Box plot:

Box plot is a nice way of viewing some statical values along with relation ship between two values.

note: from the next comming all graphs, graphs that are in blue shows value1(survived) and yellow with value2(not survived).

it uses seaborn as sns to visualise boxplot between X =’Surv_status’, y= “‘Age” , data= Cancer_surv, because it is where all our dataset loaded to

Now lets draw the Box plot between Surv_Status and Operation year.

it is observed that people that had operation in year 1958 to 1966 survived.

Voilin plot:
voilin plots also like box plots, but these give pdf along with box plots in it. they look a voilin, so named to .please see this image.

this is the same as above box plot, but here we used voilin plot to look more pretty and to get the pdf at the same time.

you can observe that people with operation year from 58 to 66 survived more as after that the grapsh decreased as you can see in above figure.

Multivariate Analysis:
Pair plot:
pair plot shows a clear and nice view of all variables and their realtion ship with all other variables.

it uses seaborn as sns to draw a pair plot with dataset variable Cancer_sur and colors the graph using Surv_status with size = 3

Observations:
finally observations are more important for every graph. as we seen above i made some conclusion to every graph.

4.Model building:
this is where ml comes into picture. as of now we did with EDA.

Excercise:
now after reading this blog please try to do some exploratory data analysis on your own dataset. for beginners i suggest titanic dataset from kaggle and iris dataset from kaggle.

for more pleas check out some great kaggle kernels to explore EDA more and also check out this kernel too.

please feel free to connect me on linkedin here below:

Purnasai Gudikandula - Udacity Pytroch Scholar - Udacity India | LinkedIn

View Purnasai Gudikandula's profile on LinkedIn, the world's largest professional community. Purnasai has 5 jobs listed…
www.linkedin.com	
please clap, share, and comment below.

Data ScienceData VisualizationDataData AnalysisBeginner
Go to the profile of purnasai gudikandula
purnasai gudikandula
Data science | Machine learning intern at applied ai https://www.linkedin.com/in/purnasai-gudikandula-366b71129/

Also tagged Beginner
A Beginner’s Guide to Machine Learning
Go to the profile of Shaurya Bhandari
Shaurya Bhandari
Feb 24
Also tagged Beginner
Writing is painful
Go to the profile of Nauris Dorbe
Nauris Dorbe
Mar 24
Also tagged Data Science
Transformers
Go to the profile of Giuliano Giacaglia
Giuliano Giacaglia
Mar 11
Responses
2 responses to your storyManage responses
Go to the profile of purnasai gudikandula
purnasai gudikandula
Nov 24, 2018
Hi Karthikeyan mg , axil_nodes_det is short form for auxilary nodes detected. auxilary nodes are also called as lymph nodes. they detect cancer causing cells in the body. and axil_nodes_det determines their count. if they are negative, no breast cancer. if positive, then they had breast cancer.

Applause from purnasai gudikandula (author)
Go to the profile of Karthikeyan mg
Karthikeyan mg
Nov 24, 2018
What does axil_nodes_det mean ?