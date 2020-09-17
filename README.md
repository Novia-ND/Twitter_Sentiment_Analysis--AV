
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="">
    <img src="https://lh3.googleusercontent.com/x3XxTcEYG6hYRZwnWAUfMavRfNNBl8OZweUgZDf2jUJ3qjg2p91Y8MudeXumaQLily0" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Twitter Sentiment Analysisr</h3>

  <p align="center">
    An Awesome ML model that Predicts the sentiment of tweet
    <br />

  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [License](#license)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

This is a project made in python which uses NLP algorithm , classification algorithm and clustering algorithm for making this ML model thats predicts whether the text entered tweet Spam or Ham

many algorithms are tested and depending upon the f1 score the best model of classification is selected and used. for clustering we used minibatchkmeans which is very effective for large datasets

also we have a frontend webpages where we integrated ML model into this using Flask micro web frameworks

<p align="center">
  <a href="">
    <img src="https://github.com/Novia-2018/Twitter_Sentiment_Analysis--AV/blob/master/screenshots/Screenshot%20(264).png?raw=true" alt="Logo" width="800" height="400">
    </a></p>

<p align="center">
  <a href="">
    <img src="https://github.com/Lance-Dsilva/Twitter_Sentiment_Analysis--AV/blob/master/screenshots/Screenshot%20(263).png?raw=true" alt="Logo" width="800" height="400">
    </a></p>

### Built With

* [Bootstrap](https://getbootstrap.com)
* [Python](https://docs.python.org/3/m)
* [django](https://docs.djangoproject.com/en/3.0/)


### Installation

This are the various libraries that need to be installed prior
```sh
pip install sklearn
pip install matplotlib
pip install pandas
pip install numpy
pip install wordcloud
```
2. for woking with flask frameworks you  first need to install flask
```sh
pip install django
```
4. command for running flask server once you have all the required file in same directory assume X directory 
```JS
cd X
django-admin startproject project_name     // to start a new project
django-admin startapp app_name      // to start a new app 
python manage.py runserver         // to start teh server
python manage.py migrate           // to migrate 
python manage.py makemigrations books       // used to migrate 
python magane.py sqlmigrate books 0001       // to migrate database
python manage.py shell
python manage.py createsuperuser
```



 
 For more information you can refer to https://towardsdatascience.com/
 
 
 

### Team members- 

http://lndindustries.online/
              
