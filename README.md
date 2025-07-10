# Student_Performance_Analysis

## repository for the course of Python for AI 🏫


<p>
    Used python 3.10 due to some compatability issues with tensorflow. 
    using docker and creating a docker container for setting the correct versions used for this project is probably easier then craeting a virtual environment with venv
    But if ´you´ wish to try i've added 2 requirements files:
</p>

<ol>
    <li>requirements.txt => used for setting up python virtual environment</li> 
    <li>requirements_docker.txt => used by the dockerfile to build the docker image</li> 
</ol>

## How to run the project ? 🖥️
in any terminal (preferably one that supports ANSI CSI) go to the root of the project
and run with python the `project.py` file
```
$ cd /path/to/repo/Student_Performance_Analysis
$ python3 project.py
```

## Feedback
<p>
    It was mentioned by my teacher that while the project was good for what we learned 
    during the course , the names of the chapters were confusing and the code structure can be much better 
    IMO. (especially the __init__.py in the root folder 😢) But this was mainly because i rushed this in under 2 Days.
    it would also been more intersesting to wrap this into some classes instead of using plain Python scripts (sure for the exploration part script might be easier)
    especially when you want to streamline thie process (data cleaning + visualization + logging) into a production ready.
    if we take a look at the book of Aurélien Géron - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
    On page 779 to 784 (appendix A) he cover's the main bullet points on how to get from a project to production ready status.
</p>
