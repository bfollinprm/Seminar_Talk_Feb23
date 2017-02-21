This folder contains a Jupyter notebook (Follin_Talk.ipynb) which contains executable code to reproduce
all the results I'll use in my talk. 

How to:
	1. First, the notebook does require some dependencies, listed in requirements.txt. If you already have a working Python
	environment, you should be able to ensure these modules are installed in your environment using
		$ pip install -r requirements.txt
	If the above throws an error, you're either disconnected to the internet or have a faulty or nonexistent easy_install
	instance, which comes with any recent Python distribution. Try updating your Python version, or consider creating a new
	python environmnet using a popular package like anaconda (https://www.continuum.io/downloads).

	2. The notebook is then opened by
		$ jupyter notebook Follin_Talk.ipynb
	The code will download some data (~ 50 MB worth) to </path/to/this/directory>/data/chains/ that's just too large to wrap 
	with the code itself. Run a cell by highlighting it and entering 
		<shift>+<return/enter>
	on your keyboard. The notebook will allow you to run cells in any order, but if you want it to work you should move from
	the top down.