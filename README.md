##Steps

1. Create conda environment:

        conda create -n condaeup_flask_ml python=3.6.1
        source activate condaeup_flask_ml

2. Create requiremerents.txt file

        numpy
        pandas
        firefly-python
        Flask
        Flask-WTF
        tensorflow
        cv2

3. Install required packages:

		#cd number_recognition_tflow
        conda install --yes --file requirements.txt
        
        or

        #iterate over all lines in the requirements.txt file.
        while read requirement; do conda install --yes $requirement; done < requirements.txt

    If a requirement is missing, then use:

        pip install firefly-python
        pip install opencv-python

    If conda fails, we can also use pip:

        pip install -r requirements.txt


4. Running Flask

To run the application you can either use the flask command or python’s -m switch with Flask. Before you can do that you need to tell your terminal the application to work with by exporting the FLASK_APP environment variable:

a. For Mac:

    $ cd webapp
    $ export FLASK_APP=webapp.py
    $ export FLASK_DEBUG = 1
    $ flask run
     * Running on http://127.0.0.1:5000/

b. For Win:

    $ cd webapp
    $ set FLASK_APP=webapp.py
    $ set FLASK_DEBUG = 1
    $ flask run
    


