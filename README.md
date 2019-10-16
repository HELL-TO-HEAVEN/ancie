# Ancie - Historic Manuscript Exploration

#### IAM Dataset
To download the IAM dataset you can run the following

    wget -r -l1 --no-parent --no-check-certificate  http://<your_user>:<your_pass>@www.fki.inf.unibe.ch/DBs/iamDB/data/forms/
    wget http://<your_user>:<your_pass>@www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt

* Available for download from [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database) (registration is required)

* You only need to following directories:

       <YOUR DATA FOLDER>
        ├── forms
            ├── *.png
        └── words.txt