The code is used for the article *Automatic Tunnel Crack Inspection Using an Efficient Mobile Imaging Module and a Lightweight CNN* which is under review for the journal of *IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS*.
The data will be made available when the article is published.


### Installation
The code was tested with Anaconda and Python 3.6. After installing the Anaconda environment:

Install dependencies:
   
    conda install --yes --file requirements.txt
   

### Training
Follow steps below to train your model:

0. Configure your dataset path in `mypath.py`
1. run visdom
    ```
	 python -m visdom.server
    ```
2. Input arguments: (see full input arguments via python train.py --help):
    ```
	python train.py 
    ```


### Testing
If you want to test the image (see full input arguments via python test.py --help),  run:
	
   ```
	python test.py 
   ```

