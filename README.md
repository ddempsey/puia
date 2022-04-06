# Puia
Time-series feature analysis and eruption forecasting for volcano data. Successor package to Whakaari. This model implements a time series feature engineering and classification workflow that issues eruption alerts based on real-time tremor data. Includes workflows for real-time forecasters with email alerting.

## Installation

Ensure you have Anaconda Python 3.7 or higher installed. Then

1. Clone the repo

```bash
git clone https://github.com/ddempsey/puia
```

2. CD into the repo, switch to the transfer-learning branch and create a conda environment

```bash
cd puia

git checkout transfer-learning

conda env create -f environment.yml

conda activate puia
```

## Real-time forecaster setup
These steps are only required if you want to run a real-time forecaster on a linux VM.

1. Install keyrings.alt

```bash
pip install keyrings.alt
```

2. Set up Apache web-server and change permissions of html folder

```bash
sudo apt install apache2

sudo chmod 777 /var/www/html
```

then follow [these instructions](sudo apt install apache2 https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-18-04-quickstart). 

The installation has been tested on Windows, Mac and Unix operating systems. Total install with Anaconda Python should be less than 10 minutes.

## Setting up the controller
Open controller.py and under "__main__" make sure keyfile, mail_from, monitor_mail_to_file, alert_mail_to_file are appropriately set.

## Running the controller
Open a screen, activate the environment and then run the controller script
```bash
screen -S controller

cd scripts

conda activate puia

python controller.py
```

Ctrl+A, Ctrl+D to close the screen and then exit the VM.

## Disclaimers
1. Eruption forecast models cannot predict future eruptions with 100% accuracy. Instead, they can signal a higher likelihood of an eruption under certain conditions. If you do not understand those conditions, you should not be making authorative statements about eruption likelihood.

2. Forecast models provide a probabilistic prediction of the future. During periods of higher risk, it issues an alert that an eruption is *more likely* to occur in the immediate future. 

3. This software is not guaranteed to be free of bugs or errors. Most codes have a few errors and, provided these are minor, they will have only a marginal effect on accuracy and performance. That being said, if you discover a bug or error, please report this at [https://github.com/ddempsey/puia/issues](https://github.com/ddempsey/puia/issues).

4. This is not intended as an API for designing your own eruption forecast model. Nor is it designed to be especially user-friendly for Python/Machine Learning novices. Nevertheless, if you do want to adapt this model for another volcano, we encourage you to do that and are happy to answer queries about the best way forward. 

