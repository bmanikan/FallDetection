We are using two different datasets such as URFD and CMDFALL dataset. 

URFD dataset website: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html

We initially started with the URFD dataset containing a total of 70 videos where 30 videos for fall activity and 40 videos for ADL activity. The sample video and extracted trajectory image for both the activity is provided in the URFD folder. The deep learning models are trained on this dataset first and evaluated / further trained on subsequent datasets in the future. 

The results presented in the results folder.

CMDFALL dataset Website: https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html

The second dataset is the CMDFALL dataset. This dataset consists of huge data when compared to URFD dataset. This dataset include 383 total videos from 50 subjects with eachone performing differetn fall activities and ADL activities. The subjects are chose from different age groups and gender to increase the generalizability performance.

Out of these 383 videos, we can extract,
* Total possible sequences include 14029
* Number of fall sequences in the dataset is 5472
* Number of adl sequences in the dataset is 8557
* Sequences greater than 50 Frames are 9481

The deep learning model trained in the URFD dataset is evaluated in this dataset and performance parameters are documented in the results folder.
