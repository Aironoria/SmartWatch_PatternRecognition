ignored_label =['touchdown','touchup','nothing']
# ignored_label =[]
network ='cnn'

siamese_train_size =1000*1
siamese_test_size=10

use_Jitter =False
use_Time_warp =False
use_Mag_warp =False

sigma=0.01
margin=0.1
printed =False
model_dir= "overall"
start_index = None

embedding_size = 128