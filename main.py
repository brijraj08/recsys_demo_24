import sys
import conv_transfer as ct
import numpy as np
import pandas as pd
import torch
from MF import MF_NEW
import time



""" working with dataset; utility matrix creation""" 

ratings_list = [i.strip().split("::") for i in open('/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('/movies.dat', 'r').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)


R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()
R = R_df.values
refined_dataset=ratings_df

#list of all users
unique_users = refined_dataset['UserID'].unique() 
#creating a list of all movie names in it
unique_movies = refined_dataset['MovieID'].unique()



users_list = refined_dataset['UserID'].tolist()
movie_list = refined_dataset['MovieID'].tolist()



ratings_list = refined_dataset['Rating'].tolist()

movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}

## creating a utility matrix for the available data

## Creating an empty array with (number of rows = number of movies) and (number of columns = number of users) rows as movies, columns as users

utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])


for i in range(int(len(ratings_list))):
  ## ith entry in users list and subtract 1 to get the index, we do the same for movies but we already defined a dictionary to get the index.
  utility_matrix[movies_dict[movie_list[i]]][int(users_list[i])-1] = ratings_list[i]

utility_matrix= utility_matrix.T
print("utility matrix created and transposed")

############## Masking
mask = np.isnan(utility_matrix)
masked_arr = np.ma.masked_array(utility_matrix, mask)


#creating a test matrix of 20%
test_m=np.zeros((len(masked_arr), len(masked_arr[0])))

for i in range(0, len(masked_arr)):
    for j in range(0, len(masked_arr[0])):
        if masked_arr.mask[i][j]==False:
            r1=np.random.randint(1,100,1)
            if r1 <20:
                test_m[i][j]=masked_arr[i][j]


#replacing 0 with nan
test_nan = np.where(test_m==0, np.nan, test_m)
#creating mask for test dataset matrix
test_mask = np.isnan(test_nan)          
masked_test_data = np.ma.masked_array(test_nan, test_mask) 
#extracting training data elements by subtracing test from full masked utility matrix 

train_nan= utility_matrix-test_m
#replacing 0 with nan
#creating masked train dataset for matrix factorization
train_mask = np.isnan(train_nan)
masked_train_data = np.ma.masked_array(train_nan, train_mask)
# masked_train_data=train_m2  



mf_orig = MF_NEW(masked_train_data, K=64, alpha=0.03, beta=0.01, iterations=150)
mf_orig.train()
user_m1=mf_orig.P
item_m1=mf_orig.Q
result1=mf_orig.full_matrix_new(user_m1, item_m1)

predicted_result1 = np.ma.masked_array(result1, test_mask) 

diff_test= predicted_result1-(masked_test_data)
diff1_test=np.abs(diff_test)
MAE_test=np.mean(diff1_test)
print("Original Performance Before Regularization")
print("MAE_test", MAE_test)

RMSE_test=np.sqrt(np.mean(np.square(diff_test)))
print("RMSE_test", RMSE_test)

# Record the start time
start_time = time.time()

import copy    

sel_data=copy.deepcopy(masked_train_data)    
p_users=.20
p_items=.30
rows=np.random.randint(0,len(sel_data), int(len(sel_data)*p_users))
cols=np.random.randint(0,len(sel_data[0]), int(len(sel_data[0])*p_items))
for i in rows:
    for j in cols:
        sel_data[i][j]=5 # Placeholder




data_with_noise=sel_data


################


v1= data_with_noise- masked_train_data
v2=np.where(v1==0, np.nan, v1)
mask_v=np.isnan(v2)
v3=np.ma.masked_array(masked_train_data, mask_v)

item_avg=np.zeros(len(masked_train_data[0]))
for i in range(0, len(masked_train_data[0])-1):
    item_avg[i]=np.mean(masked_train_data[:,i])
    if np.isnan(item_avg[i])== True:
        item_avg[i]=1

###########################################  avg replace
    
  
v4=copy.deepcopy(v3)
for i in range(0, len(v3)):
    for j in range(0, len(v3[0])):
        if v3[i,j]>0:
            v4[i,j]=item_avg[j]

mf_interm = MF_NEW(v3, K=64, alpha=0.03, beta=0.01, iterations=150)
mf_interm.train()

user_m3=mf_interm.P
item_m3=mf_interm.Q
result3=mf_interm.full_matrix_new(user_m3, item_m3)

predicted_result3 = np.ma.masked_array(result3, test_mask) 
diff_test_intm= predicted_result3-(masked_test_data)
diff1_test_intm=np.abs(diff_test_intm)
MAE_test_intm=np.mean(diff1_test_intm)
print("Performance of intermediate model")
print(MAE_test_intm)
RMSE_test_intm=np.sqrt(np.mean(np.square(diff_test_intm)))
print(RMSE_test_intm)



net =ct.ConvTransfer(64,64)
a1=torch.tensor(user_m1).float()
a2=torch.tensor(user_m3).float()
y_1= net(a1,a2,"user")


a11=torch.tensor(item_m1).float()
a21=torch.tensor(item_m3).float()
y_2= net(a11,a21,"item")

y_11=y_1.detach().numpy()
y_12=y_2.detach().numpy()

res=mf_orig.full_matrix_new(y_11,y_12)
predicted_result_conv = np.ma.masked_array(res, test_mask) 




diff_test_conv= predicted_result_conv-(masked_test_data)
diff1_test_conv=np.abs(diff_test_conv)
MAE_test_conv=np.mean(diff1_test_conv)
print("Hello! this is your MAE after using regularizer with original ratings")
print(MAE_test_conv)
print("Hello! this is your RMSE after using regularizer with original ratings")
RMSE_test_conv=np.sqrt(np.mean(np.square(diff_test_conv)))
print(RMSE_test_conv)
print(f"final model RMSE {RMSE_test_conv} V3 RMSE")
print(f"Intermediate model RMSE {RMSE_test_intm} V3 RMSE")

mf_interm4 = MF_NEW(v4, K=64, alpha=0.03, beta=0.01, iterations=150)
mf_interm4.train()

user_m4=mf_interm4.P
item_m4=mf_interm4.Q
result4=mf_interm4.full_matrix_new(user_m4, item_m4)

predicted_result4 = np.ma.masked_array(result4, test_mask) 
diff_test_intm4= predicted_result4-(masked_test_data)
diff1_test_intm4=np.abs(diff_test_intm4)
MAE_test_intm4=np.mean(diff1_test_intm4)
print("Performance of intermediate model")
print(MAE_test_intm4)
RMSE_test_intm4=np.sqrt(np.mean(np.square(diff_test_intm4)))
print(RMSE_test_intm4)



a14=torch.tensor(user_m1).float()
a24=torch.tensor(user_m4).float()
y_14= net(a14,a24,"user")


a114=torch.tensor(item_m1).float()
a214=torch.tensor(item_m4).float()
y_24= net(a114,a214,"item")

y_114=y_14.detach().numpy()
y_124=y_24.detach().numpy()

res4=mf_orig.full_matrix_new(y_114,y_124)
predicted_result_conv4 = np.ma.masked_array(res4, test_mask) 




diff_test_conv4= predicted_result_conv4-(masked_test_data)
diff1_test_conv4=np.abs(diff_test_conv4)
MAE_test_conv4=np.mean(diff1_test_conv4)
print("Hello! this is your MAE after using regularizer with Avg ratings")
print(MAE_test_conv4)
print("Hello! this is your RMSE after using regularizer with Avg ratings")
RMSE_test_conv4=np.sqrt(np.mean(np.square(diff_test_conv4)))
print(RMSE_test_conv4)
print(f"final model RMSE {RMSE_test_conv4} RMSE V4")
print(f"Intermediate model RMSE {RMSE_test_intm4} RMSE V4")
