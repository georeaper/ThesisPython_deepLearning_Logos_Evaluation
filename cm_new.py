from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import mean_absolute_error

Actual=[2,1,2,0,3,0,0,4,0,2,1,1,0,0,2,1,2,3,0,1,2,1,0,3,2,1,1,3,4,3,2,2,2,3,0,4,0,3,1,1,1,2,3,2,2,2]
Predicted=[2,1,4,2,3,3,2,0,1,4,0,1,1,0,1,3,2,1,2,1,1,1,0,3,3,1,1,1,4,1,2,0,1,1,2,4,2,1,2,3,3,2,3,4,2,2]

cm_valid = confusion_matrix(Actual,Predicted, labels=[0, 1, 2, 3, 4, 5, 6])
print("confusion_matrix upcoming for validation")
print(cm_valid)
mae=mean_absolute_error(Actual, Predicted)
print("MAE is :",mae)

Actual=[4,1,2,4,1,2,1,0,1,1,2,2,1,0,0,1,1,2,1,1,2]
Predicted=[2,1,2,3,1,2,2,1,3,1,1,3,1,1,3,1,1,1,2,1,1]

cm_valid = confusion_matrix(Actual,Predicted, labels=[0, 1, 2, 3, 4, 5, 6])
print("confusion_matrix upcoming for validation")
print(cm_valid)
mae=mean_absolute_error(Actual, Predicted)
print("MAE is :",mae)