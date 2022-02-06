import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

  # read the first training image
  folder = 'data/training'

  for i in range(1, 3):

    token = str(i)
    fname_img = '0' * (4 - len(token)) + token + '.jpg'
    img = cv2.imread(os.path.join(folder,fname_img))
    # can be changed to other color space, eg. COLOR_BGR2HSV, but please keep it consecutive and consistent
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    m, n, _ = img.shape

    judge = True
    
    while judge:

        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')
        
        # get the image mask
        mask = my_roi.get_mask(img)
        
        # get the y value for this part of training set
        val_1 = input('Is this a positive example (enter 1) or a negative example (enter 0): ')
        response_1 = ['1', '0']
        while val_1 not in response_1: val_1 = input('Invalid input. Please enter 1 for positive, 0 for negative: ')
        if val_1 == '1': y = np.array([1])
        else: y = np.array([0])
        
        # write data into csv file in format [r, g, b, y]
        fname_csv = os.path.join(folder, '0' * (4 - len(token)) + token + '.csv')
        with open(fname_csv, 'w') as f:
            for i in range(m):
                for j in range(n):
                    if mask[i, j]: np.savetxt(f, np.concatenate((img[i, j, :], y)).reshape(1, -1), delimiter = ',', newline = '\n')
        f.close()
        
        # continue to get data from this dataset if needed
        val_2 = input('Are you done with working on this dataset? (Yes or No) ')
        response_2 = ['Yes', 'yes', 'YES', 'y', 'Y', 'No', 'no', 'NO', 'n', 'N']
        while val_1 not in response_1: val_1 = input('Invalid input. Please enter yes or no: ')
        if val_2 in response_2[: 5]: judge = False
        else: judge = True
    
    # test loading data as numpy array
    curr_arr = np.genfromtxt(fname_csv, delimiter = ',')
    print (curr_arr.shape)


       