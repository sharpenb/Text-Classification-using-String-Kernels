import numpy as np
from sklearn.svm import SVC
import math



class StringSVM:
    def __init__(self, kernel):

        self.kernel = kernel

        # initiate sklearn svm
        self.svm = SVC(kernel = self.string_kernel)

        # data counter
        self.data_count = 0
    
        # data base
        self.x_data = {}
        self.y_data = {}
        self.kernel_vals = {}

    def set_params(self, params):
        self.svm.set_params(params)
        
    def string_kernel(self, X1, X2):
        R = np.zeros((len(X1), len(X2)))

        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                k = int(X1[i][0])
                l = int(X2[j][0])


                if (k, l) in self.kernel_vals:
                    v = self.kernel_vals[(k, l)]
                else:
                    v = self.kernel(self.x_data[k], self.x_data[l])
                    
                    # store kernel values, assumes symmetric
                    self.kernel_vals[(k, l)] = v
                    self.kernel_vals[(l, k)] = v

                if (k, k) in self.kernel_vals:
                    n1 = self.kernel_vals[(k, k)]
                else:
                    n1 = self.kernel(self.x_data[k], self.x_data[k])
                    self.kernel_vals[(k, k)] = n1
                
                if (l, l) in self.kernel_vals:
                    n2 = self.kernel_vals[(l, l)]
                else:
                    n2 = self.kernel(self.x_data[l], self.x_data[l])
                    self.kernel_vals[(l, l)] = n2

                R[i, j] = v / np.sqrt(n1 * n2)


        return R


    def add_data(self, new_x_data, new_y_data):
        # input to svm 
        x_in = []

        # enumerate and store data in data base
        for i in range(len(new_x_data)):
            x = new_x_data[i]
            y = new_y_data[i]
            x_in.append(self.data_count)
            self.x_data[self.data_count] = x
            self.y_data[self.data_count] = y
            self.data_count += 1

        return x_in
        

    def get_chunks(self, x, y, n_points):
        
        x_chunks = []
        y_chunks = []

        for i in range(0, len(x), n_points):
            x_chunks.append(x[i : i + n_points])
            y_chunks.append(y[i : i + n_points])
        
        return x_chunks, y_chunks
    

    def recursive_fit(self, x_train, y_train, n_points):
        
        # input to svm 
        x_in = self.add_data(x_train, y_train)

        # divide set into chunks
        x_chunks, y_chunks = self.get_chunks(x_in, y_train, n_points)

        
        # make sure first set includes both
        # positive and negative data point
        for i in range(len(x_chunks)):
            y = y_chunks[i]

            found_negative = False
            found_positive = False
            for label in y:
                if label == 1:
                    found_positive = True
                elif label == 0:
                    found_negative = True
            
            if found_positive and found_negative:
                if i != 0:
                    x_chunks[0], x_chunks[i] = x_chunks[i], x_chunks[0]
                    y_chunks[0], y_chunks[i] = y_chunks[i], y_chunks[0]
                break

        x = []
        y = []
        
        for i in range(len(x_chunks)):
            print("chunk:", i)
            
            x = np.append(x, x_chunks[i])
            y = np.append(y, y_chunks[i])
                
            x_in = np.array(x).reshape(-1, 1)
            y_in = y

            self.svm.fit(x_in, y_in)

            support_vec_ind = self.svm.support_


            x = []
            y = []
            for j in support_vec_ind:
                _x = x_in[j][0]
                x.append(_x)
                y.append(self.y_data[_x])
            
            print(x)


    def fit(self, x_train, y_train):
        
        # input to svm 
        x_in = self.add_data(x_train, y_train)

        x_in = np.array(x_in).reshape(-1, 1)
        
        self.svm.fit(x_in, y_train)



    def predict(self, x_test, y_test):
        # input to svm 
        x_in = self.add_data(x_test, y_test)
        
        x_in = np.array(x_in).reshape(-1, 1)
        
        return self.svm.predict(x_in)

        
        
         
